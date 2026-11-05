import asyncio
import json
import socket
import sys
from pathlib import Path

import aiohttp
import pytest

from ratatosk.db import Db
from ratatosk.engine import Engine
from ratatosk.gateway_llm import SESSION, AppContext, make_llm_app, parse_reply
from ratatosk.gpuqueue import GpuQueue
from ratatosk.models import Model
from ratatosk.scheduler import Scheduler
from ratatosk.settings import Route, Settings

FAKE = str(Path(__file__).parent / "fakes" / "fake_llama.py")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
async def rig(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "fake.log"))
    port = free_port()
    settings = Settings(engine_port=port, data_dir=str(tmp_path), default_model="alpha", max_tokens_cap=1024,
                        max_tokens_cap_plain=1024, aliases={"qwen3.5-9b-obliterated": "alpha"})
    alpha = tmp_path / "alpha.gguf"
    alpha.write_bytes(b"x")
    models = [Model("alpha", alpha, 1, tmp_path)]
    engine = Engine(settings, poll_seconds=0.1,
                    argv_override=lambda m, c: [sys.executable, FAKE, "--model", str(m), "--ctx-size", str(c), "--port", str(port)])
    queue = GpuQueue()
    queue.start()

    class Rig:
        pass

    r = Rig()
    r.settings, r.models, r.engine, r.queue = settings, models, engine, queue
    r.db = Db(tmp_path / "r.db")
    r.scheduler = Scheduler(settings, queue, engine, comfy=None)
    r.ctx = AppContext(settings, r.scheduler, engine, r.db, lambda: r.models)
    r.log = lambda: [json.loads(line) for line in (tmp_path / "fake.log").read_text().splitlines()] \
        if (tmp_path / "fake.log").exists() else []
    yield r
    await queue.stop()
    await engine.stop()


async def client_for(aiohttp_client, rig, profile="plain"):
    return await aiohttp_client(make_llm_app(Route(11434, "llm", profile), rig.ctx), server_kwargs={"handler_cancellation": True})


async def test_completion_streams_through(rig, aiohttp_client):
    client = await client_for(aiohttp_client, rig)
    resp = await client.post("/v1/completions", json={"model": "alpha", "prompt": "hi there", "stream": True})
    text = await resp.text()
    assert resp.status == 200 and "t0" in text and "t2" in text and "[DONE]" in text


async def test_max_tokens_capped(rig, aiohttp_client):
    client = await client_for(aiohttp_client, rig)
    await (await client.post("/v1/completions", json={"model": "alpha", "prompt": "hi", "max_tokens": 65536})).read()
    sent = [e for e in rig.log() if e["event"] == "request"][0]["body"]
    assert sent["max_tokens"] == 1024


async def test_unknown_model_404_no_default(rig, aiohttp_client):
    rig.settings.default_model = ""
    client = await client_for(aiohttp_client, rig)
    resp = await client.post("/v1/completions", json={"model": "nope", "prompt": "hi"})
    assert resp.status == 404 and "error" in await resp.json()


async def test_missing_file_clear_error(rig, aiohttp_client):
    rig.models = [Model("alpha", rig.models[0].path.parent / "gone" / "alpha.gguf", 1, rig.models[0].folder)]
    client = await client_for(aiohttp_client, rig)
    resp = await client.post("/v1/completions", json={"model": "alpha", "prompt": "hi"})
    assert resp.status == 503 and "not mounted" in (await resp.json())["error"]
    assert not any(e["event"] == "start" for e in rig.log())


async def test_oversized_prompt_413_engine_untouched(rig, aiohttp_client):
    client = await client_for(aiohttp_client, rig)
    resp = await client.post("/v1/completions", json={"model": "alpha", "prompt": "a " * 600_000})
    assert resp.status == 413
    assert "tokens of context" in (await resp.json())["error"]  # our own message, not aiohttp's body-size one
    assert rig.engine.pid is None and rig.log() == []


async def test_models_endpoint_lists_aliases_and_ids(rig, aiohttp_client):
    client = await client_for(aiohttp_client, rig)
    ids = [m["id"] for m in (await (await client.get("/v1/models")).json())["data"]]
    assert "alpha" in ids and "qwen3.5-9b-obliterated" in ids


async def test_client_disconnect_cancels_and_next_request_runs(rig, aiohttp_client):
    client = await client_for(aiohttp_client, rig)
    first = await client.post("/v1/completions", json={"model": "alpha", "prompt": "hi", "stream": True, "slow": 5})
    await first.content.readany()
    first.close()  # the player closes the tab mid-reply
    # about 10 s of the first reply were left; if its job were not cancelled the second request would wait for them
    second = await asyncio.wait_for(client.post("/v1/completions", json={"model": "alpha", "prompt": "hi"}), 3)
    assert second.status == 200


async def wait_for_request(rig, prompt):
    for _ in range(200):
        if any(e["event"] == "request" and e["body"]["prompt"] == prompt for e in rig.log()):
            return
        await asyncio.sleep(0.1)
    raise AssertionError(f"the engine never saw {prompt!r}")


async def test_disconnect_while_queued_never_runs(rig, aiohttp_client):
    client = await client_for(aiohttp_client, rig)
    blocker = asyncio.create_task(client.post("/v1/completions", json={"model": "alpha", "prompt": "blocker", "stream": True, "slow": 0.4}))
    await wait_for_request(rig, "blocker")
    victim = asyncio.create_task(client.post("/v1/completions", json={"model": "alpha", "prompt": "victim"}))
    await asyncio.sleep(0.2)
    victim.cancel()  # the player gave up while waiting in the queue
    await (await blocker).read()
    await (await client.post("/v1/completions", json={"model": "alpha", "prompt": "after"})).read()
    assert "victim" not in [e["body"]["prompt"] for e in rig.log() if e["event"] == "request"]


@pytest.mark.parametrize("value", [-1, 0, "abc", None, 5_000_000])
async def test_bad_or_huge_max_tokens_fall_back_to_the_cap(rig, aiohttp_client, value):
    client = await client_for(aiohttp_client, rig)
    resp = await client.post("/v1/completions", json={"model": "alpha", "prompt": "hi", "max_tokens": value})
    await resp.read()
    assert resp.status == 200
    assert [e for e in rig.log() if e["event"] == "request"][0]["body"]["max_tokens"] == 1024


async def test_context_reserves_room_for_a_typical_reply_not_the_whole_cap(rig, aiohttp_client):
    # a 32k context halves the text speed on this GPU (25.1 -> about 14 tok/s), so the size must not be inflated
    rig.settings.max_tokens_cap_plain, rig.settings.reply_reserve = 8192, 2048
    client = await client_for(aiohttp_client, rig)
    await (await client.post("/v1/completions", json={"model": "alpha", "prompt": "hello"})).read()
    assert rig.db.recent(1)[0]["ctx"] == 8192  # reserving the whole 8192 would have forced 16384


async def test_a_dozen_camera_sized_pictures_fit_in_a_16k_context(rig, aiohttp_client):
    rig.models = [Model("alpha", rig.models[0].path, 1, rig.models[0].folder, rig.models[0].path)]
    rig.settings.max_tokens_cap_plain, rig.settings.reply_reserve = 8192, 2048
    assert rig.settings.image_max_tokens == 512  # measured: a real camera frame costs about 500 tokens
    client = await client_for(aiohttp_client, rig)
    await (await client.post("/v1/chat/completions", json=picture_request(12))).read()
    assert rig.db.recent(1)[0]["ctx"] == 16384  # about 7k tokens of request in all, as LM Studio's log showed


async def test_request_defaults_fill_in_only_what_the_client_left_out(rig, aiohttp_client):
    rig.settings.request_defaults = {"alpha": {"temperature": 0.3, "chat_template_kwargs": {"enable_thinking": False}},
                                     "qwen3.5-9b-obliterated": {"top_p": 0.5}}
    client = await client_for(aiohttp_client, rig)
    await (await client.post("/v1/completions", json={"model": "alpha", "prompt": "a"})).read()
    await (await client.post("/v1/completions", json={"model": "alpha", "prompt": "b", "temperature": 0.9,
                                                      "chat_template_kwargs": {"x": 1}})).read()
    await (await client.post("/v1/completions", json={"model": "qwen3.5-9b-obliterated", "prompt": "c"})).read()
    sent = {e["body"]["prompt"]: e["body"] for e in rig.log() if e["event"] == "request"}
    assert sent["a"]["temperature"] == 0.3 and sent["a"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert sent["b"]["temperature"] == 0.9  # what the client sets wins
    assert sent["b"]["chat_template_kwargs"] == {"enable_thinking": False, "x": 1}  # and settings inside it are merged
    assert sent["c"]["top_p"] == 0.5 and sent["c"]["temperature"] == 0.3  # a default for the name the client used, plus the model's own
    assert "top_p" not in sent["a"]  # a default for another name is not applied


def test_the_plain_route_allows_longer_replies_than_the_roleplay_route_by_default():
    # a thinking model spends over a thousand tokens before it answers, so the plain cap must leave room for that
    assert Settings().max_tokens_cap_plain >= 4 * Settings().max_tokens_cap


@pytest.mark.parametrize("profile, asked, allowed", [("plain", 65536, 4000), ("plain", None, 4000), ("rp", 65536, 1000), ("rp", None, 1000)])
async def test_each_route_has_its_own_reply_cap(rig, aiohttp_client, profile, asked, allowed):
    rig.settings.max_tokens_cap, rig.settings.max_tokens_cap_plain = 1000, 4000
    client = await client_for(aiohttp_client, rig, profile)
    body = {"model": "alpha", "prompt": "hi", **({"max_tokens": asked} if asked else {})}
    await (await client.post("/v1/completions", json=body)).read()
    assert [e for e in rig.log() if e["event"] == "request"][0]["body"]["max_tokens"] == allowed


async def test_max_completion_tokens_is_capped_too(rig, aiohttp_client):
    client = await client_for(aiohttp_client, rig)
    await (await client.post("/v1/chat/completions", json={
        "model": "alpha", "messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 65536})).read()
    sent = [e for e in rig.log() if e["event"] == "request"][0]["body"]
    assert sent["max_tokens"] == 1024 and sent.get("max_completion_tokens", 0) <= 1024


async def test_chat_message_with_null_content_is_accepted(rig, aiohttp_client):
    client = await client_for(aiohttp_client, rig)
    resp = await client.post("/v1/chat/completions", json={"model": "alpha", "messages": [
        {"role": "assistant", "content": None, "tool_calls": []}, {"role": "user", "content": "hi"}]})
    assert resp.status == 200


async def test_body_that_is_not_an_object_is_400(rig, aiohttp_client):
    client = await client_for(aiohttp_client, rig)
    assert (await client.post("/v1/completions", json=[1, 2])).status == 400


async def test_replies_may_stream_for_longer_than_any_total_timeout(rig, aiohttp_client):
    client = await client_for(aiohttp_client, rig)
    assert client.app[SESSION].timeout.total is None  # the default of 300 s would cut a slow 1000-token reply


async def test_engine_dying_mid_reply_is_logged_as_an_error(rig, aiohttp_client):
    client = await client_for(aiohttp_client, rig)
    resp = await client.post("/v1/completions", json={"model": "alpha", "prompt": "hi", "stream": True, "drop": True})
    try:
        await resp.read()
    except aiohttp.ClientError:
        pass  # the client sees a cut stream, which is all that can be done once the reply has started
    for _ in range(30):
        if rig.db.recent(1):
            break
        await asyncio.sleep(0.1)
    assert rig.db.recent(1)[0]["status"] == "error"


async def test_engine_that_cannot_start_gives_502(rig, aiohttp_client, monkeypatch):
    monkeypatch.setenv("FAKE_EXIT", "1")
    client = await client_for(aiohttp_client, rig)
    resp = await client.post("/v1/completions", json={"model": "alpha", "prompt": "hi"})
    assert resp.status == 502 and "error" in await resp.json()
    assert rig.db.recent(1)[0]["status"] == "error" and rig.db.recent(1)[0]["error"]  # the reason is kept for the page


async def test_rp_profile_runs_before_plain(rig, aiohttp_client):
    plain = await client_for(aiohttp_client, rig, "plain")
    rp = await client_for(aiohttp_client, rig, "rp")
    blocker = asyncio.create_task(plain.post("/v1/completions", json={"model": "alpha", "prompt": "blocker", "stream": True, "slow": 0.4}))
    await asyncio.sleep(0.5)
    a = asyncio.create_task(plain.post("/v1/completions", json={"model": "alpha", "prompt": "plainjob"}))
    await asyncio.sleep(0.1)
    b = asyncio.create_task(rp.post("/v1/completions", json={"model": "alpha", "prompt": "rpjob"}))
    await asyncio.gather(blocker, a, b)
    order = [e["body"]["prompt"] for e in rig.log() if e["event"] == "request"]
    assert order.index("rpjob") < order.index("plainjob")


async def test_request_logged(rig, aiohttp_client):
    client = await client_for(aiohttp_client, rig, "rp")
    await (await client.post("/v1/completions", json={"model": "alpha", "prompt": "one two three"})).read()
    row = rig.db.recent(1)[0]
    assert (row["route"], row["profile"], row["model"], row["status"]) == ("11434", "rp", "alpha", "ok")
    assert row["prompt_tokens"] == len("one two three") // 3  # the engine was not running yet, so this is the estimate
    assert row["ctx"] == 8192 and row["seconds"] >= 0


async def test_request_detail_records_origin_content_and_timing(rig, aiohttp_client):
    client = await client_for(aiohttp_client, rig, "rp")
    await (await client.post("/v1/completions", json={"model": "alpha", "prompt": "hello there", "max_tokens": 65536},
                             headers={"User-Agent": "surveillance/1.0"})).read()
    d = rig.db.get(rig.db.recent(1)[0]["id"])
    assert (d["client"], d["user_agent"], d["path"], d["stream"]) == ("127.0.0.1", "surveillance/1.0", "/v1/completions", 0)
    assert (d["max_tokens_asked"], d["max_tokens"]) == (65536, 1024)  # what was asked, and what the cap allowed
    assert "hello there" in d["prompt_text"] and d["reply_text"] == "hello"
    assert d["gen_tokens"] == 1 and d["queue_seconds"] >= 0 and d["error"] is None


async def test_streamed_reply_is_recorded_as_text(rig, aiohttp_client):
    client = await client_for(aiohttp_client, rig)
    await (await client.post("/v1/completions", json={"model": "alpha", "prompt": "hi", "stream": True})).read()
    d = rig.db.get(rig.db.recent(1)[0]["id"])
    assert d["stream"] == 1 and d["reply_text"] == "t0t1t2"


async def test_images_pass_through_but_are_stored_as_a_placeholder(rig, aiohttp_client):
    rig.models = [Model("alpha", rig.models[0].path, 1, rig.models[0].folder, rig.models[0].path)]  # a model that can read pictures
    client = await client_for(aiohttp_client, rig)
    frame = "data:image/jpeg;base64," + "A" * 5000  # a camera frame
    await (await client.post("/v1/chat/completions", json={"model": "alpha", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "what is in this picture"}, {"type": "image_url", "image_url": {"url": frame}}]}]})).read()
    d = rig.db.get(rig.db.recent(1)[0]["id"])
    assert "what is in this picture" in d["prompt_text"] and "[image]" in d["prompt_text"]
    assert "AAAA" not in d["prompt_text"]  # the frame itself is never kept
    sent = [e for e in rig.log() if e["event"] == "request"][0]["body"]
    assert sent["messages"][0]["content"][1]["image_url"]["url"] == frame  # and the engine still gets it whole


def picture_request(count):
    frame = "data:image/jpeg;base64," + "A" * 200
    parts = [{"type": "text", "text": "what is this"}] + [{"type": "image_url", "image_url": {"url": frame}}] * count
    return {"model": "alpha", "messages": [{"role": "user", "content": parts}]}


async def test_pictures_count_toward_the_context_the_request_needs(rig, aiohttp_client):
    rig.models = [Model("alpha", rig.models[0].path, 1, rig.models[0].folder, rig.models[0].path)]  # a model that can read pictures
    rig.settings.image_max_tokens = 3000
    client = await client_for(aiohttp_client, rig)
    await (await client.post("/v1/chat/completions", json=picture_request(0))).read()
    await (await client.post("/v1/chat/completions", json=picture_request(3))).read()
    contexts = [row["ctx"] for row in rig.db.recent(2)][::-1]
    assert contexts == [8192, 16384]  # three pictures at up to 3000 tokens each do not fit in 8192


async def test_pictures_for_a_model_without_a_vision_file_are_refused_clearly(rig, aiohttp_client):
    client = await client_for(aiohttp_client, rig)  # the rig's model has no vision file
    resp = await client.post("/v1/chat/completions", json=picture_request(1))
    assert resp.status == 400 and "cannot read pictures" in (await resp.json())["error"]
    assert rig.engine.pid is None and rig.log() == []  # nothing was started for it
    text_only = await client.post("/v1/chat/completions", json=picture_request(0))
    assert text_only.status == 200


def test_parse_reply_reads_text_tokens_and_speed_from_a_stream():
    raw = (b'data: {"choices":[{"text":"a"}]}\n\n'
           b'data: {"choices":[{"text":"b"}],"timings":{"predicted_n":2,"predicted_per_second":25.5}}\n\n'
           b'data: [DONE]\n\n')
    assert parse_reply(raw, "text/event-stream") == ("ab", 2, 25.5)


def test_parse_reply_reads_a_chat_reply_and_survives_garbage():
    raw = b'{"choices":[{"message":{"content":"hi"}}],"usage":{"completion_tokens":3}}'
    assert parse_reply(raw, "application/json") == ("hi", 3, None)
    assert parse_reply(b"not json at all", "application/json") == ("", None, None)


def test_parse_reply_reads_the_native_anthropic_and_responses_shapes():
    native = (b'data: {"content":"a","stop":false}\n\n'
              b'data: {"content":"b","stop":true,"timings":{"predicted_n":2,"predicted_per_second":9.5}}\n\n')
    assert parse_reply(native, "text/event-stream") == ("ab", 2, 9.5)
    assert parse_reply(b'{"content":"hi","tokens_predicted":4}', "application/json") == ("hi", 4, None)
    assert parse_reply(b'{"content":[{"type":"text","text":"hey"}],"usage":{"output_tokens":5}}', "application/json") == ("hey", 5, None)
    stream = (b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"x"}}\n\n'
              b'event: message_delta\ndata: {"type":"message_delta","usage":{"output_tokens":7}}\n\n')
    assert parse_reply(stream, "text/event-stream") == ("x", 7, None)
    responses = b'{"output":[{"type":"message","content":[{"type":"output_text","text":"yo"}]}],"usage":{"output_tokens":2}}'
    assert parse_reply(responses, "application/json") == ("yo", 2, None)


def sent_to_engine(rig, path):
    return [e["body"] for e in rig.log() if e["event"] == "request" and e["path"] == path]


async def test_native_completion_is_queued_capped_and_logged(rig, aiohttp_client):
    client = await client_for(aiohttp_client, rig)  # llama.cpp's own API, as SillyTavern's llama.cpp type uses it
    resp = await client.post("/completion", json={"prompt": "hello there", "n_predict": 65536})
    assert resp.status == 200 and (await resp.json())["content"] == "hello"
    assert sent_to_engine(rig, "/completion")[0]["n_predict"] == 1024
    row = rig.db.get(rig.db.recent(1)[0]["id"])
    assert row["path"] == "/completion" and "hello there" in row["prompt_text"] and row["reply_text"] == "hello"
    assert (row["gen_tokens"], row["status"]) == (1, "ok")


async def test_native_completion_streams_through(rig, aiohttp_client):
    client = await client_for(aiohttp_client, rig)
    text = await (await client.post("/completion", json={"prompt": "hi", "stream": True})).text()
    assert '"content": "t0"' in text and '"stop": true' in text
    row = rig.db.get(rig.db.recent(1)[0]["id"])
    assert (row["reply_text"], row["gen_tokens"], row["tokens_per_second"]) == ("t0t1t2", 3, 25.0)


async def test_infill_is_served_and_its_reply_length_capped_even_when_not_asked(rig, aiohttp_client):
    client = await client_for(aiohttp_client, rig)
    resp = await client.post("/infill", json={"input_prefix": "def f(", "input_suffix": "):"})
    assert resp.status == 200
    assert sent_to_engine(rig, "/infill")[0]["n_predict"] == 1024
    assert "def f(" in rig.db.get(rig.db.recent(1)[0]["id"])["prompt_text"]


async def test_anthropic_messages_are_served_and_logged(rig, aiohttp_client):
    client = await client_for(aiohttp_client, rig)
    body = {"model": "claude-sonnet-x", "max_tokens": 9999, "system": "be brief",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi there"}]}]}
    resp = await client.post("/v1/messages", json=body)
    assert resp.status == 200 and (await resp.json())["content"][0]["text"] == "hello"
    assert sent_to_engine(rig, "/v1/messages")[0]["max_tokens"] == 1024  # and the unknown model name fell back to the default model
    row = rig.db.get(rig.db.recent(1)[0]["id"])
    assert "be brief" in row["prompt_text"] and "hi there" in row["prompt_text"]
    assert (row["reply_text"], row["gen_tokens"]) == ("hello", 1)


async def test_anthropic_image_blocks_count_as_pictures(rig, aiohttp_client):
    client = await client_for(aiohttp_client, rig)  # the rig's model has no vision file
    block = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}
    body = {"model": "alpha", "max_tokens": 10, "messages": [{"role": "user", "content": [{"type": "text", "text": "what"}, block]}]}
    resp = await client.post("/v1/messages", json=body)
    assert resp.status == 400 and "cannot read pictures" in (await resp.json())["error"]


async def test_responses_endpoint_is_served_and_capped(rig, aiohttp_client):
    client = await client_for(aiohttp_client, rig)
    resp = await client.post("/v1/responses", json={"model": "alpha", "input": "hi there", "max_output_tokens": 99999})
    assert resp.status == 200
    assert sent_to_engine(rig, "/v1/responses")[0]["max_output_tokens"] == 1024
    row = rig.db.get(rig.db.recent(1)[0]["id"])
    assert "hi there" in row["prompt_text"] and row["reply_text"] == "hello"


async def test_tokenize_starts_the_engine_when_there_is_none(rig, aiohttp_client):
    client = await client_for(aiohttp_client, rig)
    assert rig.engine.pid is None
    resp = await client.post("/tokenize", json={"content": "one two three"})
    assert resp.status == 200 and (await resp.json())["tokens"] == [0, 0, 0]
    pid = rig.engine.pid
    assert pid is not None
    await client.post("/tokenize", json={"content": "again"})
    assert rig.engine.pid == pid  # the second call used the engine that was already running


async def test_template_detokenize_and_slots_are_passed_on(rig, aiohttp_client):
    client = await client_for(aiohttp_client, rig)
    assert (await (await client.post("/apply-template", json={"messages": []})).json())["prompt"] == "templated"
    assert (await (await client.post("/detokenize", json={"tokens": [1]})).json())["content"] == "x"
    assert await (await client.get("/slots")).json() == [{"id": 0}]


async def test_props_and_health_answer_without_loading_a_model(rig, aiohttp_client):
    client = await client_for(aiohttp_client, rig)
    props = await (await client.get("/props")).json()
    assert props["default_generation_settings"]["n_ctx"] >= 8192 and props["model_path"].endswith("alpha.gguf")
    assert await (await client.get("/health")).json() == {"status": "ok"}
    assert rig.engine.pid is None  # neither woke the engine


async def test_unsupported_paths_are_refused_clearly_and_show_up_in_the_log(rig, aiohttp_client):
    client = await client_for(aiohttp_client, rig)
    resp = await client.post("/v1/images/generations", json={"prompt": "cat"})
    assert resp.status == 404 and "error" in await resp.json()
    embeddings = await client.post("/v1/embeddings", json={"input": "hi"})
    assert embeddings.status == 501 and "embedding" in (await embeddings.json())["error"]
    newest, older = rig.db.recent(2)
    assert (newest["status"], older["status"]) == ("unsupported", "unsupported")
    assert older["path"] == "/v1/images/generations" and older["error"]  # nothing is silent any more
    assert rig.engine.pid is None


async def client_with_agent(aiohttp_client, rig, profile, agent):
    server = make_llm_app(Route(11434, "llm", profile), rig.ctx)
    return await aiohttp_client(server, server_kwargs={"handler_cancellation": True},
                                headers={"User-Agent": agent})


async def test_a_client_line_moves_its_request_ahead_of_others_in_the_queue(rig, aiohttp_client):
    rig.settings.priority = [{"client": "127.0.0.1", "agent": "vip"}]
    plain = await client_for(aiohttp_client, rig)
    vip = await client_with_agent(aiohttp_client, rig, "plain", "vip")
    blocker = asyncio.create_task(plain.post("/v1/completions", json={"model": "alpha", "prompt": "blocker", "stream": True, "slow": 0.4}))
    await wait_for_request(rig, "blocker")
    a = asyncio.create_task(plain.post("/v1/completions", json={"model": "alpha", "prompt": "ordinary"}))
    await asyncio.sleep(0.1)
    b = asyncio.create_task(vip.post("/v1/completions", json={"model": "alpha", "prompt": "important"}))
    await asyncio.gather(blocker, a, b)
    order = [e["body"]["prompt"] for e in rig.log() if e["event"] == "request"]
    assert order.index("important") < order.index("ordinary")


async def test_a_drop_line_cuts_off_the_running_reply_and_the_log_says_who(rig, aiohttp_client):
    rig.settings.priority = [{"client": "127.0.0.1", "agent": "boss", "drop": True, "label": "boss"}]
    plain = await client_for(aiohttp_client, rig)
    boss = await client_with_agent(aiohttp_client, rig, "plain", "boss")
    victim = await plain.post("/v1/completions", json={"model": "alpha", "prompt": "victim", "stream": True, "slow": 5})
    await victim.content.readany()
    urgent = await asyncio.wait_for(boss.post("/v1/completions", json={"model": "alpha", "prompt": "urgent"}), 5)
    assert urgent.status == 200
    with pytest.raises(aiohttp.ClientError):  # the half-sent reply is cut off, not finished
        await asyncio.wait_for(victim.read(), 3)
    for _ in range(50):
        rows = [r for r in rig.db.recent(10) if r["status"] == "dropped"]
        if rows:
            break
        await asyncio.sleep(0.1)
    assert rows and rows[0]["error"] == "dropped by boss"


async def test_without_a_priority_list_the_old_order_stands(rig, aiohttp_client):
    assert rig.settings.priority is None  # test_rp_profile_runs_before_plain covers the behaviour itself
    from ratatosk.priority import rank_request
    assert rank_request(rig.settings, Route(11435, "llm", "rp"), "1.1.1.1", "").priority < rank_request(
        rig.settings, Route(11434, "llm", "plain"), "1.1.1.1", "").priority < rank_request(
        rig.settings, Route(8188, "comfy"), "1.1.1.1", "").priority


async def client_allowing(aiohttp_client, rig, **allow):
    route = Route(11434, "llm", "plain", allow={"vision": True, "reasoning": True, "tools": True, **allow})
    rig.settings.routes = [route]  # doors come from the settings, as in the running service
    return await aiohttp_client(make_llm_app(route, rig.ctx), server_kwargs={"handler_cancellation": True})


async def test_vision_off_on_a_door_refuses_pictures(rig, aiohttp_client):
    client = await client_allowing(aiohttp_client, rig, vision=False)
    resp = await client.post("/v1/chat/completions", json=picture_request(1))
    assert resp.status == 400 and "turned off" in (await resp.json())["error"]
    assert rig.log() == []  # the engine was never started for it


async def test_tools_off_on_a_door_removes_every_tool_field(rig, aiohttp_client):
    client = await client_allowing(aiohttp_client, rig, tools=False)
    body = {"model": "alpha", "messages": [{"role": "user", "content": "hi"}], "tools": [{"type": "function"}],
            "tool_choice": "auto", "functions": [{}], "function_call": "auto"}
    assert (await client.post("/v1/chat/completions", json=body)).status == 200
    sent = sent_to_engine(rig, "/v1/chat/completions")[0]
    assert not {"tools", "tool_choice", "functions", "function_call"} & set(sent)


async def test_reasoning_off_on_a_door_overrides_what_the_client_asked_for(rig, aiohttp_client):
    client = await client_allowing(aiohttp_client, rig, reasoning=False)
    body = {"model": "alpha", "messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": {"enable_thinking": True, "other": 1}}
    assert (await client.post("/v1/chat/completions", json=body)).status == 200
    assert sent_to_engine(rig, "/v1/chat/completions")[0]["chat_template_kwargs"] == {"enable_thinking": False, "other": 1}


def test_reasoning_off_is_left_alone_for_a_model_that_cannot_switch_it():
    from ratatosk.gateway_llm import apply_allow
    body = {"messages": []}
    assert apply_allow(body, {"reasoning": False}, can_reason=False) is None and "chat_template_kwargs" not in body
    assert apply_allow(body, {"reasoning": False}, can_reason=True) is None
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


async def test_door_settings_saved_while_running_apply_to_the_next_request(rig, aiohttp_client):
    client = await client_for(aiohttp_client, rig)  # the app was built with the old Route object
    rig.settings.routes = [Route(11434, "llm", "plain", max_tokens=77, allow={"vision": False, "reasoning": True, "tools": True})]  # as a Settings save does
    assert (await client.post("/v1/chat/completions", json=picture_request(1))).status == 400
    await (await client.post("/v1/completions", json={"model": "alpha", "prompt": "hi", "max_tokens": 5000})).read()
    assert sent_to_engine(rig, "/v1/completions")[-1]["max_tokens"] == 77
