import asyncio
import json
import socket
import sys
from pathlib import Path

import aiohttp
import pytest

from ratatosk.db import Db
from ratatosk.engine import Engine
from ratatosk.gateway_llm import SESSION, AppContext, make_llm_app
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
    settings = Settings(engine_port=port, data_dir=str(tmp_path), default_model="alpha",
                        aliases={"qwen3.5-9b-obliterated": "alpha"})
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
    assert rig.db.recent(1)[0]["status"] == "error"


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
