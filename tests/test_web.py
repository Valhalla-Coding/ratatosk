from pathlib import Path

import pytest

from ratatosk.db import Db
from ratatosk.gateway_llm import AppContext
from ratatosk.gpuqueue import GpuQueue
from ratatosk.scheduler import Scheduler
from ratatosk.settings import Settings, load_settings
from ratatosk.web import make_ui_app, parse_gpu


class FakeEngine:
    loaded = None
    pid = None


class FakeComfy:
    async def pending(self):
        return 0


async def fake_gpu():
    return {"used_mb": 100, "total_mb": 8192}


@pytest.fixture
async def rig(tmp_path, aiohttp_client):
    settings = Settings(data_dir=str(tmp_path))
    queue = GpuQueue()
    queue.start()
    engine, comfy = FakeEngine(), FakeComfy()
    ctx = AppContext(settings, Scheduler(settings, queue, engine, comfy), engine, Db(tmp_path / "r.db"), lambda: [])
    path = tmp_path / "settings.json"

    class Rig:
        pass

    r = Rig()
    r.settings, r.engine, r.path, r.db = settings, engine, path, ctx.db
    r.client = await aiohttp_client(make_ui_app(ctx, comfy, path, gpu_reader=fake_gpu))
    yield r
    await queue.stop()


async def test_status_shape(rig):
    data = await (await rig.client.get("/api/status")).json()
    assert set(data) == {"queue", "engine", "comfy_pending", "gpu", "recent", "doors"}
    assert data["doors"]["11434"] == {"today": 0, "last_seen": None, "busy": False}
    assert data["queue"] == {"running": None, "pending": []}
    assert data["engine"] is None and data["comfy_pending"] == 0
    assert data["gpu"] == {"used_mb": 100, "total_mb": 8192}
    rig.engine.loaded = (Path("/m/a.gguf"), 16384)
    assert (await (await rig.client.get("/api/status")).json())["engine"] == {"model": "a.gguf", "ctx": 16384}


async def test_settings_put_persists_and_rejects_bad_input(rig):
    resp = await rig.client.put("/api/settings", json={"max_tokens_cap": 512})
    assert resp.status == 200
    assert rig.settings.max_tokens_cap == 512  # applied to the running service
    assert load_settings(rig.path).max_tokens_cap == 512  # and saved
    bad_kind = await rig.client.put("/api/settings", json={"routes": [{"port": 1234, "kind": "bogus", "profile": "plain"}]})
    assert bad_kind.status == 400 and "kind" in (await bad_kind.json())["error"]
    bad_type = await rig.client.put("/api/settings", json={"max_tokens_cap": "abc"})
    assert bad_type.status == 400
    assert rig.settings.max_tokens_cap == 512  # a rejected change leaves everything as it was


@pytest.mark.parametrize("change", [
    {"routes": [{"port": 11434, "kind": "llm"}, {"port": 11434, "kind": "comfy"}]},  # the same port twice
    {"routes": [{"port": 5960, "kind": "llm"}]},  # a route on the UI's own port
    {"routes": [{"port": 8081, "kind": "llm"}]},  # a route on the engine's port
    {"routes": [{"port": 80, "kind": "llm"}]},  # a privileged port the service cannot bind
    {"routes": []},  # nothing left to serve
    {"ui_port": 11434},  # the UI moved onto a route's port
])
async def test_settings_put_rejects_port_clashes_that_would_crash_the_service(rig, change):
    before = rig.settings.ui_port, [r.port for r in rig.settings.routes]
    resp = await rig.client.put("/api/settings", json=change)
    assert resp.status == 400 and "error" in await resp.json()
    assert (rig.settings.ui_port, [r.port for r in rig.settings.routes]) == before  # nothing was applied


async def test_request_detail_endpoint(rig):
    rid = rig.db.log_request("11434", "plain", "m", 8192, 3, 4, 1.0, "ok", client="192.168.1.9", prompt_text="P", reply_text="R")
    recent = (await (await rig.client.get("/api/status")).json())["recent"]
    assert recent[0]["has_content"] == 1 and "prompt_text" not in recent[0]  # the list stays light
    detail = await (await rig.client.get(f"/api/requests/{rid}")).json()
    assert (detail["client"], detail["prompt_text"], detail["reply_text"]) == ("192.168.1.9", "P", "R")
    assert (await rig.client.get("/api/requests/9999")).status == 404
    assert (await rig.client.get("/api/requests/abc")).status == 404


async def test_index_has_both_tabs(rig):
    text = await (await rig.client.get("/")).text()
    assert "Now" in text and "Settings" in text


def test_parse_gpu():
    assert parse_gpu("7753, 8192\n") == {"used_mb": 7753, "total_mb": 8192}
    assert parse_gpu("garbage") is None


def test_gpu_processes_are_named_and_sorted_by_memory():
    from ratatosk.web import parse_gpu_processes
    text = ("2698210, /home/master/Applications/Data/Packages/ComfyUI/venv/bin/python3, 158\n"
            "306211, /home/master/.lmstudio/extensions/backends/llama.cpp-x/llama-server, 7152\n"
            "55, /home/master/ratatosk/engine/backend/llama-server, 6060\n"
            "9692, /usr/bin/ptyxis, 84\n"
            "not a row\n")
    assert parse_gpu_processes(text) == [
        {"pid": 306211, "name": "LM Studio engine", "mb": 7152},
        {"pid": 55, "name": "Ratatosk engine", "mb": 6060},
        {"pid": 2698210, "name": "ComfyUI", "mb": 158},
        {"pid": 9692, "name": "ptyxis", "mb": 84},
    ]


@pytest.mark.parametrize("priority", [
    [{"door": 9999}],  # no such door
    [{"client": "not an address"}],
    [{"client": "192.168.1.5", "drop": "yes"}],
    [{"door": 11434, "client": "192.168.1.5"}],  # a line is one or the other
    [{}],
])
async def test_priority_validation_rejects_bad_lines(rig, priority):
    resp = await rig.client.put("/api/settings", json={"priority": priority})
    assert resp.status == 400


async def test_priority_and_door_settings_roundtrip(rig):
    lines = [{"client": "192.168.1.0/24", "agent": "node", "drop": True, "label": "ST"}, {"door": 11434}]
    routes = [{"port": 11434, "kind": "llm", "profile": "plain", "max_tokens": 777,
               "allow": {"vision": False, "reasoning": True, "tools": True}}]
    resp = await rig.client.put("/api/settings", json={"priority": lines, "routes": routes})
    assert resp.status == 200, await resp.text()
    assert rig.settings.rules() == lines and rig.settings.routes[0].max_tokens == 777
    assert load_settings(rig.path).routes[0].allow["vision"] is False
    bad = await rig.client.put("/api/settings", json={"routes": [{"port": 11434, "kind": "llm", "max_tokens": 0}]})
    assert bad.status == 400


async def test_clients_endpoint_lists_who_has_been_calling(rig):
    rig.db.log_request("11434", "plain", "m", 1, 1, 1, 1.0, "ok", client="192.168.1.12", user_agent="node-fetch")
    clients = await (await rig.client.get("/api/clients")).json()
    assert [(c["client"], c["agent"], c["count"]) for c in clients] == [("192.168.1.12", "node-fetch", 1)]
    status = await (await rig.client.get("/api/status")).json()
    assert status["doors"]["11434"]["today"] == 1 and status["doors"]["11435"]["today"] == 0


async def test_the_settings_script_is_served_and_stored_text_can_be_cleared(rig):
    resp = await rig.client.get("/static/settings.js")
    assert resp.status == 200 and "initSettings" in await resp.text()
    rig.db.log_request("11434", "plain", "m", 1, 1, 1, 1.0, "ok", prompt_text="P", reply_text="R")
    assert (await rig.client.post("/api/history/clear-text")).status == 200
    detail = await (await rig.client.get("/api/requests/1")).json()
    assert detail["prompt_text"] is None and detail["reply_text"] is None and detail["gen_tokens"] == 1  # the numbers stay
