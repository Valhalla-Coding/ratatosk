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
    assert set(data) == {"queue", "engine", "comfy_pending", "gpu", "recent"}
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


async def test_index_has_both_tabs(rig):
    text = await (await rig.client.get("/")).text()
    assert "Now" in text and "Settings" in text


def test_parse_gpu():
    assert parse_gpu("7753, 8192\n") == {"used_mb": 7753, "total_mb": 8192}
    assert parse_gpu("garbage") is None
