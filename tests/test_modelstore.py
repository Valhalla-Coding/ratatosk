import asyncio
from pathlib import Path

import pytest

from ratatosk import modelstore
from ratatosk.db import Db
from ratatosk.gateway_llm import AppContext
from ratatosk.gpuqueue import GpuQueue
from ratatosk.models import scan
from ratatosk.scheduler import Scheduler
from ratatosk.settings import Settings, load_settings
from ratatosk.web import make_ui_app
from tests.fakes.fake_hf import HfState, make_app


class FakeEngine:
    loaded = None
    pid = None


class FakeComfy:
    async def pending(self):
        return 0


async def fake_gpu():
    return None


@pytest.fixture
async def rig(tmp_path, aiohttp_client, aiohttp_server):
    models = tmp_path / "models"
    (models / "pub" / "repo").mkdir(parents=True)
    for name in ("Alpha.gguf", "Beta.gguf", "mmproj-shared.gguf"):
        (models / "pub" / "repo" / name).write_bytes(b"1234")
    hf_state = HfState(repos={"owner/repo": {"M-Q4.gguf": b"x" * 3000, "M-Q8.gguf": b"y" * 5000, "mmproj-M.gguf": b"v" * 100}})
    hf = await aiohttp_server(make_app(hf_state))
    settings = Settings(data_dir=str(tmp_path), model_folders=[{"path": str(models), "label": "t"}],
                        hf_base_url=str(hf.make_url("")).rstrip("/"),
                        aliases={"cindy": "alpha", "other": "beta"}, default_model="alpha")
    queue = GpuQueue()
    queue.start()
    engine = FakeEngine()
    ctx = AppContext(settings, Scheduler(settings, queue, engine, FakeComfy()), engine, Db(tmp_path / "r.db"),
                     lambda: scan([models], settings.mmproj_files))

    class Rig:
        pass

    r = Rig()
    r.settings, r.engine, r.models, r.hf, r.path = settings, engine, models, hf_state, tmp_path / "settings.json"
    r.client = await aiohttp_client(make_ui_app(ctx, FakeComfy(), r.path, gpu_reader=fake_gpu))
    yield r
    await queue.stop()


async def test_list_shape(rig):
    data = await (await rig.client.get("/api/models")).json()
    by_id = {m["id"]: m for m in data}
    assert set(by_id) == {"alpha", "beta"}
    alpha = by_id["alpha"]
    assert alpha["names"] == ["cindy"] and alpha["default"] is True and by_id["beta"]["default"] is False
    assert alpha["size"] == 4 and set(alpha["capabilities"]) == {"vision", "tools", "reasoning"}
    assert alpha["capabilities"]["vision"] is True  # the shared vision file sits beside it


async def test_delete_removes_file_names_and_keeps_a_vision_file_another_model_uses(rig):
    resp = await rig.client.delete("/api/models/alpha")
    assert resp.status == 200
    folder = rig.models / "pub" / "repo"
    assert not (folder / "Alpha.gguf").exists() and (folder / "mmproj-shared.gguf").exists()  # beta still uses it
    assert "cindy" not in rig.settings.aliases and rig.settings.default_model == ""
    assert load_settings(rig.path).aliases == {"other": "beta"}
    assert (await rig.client.delete("/api/models/beta")).status == 200
    assert not (folder / "mmproj-shared.gguf").exists()  # nobody uses it any more


async def test_delete_is_refused_while_the_engine_serves_the_model_and_for_unknown_ids(rig):
    rig.engine.loaded = (Path(rig.models / "pub" / "repo" / "Alpha.gguf"), 8192)
    resp = await rig.client.delete("/api/models/alpha")
    assert resp.status == 409 and "in use" in (await resp.json())["error"]
    assert (rig.models / "pub" / "repo" / "Alpha.gguf").exists()
    assert (await rig.client.delete("/api/models/nope")).status == 404


@pytest.mark.parametrize("url", ["https://huggingface.co/owner/repo", "https://huggingface.co/owner/repo/blob/main/M-Q4.gguf",
                                 "huggingface.co/owner/repo/tree/main?x=1", "owner/repo"])
async def test_find_understands_link_shapes_and_lists_gguf_files(rig, url):
    data = await (await rig.client.post("/api/models/find", json={"url": url})).json()
    assert data["repo"] == "owner/repo"
    assert [(f["file"], f["size"]) for f in data["files"]] == [("M-Q4.gguf", 3000), ("M-Q8.gguf", 5000)]
    assert [(f["file"], f["size"]) for f in data["vision"]] == [("mmproj-M.gguf", 100)]


async def test_find_reports_a_missing_repo_and_a_bad_link(rig):
    assert (await rig.client.post("/api/models/find", json={"url": "owner/missing"})).status == 404
    assert (await rig.client.post("/api/models/find", json={"url": "not a link at all"})).status == 400


async def wait_done(rig):
    for _ in range(100):
        downloads = await (await rig.client.get("/api/models/downloads")).json()
        if downloads and downloads[-1]["status"] != "running":
            return downloads[-1]
        await asyncio.sleep(0.05)
    raise AssertionError("the download never finished")


async def test_download_saves_next_to_its_vision_file_and_shows_up_as_a_model(rig):
    resp = await rig.client.post("/api/models/download", json={"repo": "owner/repo", "file": "M-Q4.gguf", "vision": "mmproj-M.gguf"})
    assert resp.status == 202
    done = await wait_done(rig)
    assert done["status"] == "done" and done["done"] == done["total"] == 3100
    folder = rig.models / "owner" / "repo"
    assert (folder / "M-Q4.gguf").read_bytes() == b"x" * 3000 and (folder / "mmproj-M.gguf").exists()
    ids = [m["id"] for m in await (await rig.client.get("/api/models")).json()]
    assert "m-q4" in ids and not list(folder.glob("*.part"))


async def test_a_cut_download_leaves_no_model_and_resumes_where_it_stopped(rig):
    rig.hf.cut_once_at = 1000
    await rig.client.post("/api/models/download", json={"repo": "owner/repo", "file": "M-Q8.gguf"})
    failed = await wait_done(rig)
    assert failed["status"] == "error" and failed["error"]
    folder = rig.models / "owner" / "repo"
    assert (folder / "M-Q8.gguf.part").stat().st_size == 1000 and not (folder / "M-Q8.gguf").exists()
    assert "m-q8" not in [m["id"] for m in await (await rig.client.get("/api/models")).json()]
    await rig.client.post("/api/models/download", json={"repo": "owner/repo", "file": "M-Q8.gguf"})
    for _ in range(100):
        if (folder / "M-Q8.gguf").exists():
            break
        await asyncio.sleep(0.05)
    assert (folder / "M-Q8.gguf").read_bytes() == b"y" * 5000
    assert rig.hf.ranges[-1] == "bytes=1000-"


async def test_download_is_refused_when_the_disk_has_no_room_and_for_unsafe_names(rig, monkeypatch):
    monkeypatch.setattr(modelstore.shutil, "disk_usage", lambda p: (10**12, 10**12 - 1000, 1000))
    resp = await rig.client.post("/api/models/download", json={"repo": "owner/repo", "file": "M-Q4.gguf"})
    assert resp.status == 507 and "room" in (await resp.json())["error"]
    assert not (rig.models / "owner").exists()
    monkeypatch.undo()
    bad = await rig.client.post("/api/models/download", json={"repo": "owner/repo", "file": "../evil.gguf"})
    assert bad.status == 400
    bad = await rig.client.post("/api/models/download", json={"repo": "owner/repo", "file": "notes.txt"})
    assert bad.status == 400
