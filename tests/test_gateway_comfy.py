import asyncio
from pathlib import Path

import aiohttp
import pytest

from ratatosk.comfy import Comfy
from ratatosk.gateway_comfy import make_comfy_app
from ratatosk.gateway_llm import AppContext
from ratatosk.gpuqueue import P_LLM, GpuQueue
from ratatosk.models import Model
from ratatosk.scheduler import Scheduler
from ratatosk.settings import Route, Settings
from tests.fakes.fake_comfy import ComfyState, make_app

MODEL = Model("m", Path("/m/a.gguf"), 1, Path("/m"))


class FakeEngine:
    url = "http://engine"

    def __init__(self):
        self.loaded, self.running = None, False

    @property
    def pid(self):
        return 1 if self.running else None

    async def ensure(self, model, ctx):
        self.loaded, self.running = (model, ctx), True

    async def stop(self):
        self.loaded, self.running = None, False


@pytest.fixture
async def rig(aiohttp_server, aiohttp_client):
    state = ComfyState()
    fake = await aiohttp_server(make_app(state))
    settings = Settings(comfy_url=str(fake.make_url("")).rstrip("/"), image_wait_seconds=5)
    session = aiohttp.ClientSession()
    queue = GpuQueue()
    queue.start()
    engine = FakeEngine()
    comfy = Comfy(settings.comfy_url, session, settle_poll=0.05, settle_quiet=0.2)
    scheduler = Scheduler(settings, queue, engine, comfy, poll_seconds=0.05)

    class Rig:
        pass

    r = Rig()
    r.state, r.settings, r.scheduler, r.engine = state, settings, scheduler, engine
    ctx = AppContext(settings, scheduler, engine, db=None, models_provider=lambda: [])
    r.client = await aiohttp_client(make_comfy_app(Route(8188, "comfy"), ctx))

    async def hold_gpu(gate):
        async def slow(url):
            await gate.wait()

        task = asyncio.create_task(scheduler.run_llm(P_LLM, MODEL, 5000, slow))
        for _ in range(200):  # the job first frees ComfyUI's memory, then starts the engine, then holds the GPU
            if engine.running:
                break
            await asyncio.sleep(0.02)
        return task

    r.hold_gpu = hold_gpu
    yield r
    await queue.stop()
    await session.close()


async def test_passthrough_system_stats(rig):
    resp = await rig.client.get("/system_stats")
    assert resp.status == 200 and (await resp.json())["system"] == {"fake": True}


async def test_prompt_waits_behind_llm_job(rig):
    gate = asyncio.Event()
    llm = await rig.hold_gpu(gate)
    post = asyncio.create_task(rig.client.post("/prompt", json={"prompt": {}}))
    await asyncio.sleep(0.2)
    assert rig.state.prompts == [] and rig.engine.running  # still waiting its turn
    gate.set()
    resp = await post
    await llm
    assert resp.status == 200 and (await resp.json())["prompt_id"] == "fake"
    assert len(rig.state.prompts) == 1 and not rig.engine.running  # the model was unloaded for the image


async def test_prompt_503_when_wait_exceeded(rig):
    rig.settings.image_wait_seconds = 0.1
    gate = asyncio.Event()
    llm = await rig.hold_gpu(gate)
    resp = await rig.client.post("/prompt", json={"prompt": {}})
    assert resp.status == 503 and "busy" in (await resp.json())["error"].lower()
    assert rig.state.prompts == []
    gate.set()
    await llm


async def test_second_prompt_waits_until_comfy_idle(rig):
    rig.state.job_seconds = 0.5
    a = asyncio.create_task(rig.client.post("/prompt", json={"prompt": {"n": 1}}))
    b = asyncio.create_task(rig.client.post("/prompt", json={"prompt": {"n": 2}}))
    await asyncio.gather(a, b)
    first, second = rig.state.prompt_times
    assert second - first >= 0.45


async def test_prompt_keeps_the_clients_host_so_comfyuis_origin_check_passes(rig):
    # A browser on the LAN sends Host and Origin for the address it used. ComfyUI refuses the request
    # when it sees a loopback Host with a different Origin, which is what a rewritten Host would look like.
    resp = await rig.client.post("/prompt", json={"prompt": {}},
                                 headers={"Host": "192.168.1.11:8188", "Origin": "http://192.168.1.11:8188"})
    assert resp.status == 200


async def test_large_upload_passes_through(rig):
    resp = await rig.client.post("/upload/image", data=b"x" * (3 * 2**20))
    assert resp.status == 200 and (await resp.json())["size"] == 3 * 2**20


async def test_ws_echo_both_directions(rig):
    async with rig.client.ws_connect("/ws") as ws:
        await ws.send_str("hi")
        assert (await ws.receive_str()) == "echo:hi"
