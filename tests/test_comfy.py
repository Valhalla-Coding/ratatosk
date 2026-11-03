import aiohttp

from ratatosk.comfy import Comfy
from tests.fakes.fake_comfy import ComfyState, make_app


async def test_pending_sums_running_and_pending(aiohttp_server):
    state = ComfyState(running=2, pending=3)
    server = await aiohttp_server(make_app(state))
    async with aiohttp.ClientSession() as session:
        assert await Comfy(str(server.make_url("")).rstrip("/"), session).pending() == 5


async def test_free_body(aiohttp_server):
    state = ComfyState()
    server = await aiohttp_server(make_app(state))
    async with aiohttp.ClientSession() as session:
        await Comfy(str(server.make_url("")).rstrip("/"), session).free()
    assert state.freed == [{"unload_models": True, "free_memory": True}]
