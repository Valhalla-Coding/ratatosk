import asyncio

import aiohttp

from ratatosk.comfy import Comfy
from tests.fakes.fake_comfy import ComfyState, make_app


async def test_pending_sums_running_and_pending(aiohttp_server):
    state = ComfyState(running=2, pending=3)
    server = await aiohttp_server(make_app(state))
    async with aiohttp.ClientSession() as session:
        assert await Comfy(str(server.make_url("")).rstrip("/"), session).pending() == 5


async def test_free_waits_for_a_release_that_starts_late(aiohttp_server):
    state = ComfyState(release_delay=1.2)  # the real worker took longer than a second to begin releasing
    server = await aiohttp_server(make_app(state))
    async with aiohttp.ClientSession() as session:
        await Comfy(str(server.make_url("")).rstrip("/"), session, settle_poll=0.1, settle_cap=5).free()
    assert state.torch_held < 256 * 2**20


async def test_free_returns_at_once_when_comfyui_holds_nothing(aiohttp_server):
    state = ComfyState(torch_held=32_000_000)
    server = await aiohttp_server(make_app(state))
    async with aiohttp.ClientSession() as session:
        started = asyncio.get_running_loop().time()
        await Comfy(str(server.make_url("")).rstrip("/"), session, settle_poll=0.1, settle_cap=5).free()
        assert asyncio.get_running_loop().time() - started < 0.5  # nothing to wait for: no delay on every swap
    assert state.freed == [{"unload_models": True, "free_memory": True}]  # the request is still sent


async def test_free_gives_up_waiting_after_the_cap(aiohttp_server):
    state = ComfyState(release_delay=100)  # a ComfyUI that never lets go must not block text forever
    server = await aiohttp_server(make_app(state))
    async with aiohttp.ClientSession() as session:
        comfy = Comfy(str(server.make_url("")).rstrip("/"), session, settle_poll=0.05, settle_cap=0.6)
        await asyncio.wait_for(comfy.free(), 5)
    assert state.torch_held > 256 * 2**20


async def test_free_body(aiohttp_server):
    state = ComfyState()
    server = await aiohttp_server(make_app(state))
    async with aiohttp.ClientSession() as session:
        await Comfy(str(server.make_url("")).rstrip("/"), session).free()
    assert state.freed == [{"unload_models": True, "free_memory": True}]
