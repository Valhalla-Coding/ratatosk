import asyncio

import aiohttp

from ratatosk.comfy import Comfy
from tests.fakes.fake_comfy import ComfyState, make_app


async def test_pending_sums_running_and_pending(aiohttp_server):
    state = ComfyState(running=2, pending=3)
    server = await aiohttp_server(make_app(state))
    async with aiohttp.ClientSession() as session:
        assert await Comfy(str(server.make_url("")).rstrip("/"), session).pending() == 5


async def test_free_waits_until_the_memory_is_actually_released(aiohttp_server):
    state = ComfyState()  # the fake releases its memory in steps after /free, like ComfyUI's worker thread
    server = await aiohttp_server(make_app(state))
    async with aiohttp.ClientSession() as session:
        await Comfy(str(server.make_url("")).rstrip("/"), session, settle_poll=0.05, settle_quiet=0.4).free()
    assert state.vram_free == state.vram_total


async def test_free_gives_up_waiting_after_the_cap(aiohttp_server):
    state = ComfyState()  # a ComfyUI whose reading never settles must not block text forever

    async def drift():
        while True:
            state.vram_free += 1
            await asyncio.sleep(0.02)

    task = asyncio.create_task(drift())
    server = await aiohttp_server(make_app(state))
    try:
        async with aiohttp.ClientSession() as session:
            comfy = Comfy(str(server.make_url("")).rstrip("/"), session, settle_poll=0.05, settle_quiet=0.4, settle_cap=0.6)
            await asyncio.wait_for(comfy.free(), 5)
    finally:
        task.cancel()


async def test_free_body(aiohttp_server):
    state = ComfyState()
    server = await aiohttp_server(make_app(state))
    async with aiohttp.ClientSession() as session:
        await Comfy(str(server.make_url("")).rstrip("/"), session).free()
    assert state.freed == [{"unload_models": True, "free_memory": True}]
