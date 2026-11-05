"""A fake ComfyUI for tests: /queue, /free, /prompt, /system_stats, /upload/image and a /ws echo.

It copies two behaviours of the real one that a naive proxy trips over:
- the origin-only middleware: a request whose Host is loopback and whose Origin names another host is refused (403);
- /free only sets a flag, the memory is released a moment later (state.torch_held drops after release_delay)."""
import asyncio
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

from aiohttp import WSMsgType, web

LOOPBACK = {"127.0.0.1", "localhost", "::1"}


@dataclass
class ComfyState:
    running: int = 0
    pending: int = 0
    job_seconds: float = 0.05
    freed: list = field(default_factory=list)
    interrupts: int = 0
    cleared: int = 0
    prompts: list = field(default_factory=list)
    prompt_times: list = field(default_factory=list)  # time.monotonic() when each /prompt arrived
    vram_total: int = 8_000_000_000
    torch_held: int = 5_000_000_000  # memory ComfyUI's torch holds; /free drops it to 32 MB after release_delay
    release_delay: float = 0.1  # the real worker thread acts on /free a moment later, sometimes more than a second


@web.middleware
async def origin_only(request, handler):
    host = request.headers.get("Host", "")
    origin = request.headers.get("Origin")
    if host.rsplit(":", 1)[0] in LOOPBACK and origin is not None and urlparse(origin).netloc != host:
        return web.Response(status=403, text="origin does not match host")
    return await handler(request)


def make_app(state: ComfyState) -> web.Application:
    async def queue(_):
        return web.json_response({"queue_running": [1] * state.running, "queue_pending": [1] * state.pending})

    async def free(request):
        state.freed.append(await request.json())

        async def release():
            await asyncio.sleep(state.release_delay)
            state.torch_held = 32_000_000

        asyncio.get_running_loop().create_task(release())
        return web.json_response({})

    async def interrupt(_):
        state.interrupts += 1
        state.running = 0
        return web.json_response({})

    async def clear_queue(request):
        if (await request.json()).get("clear"):
            state.cleared += 1
            state.pending = 0
        return web.json_response({})

    async def prompt(request):
        state.prompts.append(await request.json())
        state.prompt_times.append(time.monotonic())
        state.running += 1

        async def finish():
            await asyncio.sleep(state.job_seconds)
            state.running -= 1

        asyncio.get_running_loop().create_task(finish())
        return web.json_response({"prompt_id": "fake"})

    async def stats(_):
        return web.json_response({"system": {"fake": True}, "devices": [{
            "name": "fake", "vram_total": state.vram_total, "vram_free": state.vram_total - state.torch_held - 500_000_000,
            "torch_vram_total": state.torch_held, "torch_vram_free": 0}]})

    async def upload(request):
        return web.json_response({"size": len(await request.read())})

    async def ws(request):
        sock = web.WebSocketResponse()
        await sock.prepare(request)
        async for msg in sock:
            if msg.type == WSMsgType.TEXT:
                await sock.send_str("echo:" + msg.data)
        return sock

    app = web.Application(middlewares=[origin_only], client_max_size=100 * 2**20)
    app.router.add_get("/queue", queue)
    app.router.add_post("/free", free)
    app.router.add_post("/interrupt", interrupt)
    app.router.add_post("/queue", clear_queue)
    app.router.add_post("/prompt", prompt)
    app.router.add_get("/system_stats", stats)
    app.router.add_post("/upload/image", upload)
    app.router.add_get("/ws", ws)
    return app
