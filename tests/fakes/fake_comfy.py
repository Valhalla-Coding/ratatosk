"""A fake ComfyUI for tests: /queue, /free, /prompt, /system_stats and a /ws echo."""
import asyncio
from dataclasses import dataclass, field

from aiohttp import WSMsgType, web


@dataclass
class ComfyState:
    running: int = 0
    pending: int = 0
    job_seconds: float = 0.05
    freed: list = field(default_factory=list)
    prompts: list = field(default_factory=list)


def make_app(state: ComfyState) -> web.Application:
    async def queue(_):
        return web.json_response({"queue_running": [1] * state.running, "queue_pending": [1] * state.pending})

    async def free(request):
        state.freed.append(await request.json())
        return web.json_response({})

    async def prompt(request):
        state.prompts.append(await request.json())
        state.running += 1

        async def finish():
            await asyncio.sleep(state.job_seconds)
            state.running -= 1

        asyncio.get_running_loop().create_task(finish())
        return web.json_response({"prompt_id": "fake"})

    async def stats(_):
        return web.json_response({"system": {"fake": True}})

    async def ws(request):
        sock = web.WebSocketResponse()
        await sock.prepare(request)
        async for msg in sock:
            if msg.type == WSMsgType.TEXT:
                await sock.send_str("echo:" + msg.data)
        return sock

    app = web.Application()
    app.router.add_get("/queue", queue)
    app.router.add_post("/free", free)
    app.router.add_post("/prompt", prompt)
    app.router.add_get("/system_stats", stats)
    app.router.add_get("/ws", ws)
    return app
