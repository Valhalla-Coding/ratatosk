import asyncio

import aiohttp
from aiohttp import WSMsgType, web

from ratatosk.gateway_llm import AppContext
from ratatosk.scheduler import GpuBusy
from ratatosk.settings import Route

SESSION = web.AppKey("session", aiohttp.ClientSession)
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
              "transfer-encoding", "upgrade", "host", "content-length"}


def _clean(headers) -> dict:
    return {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP}


def make_comfy_app(route: Route, ctx: AppContext) -> web.Application:
    def base() -> str:
        return ctx.settings.comfy_url.rstrip("/")

    async def start_session(app):
        app[SESSION] = aiohttp.ClientSession(auto_decompress=False)  # pass compressed bodies through untouched

    async def close_session(app):
        await app[SESSION].close()

    async def prompt(request: web.Request):
        """Queued image jobs: wait for the GPU, free the model, then hand the job to ComfyUI."""
        body = await request.read()
        headers = _clean(request.headers)

        async def forward():
            async with request.app[SESSION].post(base() + request.path_qs, data=body, headers=headers) as up:
                return web.Response(body=await up.read(), status=up.status, headers=_clean(up.headers))

        try:
            return await ctx.scheduler.run_image(forward)
        except GpuBusy as exc:
            return web.json_response({"error": f"GPU busy: {exc}"}, status=503)

    async def proxy_ws(request: web.Request):
        server = web.WebSocketResponse(max_msg_size=0)  # ComfyUI sends large preview frames
        await server.prepare(request)
        url = base().replace("http", "ws", 1) + request.path_qs
        async with request.app[SESSION].ws_connect(url, max_msg_size=0) as client:
            async def pump(src, dst):
                async for msg in src:
                    if msg.type == WSMsgType.TEXT:
                        await dst.send_str(msg.data)
                    elif msg.type == WSMsgType.BINARY:
                        await dst.send_bytes(msg.data)
                    else:
                        break

            tasks = [asyncio.ensure_future(pump(server, client)), asyncio.ensure_future(pump(client, server))]
            _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
        await server.close()
        return server

    async def passthrough(request: web.Request):
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return await proxy_ws(request)
        body = await request.read()
        async with request.app[SESSION].request(request.method, base() + request.path_qs, data=body or None,
                                                headers=_clean(request.headers), allow_redirects=False) as up:
            resp = web.StreamResponse(status=up.status, headers=_clean(up.headers))
            await resp.prepare(request)
            async for chunk in up.content.iter_any():
                await resp.write(chunk)
            await resp.write_eof()
            return resp

    app = web.Application()
    app.on_startup.append(start_session)
    app.on_cleanup.append(close_session)
    app.router.add_post("/prompt", prompt)
    app.router.add_post("/api/prompt", prompt)
    app.router.add_route("*", "/{tail:.*}", passthrough)
    return app
