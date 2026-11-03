import asyncio
import time
from dataclasses import dataclass
from typing import Callable

import aiohttp
from aiohttp import web

from ratatosk.buckets import needed_ctx
from ratatosk.db import Db
from ratatosk.engine import Engine, EngineError
from ratatosk.gpuqueue import P_LLM, P_RP
from ratatosk.models import Model, resolve
from ratatosk.scheduler import Scheduler
from ratatosk.settings import Route, Settings


SESSION = web.AppKey("session", aiohttp.ClientSession)


@dataclass
class AppContext:
    settings: Settings
    scheduler: Scheduler
    engine: Engine
    db: Db
    models_provider: Callable[[], list[Model]]


def _prompt_text(body: dict) -> str:
    if "messages" in body:
        parts = []
        for message in body["messages"]:
            content = message.get("content", "")
            parts.append(content if isinstance(content, str)
                         else " ".join(p.get("text", "") for p in content if isinstance(p, dict)))
        return "\n".join(parts)
    prompt = body.get("prompt", "")
    return prompt if isinstance(prompt, str) else "\n".join(map(str, prompt))


def _error(status: int, message: str) -> web.Response:
    return web.json_response({"error": message}, status=status)


def make_llm_app(route: Route, ctx: AppContext) -> web.Application:
    priority = P_RP if route.profile == "rp" else P_LLM

    async def start_session(app):
        app[SESSION] = aiohttp.ClientSession()

    async def close_session(app):
        await app[SESSION].close()

    async def list_models(_):
        ids = list(ctx.settings.aliases) + [m.id for m in ctx.models_provider()]
        return web.json_response({"object": "list", "data": [{"id": i, "object": "model"} for i in dict.fromkeys(ids)]})

    async def complete(request: web.Request):
        try:
            body = await request.json()
        except ValueError:
            return _error(400, "the request body is not valid JSON")
        name = body.get("model", "")
        model = resolve(name, ctx.models_provider(), ctx.settings.aliases, ctx.settings.default_model)
        if model is None:
            return _error(404, f"unknown model {name!r}, and no default model is set")
        if not model.path.exists():
            return _error(503, f"the file for model {model.id!r} is missing; its drive is probably not mounted")

        cap = ctx.settings.max_tokens_cap
        capped = min(int(body.get("max_tokens") or body.get("n_predict") or cap), cap)
        body["max_tokens"] = capped
        body.pop("n_predict", None)
        text = _prompt_text(body)
        prompt_tokens = await ctx.engine.count_tokens(text)
        if prompt_tokens is None:
            prompt_tokens = len(text) // 3
        needed = needed_ctx(prompt_tokens, capped)
        if needed > ctx.scheduler.limit_bucket():
            return _error(413, f"the request needs about {needed} tokens of context; the limit is {ctx.scheduler.limit_bucket()}")

        started, status = time.monotonic(), "ok"

        async def forward(engine_url: str):
            async with request.app[SESSION].post(engine_url + request.path, json=body) as upstream:
                resp = web.StreamResponse(status=upstream.status, headers={
                    "Content-Type": upstream.headers.get("Content-Type", "application/json")})
                await resp.prepare(request)
                async for chunk in upstream.content.iter_any():
                    await resp.write(chunk)
                await resp.write_eof()
                return resp

        try:
            return await ctx.scheduler.run_llm(priority, model, needed, forward)
        except EngineError as exc:
            status = "error"
            return _error(502, str(exc))
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        finally:
            ctx.db.log_request(str(route.port), route.profile, model.id,
                               ctx.engine.loaded[1] if ctx.engine.loaded else 0,
                               prompt_tokens, 0, time.monotonic() - started, status)

    app = web.Application()
    app.on_startup.append(start_session)
    app.on_cleanup.append(close_session)
    app.router.add_get("/v1/models", list_models)
    app.router.add_post("/v1/completions", complete)
    app.router.add_post("/v1/chat/completions", complete)
    return app
