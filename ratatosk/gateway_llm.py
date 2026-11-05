import asyncio
import json
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
            content = message.get("content") if isinstance(message, dict) else None  # null for tool-call turns
            if isinstance(content, list):
                content = " ".join("[image]" if p.get("type") in ("image_url", "input_image") else str(p.get("text", ""))
                                   for p in content if isinstance(p, dict))
            parts.append(content if isinstance(content, str) else "")
        return "\n".join(parts)
    prompt = body.get("prompt", "")
    return prompt if isinstance(prompt, str) else "\n".join(map(str, prompt))


def _count_pictures(body: dict) -> int:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return 0
    return sum(1 for m in messages if isinstance(m, dict) and isinstance(m.get("content"), list)
               for part in m["content"] if isinstance(part, dict) and part.get("type") in ("image_url", "input_image"))


def _asked_tokens(body: dict) -> int | None:
    """The reply length the client asked for, if it is a sensible number."""
    def positive(value):
        return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None

    return positive(body.get("max_tokens")) or positive(body.get("n_predict")) or positive(body.get("max_completion_tokens"))


def _capped_tokens(body: dict, cap: int) -> int:
    """The reply length to allow: what the client asked for, never more than the cap."""
    return min(_asked_tokens(body) or cap, cap)  # -1 would mean "until the context is full" to llama-server


def _clip(text: str, keep: int = 120_000) -> str:
    """A very long prompt is stored as its beginning and its end, which is where the instructions usually are."""
    return text if len(text) <= keep else text[:20_000] + f"\n[... {len(text) - keep} characters left out ...]\n" + text[-100_000:]


def parse_reply(raw: bytes, content_type: str) -> tuple[str, int | None, float | None]:
    """The reply text, its length in tokens and its speed, from a streamed or a plain completion response."""
    if "event-stream" in content_type:
        docs = []
        for line in raw.decode("utf-8", "replace").splitlines():
            if line.startswith("data:") and "[DONE]" not in line:
                try:
                    docs.append(json.loads(line[5:]))
                except ValueError:
                    pass
    else:
        try:
            docs = [json.loads(raw)]
        except ValueError:
            docs = []
    pieces, usage, timings = [], None, None
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        for choice in doc.get("choices") or []:
            piece = choice.get("text") or (choice.get("delta") or {}).get("content") or (choice.get("message") or {}).get("content")
            if isinstance(piece, str):
                pieces.append(piece)
        usage, timings = doc.get("usage") or usage, doc.get("timings") or timings
    tokens = (usage or {}).get("completion_tokens") or (timings or {}).get("predicted_n")
    return "".join(pieces), tokens, (timings or {}).get("predicted_per_second")


def _error(status: int, message: str) -> web.Response:
    return web.json_response({"error": message}, status=status)


def make_llm_app(route: Route, ctx: AppContext) -> web.Application:
    priority = P_RP if route.profile == "rp" else P_LLM

    async def start_session(app):
        # No total timeout: a slow model needs minutes for one reply, and the default of 300 s would cut it off.
        app[SESSION] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=10))

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
        if not isinstance(body, dict):
            return _error(400, "the request body must be a JSON object")
        name = body.get("model", "")
        model = resolve(name, ctx.models_provider(), ctx.settings.aliases, ctx.settings.default_model)
        if model is None:
            return _error(404, f"unknown model {name!r}, and no default model is set")
        if not model.path.exists():
            return _error(503, f"the file for model {model.id!r} is missing; its drive is probably not mounted")

        defaults = {**ctx.settings.request_defaults.get(model.id, {}), **ctx.settings.request_defaults.get(name, {})}
        for key, value in defaults.items():  # what the client set wins; settings inside a dict (chat_template_kwargs) are merged
            if isinstance(value, dict) and isinstance(body.get(key), dict):
                body[key] = {**value, **body[key]}
            elif key not in body:
                body[key] = value
        pictures = _count_pictures(body)
        if pictures and model.mmproj is None:
            return _error(400, f"model {model.id!r} cannot read pictures: it has no vision file "
                               "(put one beside it, or set it under mmproj_files in Settings)")
        asked = _asked_tokens(body)
        capped = _capped_tokens(body, ctx.settings.max_tokens_cap if route.profile == "rp" else ctx.settings.max_tokens_cap_plain)
        body["max_tokens"] = capped
        body.pop("n_predict", None)
        if "max_completion_tokens" in body:
            body["max_completion_tokens"] = capped
        text = _prompt_text(body)
        prompt_tokens = await ctx.engine.count_tokens(text)
        if prompt_tokens is None:
            prompt_tokens = len(text) // 3
        prompt_tokens += pictures * ctx.settings.image_max_tokens  # the engine cannot count these ahead; assume the most
        needed = needed_ctx(prompt_tokens, min(capped, ctx.settings.reply_reserve))  # a longer reply simply ends at the context limit
        if needed > ctx.scheduler.limit_bucket():
            return _error(413, f"the request needs about {needed} tokens of context; the limit is {ctx.scheduler.limit_bucket()}")

        started, status, reply_started = time.monotonic(), "ok", False
        waited, error, content_type, raw = None, None, "", bytearray()  # kept for the request log

        async def forward(engine_url: str):
            nonlocal reply_started, waited, content_type
            waited = time.monotonic() - started  # includes freeing ComfyUI and starting the engine, if that was needed
            async with request.app[SESSION].post(engine_url + request.path, json=body) as upstream:
                content_type = upstream.headers.get("Content-Type", "application/json")
                resp = web.StreamResponse(status=upstream.status, headers={"Content-Type": content_type})
                await resp.prepare(request)
                reply_started = True
                async for chunk in upstream.content.iter_any():
                    if len(raw) < 400_000:
                        raw.extend(chunk)
                    await resp.write(chunk)
                await resp.write_eof()
                return resp

        try:
            return await ctx.scheduler.run_llm(priority, model, needed, forward)
        except EngineError as exc:
            status, error = "error", str(exc)
            return _error(502, str(exc))
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            status, error = "error", f"{type(exc).__name__}: {exc}"
            if not reply_started:
                return _error(502, f"the engine did not answer: {exc}")
            raise  # the reply was already under way; dropping the connection is all that is left
        except asyncio.CancelledError:
            status, error = "cancelled", "the client disconnected"
            raise
        finally:
            reply, out_tokens, speed = parse_reply(bytes(raw), content_type)
            ctx.db.log_request(str(route.port), route.profile, model.id,
                               ctx.engine.loaded[1] if ctx.engine.loaded else 0,
                               prompt_tokens, out_tokens or 0, time.monotonic() - started, status,
                               client=request.remote, user_agent=request.headers.get("User-Agent", "")[:200],
                               path=request.path, stream=int(bool(body.get("stream"))), max_tokens_asked=asked,
                               max_tokens=capped, queue_seconds=waited, tokens_per_second=speed, error=error,
                               prompt_text=_clip(text), reply_text=reply or None)

    app = web.Application(client_max_size=16 * 2**20)  # a long chat history is far bigger than aiohttp's 1 MiB default
    app.on_startup.append(start_session)
    app.on_cleanup.append(close_session)
    app.router.add_get("/v1/models", list_models)
    app.router.add_post("/v1/completions", complete)
    app.router.add_post("/v1/chat/completions", complete)
    return app
