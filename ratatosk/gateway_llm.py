import asyncio
import json
import time
from dataclasses import dataclass
from typing import Callable

import aiohttp
from aiohttp import web

from ratatosk.buckets import BUCKETS, needed_ctx
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


PICTURE_TYPES = ("image_url", "input_image", "image")  # OpenAI chat, OpenAI Responses, Anthropic
CAP_FIELD = {  # every generation path Ratatosk queues, and how that API spells its reply length
    "/v1/chat/completions": "max_tokens", "/v1/completions": "max_tokens", "/v1/messages": "max_tokens",
    "/v1/responses": "max_output_tokens", "/completion": "n_predict", "/infill": "n_predict",
}
LENGTH_FIELDS = ("max_tokens", "n_predict", "max_completion_tokens", "max_output_tokens")


def _blocks_text(content) -> str:
    """Text of one message: a string, or a list of typed blocks (the three chat APIs spell them alike)."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return " ".join("" if not isinstance(p, dict) else "[image]" if p.get("type") in PICTURE_TYPES else str(p.get("text", ""))
                    for p in content)


def _prompt_text(body: dict) -> str:
    """All the text a request carries, in any of the supported APIs; pictures appear as [image]."""
    parts = [_blocks_text(body.get("instructions")), _blocks_text(body.get("system"))]  # Responses, Anthropic
    for key in ("messages", "input"):
        value = body.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(_blocks_text(m.get("content")) if isinstance(m, dict) else "" for m in value)
    for key in ("prompt", "input_prefix", "input_suffix"):  # completions, llama.cpp native, infill
        value = body.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(map(str, value))
        elif isinstance(value, dict):  # native /completion with pictures: {"prompt_string": ..., "multimodal_data": [...]}
            parts.append(str(value.get("prompt_string", "")))
    return "\n".join(p for p in parts if p)


def _count_pictures(body: dict) -> int:
    count = 0
    for key in ("messages", "input"):
        for m in body.get(key) if isinstance(body.get(key), list) else []:
            content = m.get("content") if isinstance(m, dict) else None
            if isinstance(content, list):
                count += sum(1 for p in content if isinstance(p, dict) and p.get("type") in PICTURE_TYPES)
    return count


def _asked_tokens(body: dict) -> int | None:
    """The reply length the client asked for, if it is a sensible number."""
    def positive(value):
        return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None

    return next((v for v in map(positive, (body.get(k) for k in LENGTH_FIELDS)) if v), None)


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
        for choice in doc.get("choices") or []:  # OpenAI chat and completions
            piece = choice.get("text") or (choice.get("delta") or {}).get("content") or (choice.get("message") or {}).get("content")
            if isinstance(piece, str):
                pieces.append(piece)
        content = doc.get("content")  # llama.cpp native (a string) and Anthropic (a list of blocks)
        if isinstance(content, str):
            pieces.append(content)
        elif isinstance(content, list):
            pieces.extend(b["text"] for b in content if isinstance(b, dict) and isinstance(b.get("text"), str))
        delta = doc.get("delta")  # Anthropic and Responses stream events
        if isinstance(delta, dict) and isinstance(delta.get("text"), str):
            pieces.append(delta["text"])
        elif isinstance(delta, str):
            pieces.append(delta)
        for item in doc.get("output") or []:  # a plain Responses reply
            for block in (item.get("content") or []) if isinstance(item, dict) else []:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    pieces.append(block["text"])
        usage, timings = doc.get("usage") or usage, doc.get("timings") or timings
    used, timed = usage or {}, timings or {}
    native = next((d["tokens_predicted"] for d in reversed(docs) if isinstance(d, dict) and d.get("tokens_predicted")), None)
    tokens = used.get("completion_tokens") or used.get("output_tokens") or timed.get("predicted_n") or native
    return "".join(pieces), tokens, timed.get("predicted_per_second")


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
        for field in LENGTH_FIELDS:  # the cap goes into whatever this API calls the reply length, and into any other the client set
            if field in body or field == CAP_FIELD.get(request.path):
                body[field] = capped
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

    def default_model():
        return resolve("", ctx.models_provider(), ctx.settings.aliases, ctx.settings.default_model)

    async def relay(request: web.Request, engine_url: str) -> web.Response:
        """Hand a small request to the engine as it is, and return its answer as it is."""
        body = await request.read() if request.can_read_body else None
        headers = {"Content-Type": request.headers.get("Content-Type", "application/json")} if body else {}
        async with request.app[SESSION].request(request.method, engine_url + request.path, data=body, headers=headers) as up:
            return web.Response(body=await up.read(), status=up.status,
                                headers={"Content-Type": up.headers.get("Content-Type", "application/json")})

    async def with_engine(request: web.Request):
        """Quick endpoints that need a loaded model (tokenize and so on): the default model is started if none is running."""
        if ctx.engine.pid is not None:
            return await relay(request, ctx.engine.url)
        model = default_model()
        if model is None:
            return _error(503, "no model is loaded and no default model is set")
        try:
            return await ctx.scheduler.run_llm(priority, model, 1, lambda url: relay(request, url))
        except EngineError as exc:
            return _error(502, str(exc))

    async def props(request: web.Request):
        if ctx.engine.pid is not None:
            return await relay(request, ctx.engine.url)
        model = default_model()
        if model is None:
            return _error(503, "no default model is set")
        return web.json_response({"default_generation_settings": {"n_ctx": BUCKETS[0], "params": {}}, "total_slots": 1,
                                  "model_path": str(model.path), "modalities": {"vision": model.mmproj is not None},
                                  "chat_template": "", "build_info": "ratatosk"})

    async def health(_):
        return web.json_response({"status": "ok"})  # Ratatosk is up; the model is loaded when a request needs it

    async def engine_list(request: web.Request):
        if ctx.engine.pid is not None:
            return await relay(request, ctx.engine.url)
        return web.json_response([])

    async def unsupported(request: web.Request):
        kind = "reranking" if "rerank" in request.path else "embedding" if "embed" in request.path else None
        status, message = (501, f"no {kind} model is configured yet") if kind else (404, f"{request.path} is not supported by Ratatosk")
        ctx.db.log_request(str(route.port), route.profile, "-", 0, 0, 0, 0.0, "unsupported", client=request.remote,
                           user_agent=request.headers.get("User-Agent", "")[:200], path=request.path, stream=0, error=message)
        return _error(status, message)

    app = web.Application(client_max_size=16 * 2**20)  # a long chat history is far bigger than aiohttp's 1 MiB default
    app.on_startup.append(start_session)
    app.on_cleanup.append(close_session)
    app.router.add_get("/v1/models", list_models)
    for path in CAP_FIELD:
        app.router.add_post(path, complete)
    for path in ("/tokenize", "/detokenize", "/apply-template"):
        app.router.add_post(path, with_engine)
    app.router.add_get("/props", props)
    app.router.add_get("/health", health)
    app.router.add_get("/slots", engine_list)
    app.router.add_get("/lora-adapters", engine_list)
    app.router.add_route("*", "/{tail:.*}", unsupported)  # anything else: a clear answer, and a line in the log
    return app
