import asyncio
import re
from dataclasses import asdict, fields
from pathlib import Path
from typing import Awaitable, Callable

from aiohttp import web

from ratatosk.comfy import Comfy
from ratatosk.gateway_llm import AppContext
from ratatosk.settings import Settings, save_settings, settings_from_dict

INT_FIELDS = ("ui_port", "engine_port", "max_tokens_cap", "idle_minutes", "reserve_gb", "image_wait_seconds", "threads")
STR_FIELDS = ("comfy_url", "engine_dir", "default_model", "data_dir")
PORT_FIELDS = ("ui_port", "engine_port")
KINDS, PROFILES = ("llm", "comfy"), ("plain", "rp")


def parse_gpu(text: str) -> dict | None:
    """First line of `nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits`."""
    match = re.match(r"\s*(\d+)\s*,\s*(\d+)\s*$", (text.strip().splitlines() or [""])[0])
    return {"used_mb": int(match[1]), "total_mb": int(match[2])} if match else None


async def read_gpu() -> dict | None:
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        out, _ = await asyncio.wait_for(proc.communicate(), 5)
    except (OSError, asyncio.TimeoutError):
        return None
    return parse_gpu(out.decode(errors="replace"))


def _is_port(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 65535


def validate_settings(data) -> None:
    if not isinstance(data, dict):
        raise ValueError("the body must be a JSON object")
    for key in INT_FIELDS:
        if key in data and (isinstance(data[key], bool) or not isinstance(data[key], int) or data[key] < 1):
            raise ValueError(f"{key} must be a positive whole number")
    for key in PORT_FIELDS:
        if key in data and not _is_port(data[key]):
            raise ValueError(f"{key} must be a port number from 1 to 65535")
    for key in STR_FIELDS:
        if key in data and not isinstance(data[key], str):
            raise ValueError(f"{key} must be text")
    if "aliases" in data and not (isinstance(data["aliases"], dict)
                                  and all(isinstance(k, str) and isinstance(v, str) for k, v in data["aliases"].items())):
        raise ValueError("aliases must map names to model ids or paths")
    if "model_folders" in data and not (isinstance(data["model_folders"], list) and all(
            isinstance(f, dict) and isinstance(f.get("path"), str) for f in data["model_folders"])):
        raise ValueError("model_folders must be a list of objects with a path")
    for route in data.get("routes", []) if isinstance(data.get("routes", []), list) else [None]:
        if not isinstance(route, dict) or not _is_port(route.get("port")):
            raise ValueError("each route needs a port from 1 to 65535")
        if route.get("kind") not in KINDS:
            raise ValueError(f"route kind must be one of {KINDS}")
        if route.get("profile", "plain") not in PROFILES:
            raise ValueError(f"route profile must be one of {PROFILES}")


def validate_ports(effective: dict) -> None:
    """The settings as they would be after the change: a service that cannot bind would crash-loop with no UI left."""
    routes = effective.get("routes") or []
    if not routes:
        raise ValueError("at least one route is needed")
    ports = [r["port"] for r in routes] + [effective["ui_port"], effective["engine_port"]]
    duplicates = sorted({p for p in ports if ports.count(p) > 1})
    if duplicates:
        raise ValueError(f"each port can be used once (routes, the UI and the engine); repeated: {duplicates}")
    low = sorted(p for p in ports if p < 1024)
    if low:
        raise ValueError(f"ports below 1024 need root and cannot be used: {low}")


def make_ui_app(ctx: AppContext, comfy: Comfy, settings_path: Path,
                gpu_reader: Callable[[], Awaitable[dict | None]] = read_gpu) -> web.Application:
    async def status(_):
        loaded = ctx.engine.loaded
        try:
            pending = await comfy.pending()
        except Exception:  # ComfyUI down or restarting
            pending = None
        return web.json_response({
            "queue": ctx.scheduler.queue.snapshot(),
            "engine": {"model": Path(loaded[0]).name, "ctx": loaded[1]} if loaded else None,
            "comfy_pending": pending,
            "gpu": await gpu_reader(),
            "recent": ctx.db.recent(10),
        })

    async def get_settings(_):
        return web.json_response(asdict(ctx.settings))

    async def put_settings(request: web.Request):
        try:
            data = await request.json()
            validate_settings(data)
            validate_ports({**asdict(ctx.settings), **data})
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        candidate = settings_from_dict({**asdict(ctx.settings), **data})
        for f in fields(Settings):
            setattr(ctx.settings, f.name, getattr(candidate, f.name))
        save_settings(ctx.settings, settings_path)
        return web.json_response(asdict(ctx.settings))

    async def index(_):
        return web.FileResponse(Path(__file__).parent / "static" / "index.html")

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/api/status", status)
    app.router.add_get("/api/settings", get_settings)
    app.router.add_put("/api/settings", put_settings)
    return app
