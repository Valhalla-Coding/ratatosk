import asyncio
import re
from dataclasses import asdict, fields
from pathlib import Path
from typing import Awaitable, Callable

from aiohttp import web

from ratatosk.comfy import Comfy
from ratatosk.gateway_llm import AppContext
from ratatosk.settings import Settings, save_settings, settings_from_dict

INT_FIELDS = ("ui_port", "engine_port", "max_tokens_cap", "max_tokens_cap_plain", "idle_minutes", "reserve_gb", "image_wait_seconds", "threads",
              "keep_content", "image_max_tokens", "reply_reserve")
STR_FIELDS = ("comfy_url", "engine_dir", "default_model", "data_dir")
PORT_FIELDS = ("ui_port", "engine_port")
KINDS, PROFILES = ("llm", "comfy"), ("plain", "rp")


def parse_gpu(text: str) -> dict | None:
    """First line of `nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits`."""
    match = re.match(r"\s*(\d+)\s*,\s*(\d+)\s*$", (text.strip().splitlines() or [""])[0])
    return {"used_mb": int(match[1]), "total_mb": int(match[2])} if match else None


def label_process(name: str) -> str:
    low = name.lower()
    if "/ratatosk/engine" in low:
        return "Ratatosk engine"
    if ".lmstudio" in low or "lm-studio" in low:
        return "LM Studio engine"
    if "comfyui" in low:
        return "ComfyUI"
    return name.rsplit("/", 1)[-1]


def parse_gpu_processes(text: str) -> list[dict]:
    """`nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits`, biggest first."""
    rows = []
    for line in text.strip().splitlines():
        try:
            pid, rest = line.split(",", 1)
            name, mb = rest.rsplit(",", 1)
            rows.append({"pid": int(pid), "name": label_process(name.strip()), "mb": int(mb)})
        except ValueError:
            continue
    return sorted(rows, key=lambda r: -r["mb"])


async def _nvidia_smi(*args: str) -> str | None:
    try:
        proc = await asyncio.create_subprocess_exec("nvidia-smi", *args, "--format=csv,noheader,nounits",
                                                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        out, _ = await asyncio.wait_for(proc.communicate(), 5)
    except (OSError, asyncio.TimeoutError):
        return None
    return out.decode(errors="replace")


async def read_gpu() -> dict | None:
    """Memory used and total, plus which programs hold it (so a full GPU is never unexplained)."""
    memory = await _nvidia_smi("--query-gpu=memory.used,memory.total")
    gpu = parse_gpu(memory) if memory else None
    if gpu:
        gpu["processes"] = parse_gpu_processes(await _nvidia_smi("--query-compute-apps=pid,process_name,used_memory") or "")
    return gpu


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
    if "mmproj_on_gpu" in data and not isinstance(data["mmproj_on_gpu"], bool):
        raise ValueError("mmproj_on_gpu must be true or false")
    if "request_defaults" in data and not (isinstance(data["request_defaults"], dict) and all(
            isinstance(k, str) and isinstance(v, dict) for k, v in data["request_defaults"].items())):
        raise ValueError("request_defaults must map a model name to an object of request fields")
    for key in STR_FIELDS:
        if key in data and not isinstance(data[key], str):
            raise ValueError(f"{key} must be text")
    for key, meaning in (("aliases", "names to model ids or paths"), ("mmproj_files", "model ids to vision file paths")):
        if key in data and not (isinstance(data[key], dict)
                                and all(isinstance(k, str) and isinstance(v, str) for k, v in data[key].items())):
            raise ValueError(f"{key} must map {meaning}")
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
            "recent": ctx.db.recent(20),
        })

    async def request_detail(request: web.Request):
        row = ctx.db.get(int(request.match_info["id"]))
        return web.json_response(row) if row else web.json_response({"error": "no such request"}, status=404)

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
    app.router.add_get(r"/api/requests/{id:\d+}", request_detail)
    app.router.add_get("/api/settings", get_settings)
    app.router.add_put("/api/settings", put_settings)
    return app
