"""Model management for the Settings page: list, delete, and add by Hugging Face link."""
import asyncio
import itertools
import re
import shutil
from pathlib import Path

import aiohttp
from aiohttp import web

from ratatosk.gateway_llm import AppContext
from ratatosk.models import Model
from ratatosk.settings import Settings, save_settings

MARGIN = 2**30  # a download must leave this much of the disk free
CHUNK = 1 << 20
LINK = re.compile(r"^(?:https?://)?(?:www\.)?huggingface\.co/([\w.-]+/[\w.-]+)|^([\w.-]+/[\w.-]+)$")
REPO = re.compile(r"^[\w.-]+/[\w.-]+$")
FILE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.gguf$")  # a plain file name: no folders, no ".."


def _error(status: int, message: str) -> web.Response:
    return web.json_response({"error": message}, status=status)


def _points_at(value: str, model: Model) -> bool:
    return value.lower() in (model.id, str(model.path).lower())


def describe(settings: Settings, model: Model) -> dict:
    caps = {"vision": False, "tools": False, "reasoning": False, **model.capabilities}
    return {"id": model.id, "file": model.path.name, "path": str(model.path), "size": model.size, "capabilities": caps,
            "names": [n for n, v in settings.aliases.items() if _points_at(v, model)],
            "default": bool(settings.default_model) and _points_at(settings.default_model, model)}


def add_routes(app: web.Application, ctx: AppContext, settings_path: Path) -> None:
    settings = ctx.settings
    downloads: dict[int, dict] = {}
    tasks: set[asyncio.Task] = set()
    counter = itertools.count(1)

    def base() -> str:
        return settings.hf_base_url.rstrip("/")

    async def list_models(_):
        return web.json_response([describe(settings, m) for m in ctx.models_provider()])

    async def delete_model(request: web.Request):
        models = ctx.models_provider()
        model = next((m for m in models if m.id == request.match_info["id"]), None)
        if model is None:
            return _error(404, "no such model")
        loaded = ctx.engine.loaded
        if loaded and Path(loaded[0]) == model.path:
            return _error(409, "this model is in use right now; wait until the engine has unloaded it")
        model.path.unlink(missing_ok=True)
        vision = model.mmproj
        shared = any(m.mmproj == vision for m in models if m is not model)
        if vision and not shared and vision.is_relative_to(model.folder):
            vision.unlink(missing_ok=True)  # a vision file the user pointed at elsewhere is theirs; leave it
        settings.aliases = {n: v for n, v in settings.aliases.items() if not _points_at(v, model)}
        settings.mmproj_files = {k: v for k, v in settings.mmproj_files.items() if k != model.id}
        if settings.default_model and _points_at(settings.default_model, model):
            settings.default_model = ""
        save_settings(settings, settings_path)
        return web.json_response({"deleted": model.id})

    async def find(request: web.Request):
        try:
            url = (await request.json()).get("url", "")
        except (ValueError, AttributeError):
            url = ""
        match = LINK.match(url.strip()) if isinstance(url, str) else None
        if not match:
            return _error(400, "that does not look like a Hugging Face link (owner/model or a huggingface.co address)")
        repo = match.group(1) or match.group(2)
        try:
            async with aiohttp.ClientSession() as session, session.get(f"{base()}/api/models/{repo}/tree/main") as resp:
                if resp.status == 404:
                    return _error(404, f"Hugging Face has no repository {repo}")
                if resp.status >= 400:
                    return _error(502, f"Hugging Face answered {resp.status}")
                tree = await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            return _error(502, f"could not reach Hugging Face: {exc}")
        found = [{"file": f["path"], "size": (f.get("lfs") or {}).get("size") or f.get("size", 0)}
                 for f in tree if f.get("type") == "file" and f["path"].lower().endswith(".gguf")]
        return web.json_response({"repo": repo, "files": [f for f in found if "mmproj" not in f["file"].lower()],
                                  "vision": [f for f in found if "mmproj" in f["file"].lower()]})

    async def size_of(session: aiohttp.ClientSession, repo: str, name: str) -> int:
        async with session.head(f"{base()}/{repo}/resolve/main/{name}", allow_redirects=True) as resp:
            if resp.status == 404:
                raise FileNotFoundError(name)
            resp.raise_for_status()
            return int(resp.headers["Content-Length"])

    async def fetch(session: aiohttp.ClientSession, entry: dict, repo: str, name: str, size: int, folder: Path) -> None:
        dest = folder / name
        if dest.exists() and dest.stat().st_size == size:
            entry["done"] += size
            return
        part = folder / f"{name}.part"
        have = part.stat().st_size if part.exists() else 0
        if have > size:
            part.unlink()
            have = 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        async with session.get(f"{base()}/{repo}/resolve/main/{name}", headers=headers) as resp:
            resp.raise_for_status()
            if have and resp.status != 206:  # the server ignored the range: start over
                have = 0
            entry["done"] += have
            with open(part, "ab" if have else "wb") as out:
                async for chunk in resp.content.iter_chunked(CHUNK):
                    out.write(chunk)
                    entry["done"] += len(chunk)
        if part.stat().st_size != size:
            raise OSError(f"{name}: got {part.stat().st_size} of {size} bytes")
        part.replace(dest)

    async def run(entry: dict, repo: str, names: list[tuple[str, int]], folder: Path) -> None:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=15)) as session:
                for name, size in names:
                    await fetch(session, entry, repo, name, size, folder)
            entry["status"] = "done"
        except asyncio.CancelledError:
            entry["status"], entry["error"] = "error", "stopped"
            raise
        except Exception as exc:
            entry["status"], entry["error"] = "error", f"{type(exc).__name__}: {exc}"

    async def download(request: web.Request):
        try:
            body = await request.json()
        except ValueError:
            body = None
        repo, name = (body or {}).get("repo", ""), (body or {}).get("file", "")
        vision = (body or {}).get("vision") or None
        if not (isinstance(repo, str) and REPO.match(repo) and not {".", ".."} & set(repo.split("/"))
                and isinstance(name, str) and FILE.match(name)
                and (vision is None or (isinstance(vision, str) and FILE.match(vision)))):
            return _error(400, "send a repository (owner/model) and a plain .gguf file name")
        if not settings.model_folders:
            return _error(400, "there is no model folder to save into")
        if any(d["status"] == "running" and d["repo"] == repo and d["file"] == name for d in downloads.values()):
            return _error(409, "that file is already downloading")
        root = Path(settings.model_folders[0]["path"])
        wanted = [name] + ([vision] if vision else [])
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                sizes = [(n, await size_of(session, repo, n)) for n in wanted]
        except FileNotFoundError as exc:
            return _error(404, f"{exc} is not in {repo}")
        except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, ValueError) as exc:
            return _error(502, f"could not reach Hugging Face: {exc}")
        folder = root / repo
        have = sum(p.stat().st_size for n, _ in sizes if (p := folder / f"{n}.part").exists())
        need = sum(s for _, s in sizes) - have + MARGIN
        probe = root if root.exists() else root.parent
        if shutil.disk_usage(probe)[2] < need:
            return _error(507, f"not enough room: about {need // 2**20} MB are needed (including 1 GB left free)")
        folder.mkdir(parents=True, exist_ok=True)
        entry = {"id": next(counter), "repo": repo, "file": name, "done": 0, "total": sum(s for _, s in sizes),
                 "status": "running", "error": None}
        downloads[entry["id"]] = entry
        task = asyncio.ensure_future(run(entry, repo, sizes, folder))
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        return web.json_response(entry, status=202)

    async def list_downloads(_):
        return web.json_response(sorted(downloads.values(), key=lambda d: d["id"]))

    async def stop_downloads(_):
        for task in list(tasks):
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    app.on_cleanup.append(stop_downloads)
    app.router.add_get("/api/models", list_models)
    app.router.add_get("/api/models/downloads", list_downloads)
    app.router.add_post("/api/models/find", find)
    app.router.add_post("/api/models/download", download)
    app.router.add_delete(r"/api/models/{id}", delete_model)
