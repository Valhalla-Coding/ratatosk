"""A fake Hugging Face: a repo tree listing and file downloads that honour Range, and can cut a download short once."""
import asyncio
from dataclasses import dataclass, field

from aiohttp import web


@dataclass
class HfState:
    repos: dict = field(default_factory=dict)  # "owner/repo" -> {"file.gguf": bytes}
    cut_once_at: int | None = None  # the first download of any file stops after this many bytes
    ranges: list = field(default_factory=list)  # the Range header of every download


def make_app(state: HfState) -> web.Application:
    async def tree(request):
        repo = f"{request.match_info['owner']}/{request.match_info['repo']}"
        if repo not in state.repos:
            return web.json_response({"error": "Repository not found"}, status=404)
        return web.json_response([{"type": "file", "path": name, "size": len(data)} for name, data in state.repos[repo].items()]
                                 + [{"type": "directory", "path": "docs"}])

    async def resolve(request):
        repo = f"{request.match_info['owner']}/{request.match_info['repo']}"
        data = state.repos.get(repo, {}).get(request.match_info["file"])
        if data is None:
            return web.Response(status=404)
        header = request.headers.get("Range")
        if request.method != "HEAD":
            state.ranges.append(header)
        start = int(header.split("=")[1].rstrip("-")) if header else 0
        body = data[start:]
        headers = {"Content-Length": str(len(body)), "Accept-Ranges": "bytes"}
        if header:
            headers["Content-Range"] = f"bytes {start}-{len(data) - 1}/{len(data)}"
        if request.method == "HEAD":
            headers["Content-Length"] = str(len(data))
            return web.Response(status=200, headers=headers)
        resp = web.StreamResponse(status=206 if header else 200, headers=headers)
        await resp.prepare(request)
        if state.cut_once_at is not None:
            cut, state.cut_once_at = state.cut_once_at, None
            await resp.write(body[:cut])
            await asyncio.sleep(0.2)  # let the bytes reach the client before the connection drops
            request.transport.abort()
            return resp
        await resp.write(body)
        return resp

    app = web.Application()
    app.router.add_get("/api/models/{owner}/{repo}/tree/main", tree)
    app.router.add_route("*", "/{owner}/{repo}/resolve/main/{file}", resolve)
    return app
