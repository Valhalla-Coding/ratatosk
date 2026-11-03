"""A fake llama-server for tests. Takes the same flags; behaviour is steered by environment variables:
FAKE_LOG (file that gets one JSON line per start and per request), FAKE_START_DELAY (seconds before it
listens), FAKE_CHUNK_DELAY (seconds between streamed chunks), FAKE_EXIT=1 (exit at once)."""
import argparse
import asyncio
import json
import os
import sys
import time

from aiohttp import web

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--ctx-size", type=int, default=0)
parser.add_argument("--model", default="")
args, _ = parser.parse_known_args()

if os.environ.get("FAKE_EXIT") == "1":
    sys.exit(1)

LOG = os.environ.get("FAKE_LOG")
CHUNK_DELAY = float(os.environ.get("FAKE_CHUNK_DELAY", "0"))


def log(record):
    if LOG:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


async def health(_):
    return web.json_response({"status": "ok"})


async def tokenize(request):
    body = await request.json()
    return web.json_response({"tokens": [0] * len(str(body.get("content", "")).split())})


async def completions(request):
    body = await request.json()
    log({"event": "request", "path": request.path, "body": body, "ctx": args.ctx_size, "pid": os.getpid()})
    if not body.get("stream"):
        return web.json_response({"choices": [{"text": "hello"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    delay = float(body["slow"]) if body.get("slow") else CHUNK_DELAY  # a request can ask for slow chunks
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    for i in range(3):
        await resp.write(f'data: {json.dumps({"choices": [{"text": f"t{i}"}]})}\n\n'.encode())
        if body.get("drop"):  # the engine dies mid-reply
            request.transport.abort()
            return resp
        await asyncio.sleep(delay)
    await resp.write(b"data: [DONE]\n\n")
    return resp


async def started(_):
    log({"event": "start", "ctx": args.ctx_size, "model": args.model, "pid": os.getpid()})


app = web.Application()
app.on_startup.append(started)
app.router.add_get("/health", health)
app.router.add_post("/tokenize", tokenize)
app.router.add_post("/v1/completions", completions)
app.router.add_post("/v1/chat/completions", completions)

time.sleep(float(os.environ.get("FAKE_START_DELAY", "0")))
web.run_app(app, host="127.0.0.1", port=args.port, print=None)
