#!/usr/bin/env python3
"""Does a model, as served through Ratatosk, do what clients expect? Standard library only.

    python3 scripts/capability_check.py --url http://127.0.0.1:21434 --model qwen3.5-9b-obliterated

Checks text, chat, streaming, thinking, pictures, structured JSON output and tool calls, and prints one line each.
Exit code 1 if any check fails. Pictures need a model with a vision file; a model without one is told so, not failed."""
import argparse
import base64
import json
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib


def png(w=448, h=448) -> bytes:
    """Blue background, a large red square on the left, a small yellow square top right."""
    rows = []
    for y in range(h):
        row = bytearray([0])
        for x in range(w):
            if 60 <= x < 260 and 100 <= y < 360:
                row += bytes((220, 30, 30))
            elif 330 <= x < 400 and 40 <= y < 110:
                row += bytes((240, 220, 40))
            else:
                row += bytes((30, 60, 200))
        rows.append(bytes(row))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b""))


class Client:
    def __init__(self, url, model, timeout):
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout

    def post(self, path, body):
        req = urllib.request.Request(self.url + path, json.dumps({"model": self.model, **body}).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return resp.read()

    def chat(self, messages, **extra):
        return json.loads(self.post("/v1/chat/completions", {"messages": messages, **extra}))["choices"][0]["message"]


NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}


def check_models(c):
    ids = [m["id"] for m in json.load(urllib.request.urlopen(c.url + "/v1/models", timeout=30))["data"]]
    assert c.model in ids, f"{c.model!r} is not in {ids[:5]}"
    return f"{len(ids)} models listed"


def check_completion(c):
    text = json.loads(c.post("/v1/completions", {"prompt": "Count: 1, 2, 3,", "max_tokens": 12, "temperature": 0}))["choices"][0]["text"]
    assert text.strip(), "empty reply"
    return repr(text.strip()[:40])


def check_chat(c):
    reply = c.chat([{"role": "user", "content": "Say hello in three words."}], max_tokens=40, **NO_THINKING)["content"]
    assert reply and reply.strip(), "empty reply"
    return repr(reply.strip()[:50])


def check_streaming(c):
    raw = c.post("/v1/chat/completions", {"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 30, "stream": True, **NO_THINKING}).decode()
    chunks = [json.loads(l[5:]) for l in raw.splitlines() if l.startswith("data:") and "[DONE]" not in l]
    text = "".join((ch["choices"][0].get("delta") or {}).get("content") or "" for ch in chunks if ch.get("choices"))
    assert "[DONE]" in raw and text.strip(), "no [DONE] marker or no text"
    return f"{len(chunks)} chunks, {text.strip()[:30]!r}"


def check_thinking(c):
    msg = c.chat([{"role": "user", "content": "What is 17 + 25? Answer with just the number."}], max_tokens=2000)
    assert (msg.get("content") or "").strip(), f"no answer after thinking (reasoning text: {len(msg.get('reasoning_content') or '')} chars)"
    return f"answer {msg['content'].strip()[:20]!r}, thinking kept apart: {bool(msg.get('reasoning_content'))}"


def check_vision(c):
    url = "data:image/png;base64," + base64.b64encode(png()).decode()
    parts = [{"type": "text", "text": "What colour is the large square, and what colour is the background? One short sentence."},
             {"type": "image_url", "image_url": {"url": url}}]
    try:
        reply = c.chat([{"role": "user", "content": parts}], max_tokens=100, temperature=0, **NO_THINKING)["content"]
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        if e.code == 400 and "cannot read pictures" in body:
            return "SKIP: this model has no vision file"
        raise
    assert "red" in reply.lower(), f"expected 'red', got {reply!r}"
    return repr(reply.strip()[:70])


def check_structured_output(c):
    schema = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"], "additionalProperties": False}
    reply = c.chat([{"role": "user", "content": "Invent a pet: a name and an age in years."}], max_tokens=100, temperature=0,
                   response_format={"type": "json_schema", "json_schema": {"name": "pet", "strict": True, "schema": schema}}, **NO_THINKING)["content"]
    data = json.loads(reply)
    assert isinstance(data.get("name"), str) and isinstance(data.get("age"), int), f"wrong shape: {data}"
    return json.dumps(data)


def check_tools(c):
    tool = {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
    msg = c.chat([{"role": "user", "content": "What is the weather in Paris right now?"}], tools=[tool], tool_choice="auto", max_tokens=200, temperature=0, **NO_THINKING)
    calls = msg.get("tool_calls") or []
    assert calls, f"no tool call; the model said {(msg.get('content') or '')[:80]!r}"
    args = json.loads(calls[0]["function"]["arguments"])
    assert calls[0]["function"]["name"] == "get_weather" and "paris" in str(args.get("city", "")).lower(), f"wrong call: {calls[0]['function']}"
    return f"get_weather({args})"


CHECKS = [("models listed", check_models), ("plain text", check_completion), ("chat", check_chat), ("streaming", check_streaming),
          ("thinking", check_thinking), ("pictures", check_vision), ("structured JSON output", check_structured_output), ("tool calls", check_tools)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    client, failed = Client(args.url, args.model, args.timeout), 0
    for name, check in CHECKS:
        started = time.time()
        try:
            note, status = check(client), "PASS"
            if note.startswith("SKIP"):
                status = "SKIP"
        except Exception as exc:  # a failed check must not stop the others
            note, status, failed = f"{type(exc).__name__}: {exc}"[:200], "FAIL", failed + 1
        print(f"{status:4}  {name:24} {time.time() - started:5.1f}s  {note}", flush=True)
    print("all checks passed" if not failed else f"{failed} check(s) failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
