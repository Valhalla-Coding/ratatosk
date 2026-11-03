import json
import socket
import sys
import time
from pathlib import Path

import pytest

from ratatosk.engine import Engine, EngineError
from ratatosk.settings import Settings

FAKE = str(Path(__file__).parent / "fakes" / "fake_llama.py")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def fake_argv(port):
    return lambda model, ctx: [sys.executable, FAKE, "--model", str(model), "--ctx-size", str(ctx), "--port", str(port)]


@pytest.fixture
def make_engine(tmp_path, monkeypatch):
    engines = []

    def make(**kwargs):
        port = free_port()
        settings = Settings(engine_port=port, data_dir=str(tmp_path))
        engine = Engine(settings, argv_override=fake_argv(port), poll_seconds=0.1, **kwargs)
        engines.append(engine)
        return engine

    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "fake.log"))
    yield make
    for e in engines:  # safety net: tests stop their engines themselves
        if e._proc and e._proc.returncode is None:
            e._proc.kill()


def test_argv_exact():
    settings = Settings(engine_dir="/e", engine_port=8081, threads=3)
    model = Path("/m/a.gguf")
    assert Engine(settings).argv(model, 16384) == [
        "/e/backend/llama-server", "--model", str(model), "--ctx-size", "16384",
        "--host", "127.0.0.1", "--port", "8081", "--fit", "on", "--flash-attn", "on",
        "--parallel", "1", "--threads", "3",
    ]


async def test_ensure_starts_and_waits_for_health(make_engine, monkeypatch):
    monkeypatch.setenv("FAKE_START_DELAY", "1")
    engine = make_engine()
    started = time.monotonic()
    try:
        await engine.ensure(Path("a.gguf"), 8192)
        assert time.monotonic() - started >= 1
        assert engine.loaded == (Path("a.gguf"), 8192)
    finally:
        await engine.stop()


async def test_ensure_same_args_does_not_restart(make_engine):
    engine = make_engine()
    try:
        await engine.ensure(Path("a.gguf"), 8192)
        pid = engine.pid
        await engine.ensure(Path("a.gguf"), 8192)
        assert engine.pid == pid
    finally:
        await engine.stop()


async def test_ensure_new_ctx_restarts(make_engine):
    engine = make_engine()
    try:
        await engine.ensure(Path("a.gguf"), 8192)
        pid = engine.pid
        await engine.ensure(Path("a.gguf"), 16384)
        assert engine.pid != pid
        assert engine.loaded == (Path("a.gguf"), 16384)
    finally:
        await engine.stop()


async def test_ensure_failure_resets_state(make_engine, monkeypatch):
    monkeypatch.setenv("FAKE_EXIT", "1")
    engine = make_engine()
    with pytest.raises(EngineError):
        await engine.ensure(Path("a.gguf"), 8192)
    assert engine.loaded is None and engine.pid is None


async def test_ensure_health_timeout_fails_and_cleans_up(make_engine, monkeypatch):
    monkeypatch.setenv("FAKE_START_DELAY", "5")
    engine = make_engine(health_timeout=0.5)
    with pytest.raises(EngineError):
        await engine.ensure(Path("a.gguf"), 8192)
    assert engine.loaded is None and engine.pid is None


async def test_count_tokens(make_engine):
    engine = make_engine()
    assert await engine.count_tokens("one two three") is None  # not running
    try:
        await engine.ensure(Path("a.gguf"), 8192)
        assert await engine.count_tokens("one two three") == 3
    finally:
        await engine.stop()


async def test_stop_kills_process(make_engine):
    engine = make_engine()
    await engine.ensure(Path("a.gguf"), 8192)
    proc = engine._proc
    await engine.stop()
    assert proc.returncode is not None
    assert engine.loaded is None and engine.pid is None
    assert await engine.count_tokens("x") is None
