import asyncio
from pathlib import Path

import pytest

from ratatosk.engine import EngineError
from ratatosk.gpuqueue import P_LLM, GpuQueue
from ratatosk.models import Model
from ratatosk.scheduler import GpuBusy, Scheduler
from ratatosk.settings import Settings

MODEL = Model("m", Path("/m/a.gguf"), 1, Path("/m"))


class FakeEngine:
    url = "http://engine"

    def __init__(self, calls):
        self.calls, self.loaded, self.fail, self.running = calls, None, False, False

    @property
    def pid(self):
        return 1 if self.running else None

    async def ensure(self, model, ctx):
        self.calls.append(("ensure", ctx))
        if self.fail:
            raise EngineError("no memory")
        self.loaded, self.running = (model, ctx), True

    async def stop(self):
        self.calls.append(("stop",))
        self.loaded, self.running = None, False


class FakeComfy:
    def __init__(self, calls, pending=()):
        self.calls, self._pending = calls, list(pending)

    async def free(self):
        self.calls.append(("free",))

    async def pending(self):
        self.calls.append(("pending",))
        return self._pending.pop(0) if self._pending else 0


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


async def ok(url):
    return "ok:" + url


@pytest.fixture
async def rig():
    calls = []
    queue = GpuQueue()
    queue.start()
    clock = Clock()
    engine, comfy = FakeEngine(calls), FakeComfy(calls)

    def make(settings=None, pending=()):
        comfy._pending = list(pending)
        return Scheduler(settings or Settings(), queue, engine, comfy, clock=clock, poll_seconds=0.01)

    yield make, calls, engine, clock, queue
    await queue.stop()


async def image_ok():
    return "image"


async def test_llm_after_image_frees_comfy_first(rig):
    make, calls, *_ = rig
    s = make()
    await s.run_image(image_ok)
    calls.clear()
    assert await s.run_llm(P_LLM, MODEL, 5000, ok) == "ok:http://engine"
    assert calls == [("free",), ("ensure", 8192)]


async def test_image_after_llm_stops_engine_first(rig):
    make, calls, *_ = rig
    s = make()
    await s.run_llm(P_LLM, MODEL, 5000, ok)
    calls.clear()

    async def handler():
        calls.append(("handler",))

    await s.run_image(handler)
    assert calls[:2] == [("stop",), ("handler",)]


async def test_llm_after_llm_does_neither(rig):
    make, calls, *_ = rig
    s = make()
    await s.run_llm(P_LLM, MODEL, 5000, ok)
    calls.clear()
    await s.run_llm(P_LLM, MODEL, 5000, ok)
    assert ("free",) not in calls and ("stop",) not in calls


async def test_bucket_used(rig):
    make, calls, *_ = rig
    s = make()
    await s.run_llm(P_LLM, MODEL, 11280, ok)
    assert ("ensure", 16384) in calls


async def test_idle_stops_engine(rig):
    make, calls, engine, clock, _ = rig
    s = make()
    await s.run_llm(P_LLM, MODEL, 5000, ok)
    clock.t += 29 * 60
    await s.idle_check()
    assert engine.running
    clock.t += 2 * 60
    await s.idle_check()
    assert not engine.running


async def test_engine_failure_fails_job_and_queue_continues(rig):
    make, calls, engine, *_ = rig
    s = make()
    engine.fail = True
    with pytest.raises(EngineError):
        await s.run_llm(P_LLM, MODEL, 5000, ok)
    engine.fail = False
    assert await s.run_llm(P_LLM, MODEL, 5000, ok) == "ok:http://engine"


async def test_image_holds_slot_until_comfy_idle(rig):
    make, calls, *_ = rig
    s = make(pending=[2, 1, 0])
    await s.run_image(image_ok)
    await s.run_llm(P_LLM, MODEL, 5000, ok)
    assert calls.index(("ensure", 8192)) > max(i for i, c in enumerate(calls) if c == ("pending",))
    assert calls.count(("pending",)) == 3


async def test_image_wait_exceeded_raises_gpubusy(rig):
    make, calls, engine, clock, queue = rig
    s = make(Settings(image_wait_seconds=0.05))
    gate = asyncio.Event()

    async def slow(url):
        await gate.wait()

    long_job = asyncio.create_task(s.run_llm(P_LLM, MODEL, 5000, slow))
    await asyncio.sleep(0.02)
    with pytest.raises(GpuBusy):
        await s.run_image(image_ok)
    assert queue.snapshot()["pending"] == []  # the timed-out request is not left in the queue
    gate.set()
    await long_job
    assert await s.run_llm(P_LLM, MODEL, 5000, ok) == "ok:http://engine"
