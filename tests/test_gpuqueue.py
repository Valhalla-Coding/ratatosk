import asyncio

import pytest

from ratatosk.gpuqueue import P_IMAGE, P_LLM, P_RP, P_TUNE, GpuQueue, Job


@pytest.fixture
async def q():
    queue = GpuQueue()
    queue.start()
    yield queue
    await queue.stop()


def recorder(order, label):
    async def run():
        order.append(label)
        return label

    return run


async def hold(q, gate, name="block", priority=P_LLM):
    """Submit a job that runs until `gate` is set; returns its submit task once it is running."""

    async def run():
        await gate.wait()

    task = asyncio.create_task(q.submit(Job(name, priority, run)))
    await asyncio.sleep(0.01)
    return task


async def test_priority_order(q):
    order, gate = [], asyncio.Event()
    blocker = await hold(q, gate)
    tasks = [asyncio.create_task(q.submit(Job(f"p{p}", p, recorder(order, p)))) for p in (P_IMAGE, P_RP, P_LLM)]
    await asyncio.sleep(0.01)
    gate.set()
    await asyncio.gather(blocker, *tasks)
    assert order == [P_RP, P_LLM, P_IMAGE]


async def test_one_at_a_time(q):
    active, peak = 0, 0

    async def run():
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1

    await asyncio.gather(*(q.submit(Job(f"j{i}", P_LLM, run)) for i in range(5)))
    assert peak == 1


async def test_cancel_pending_never_runs(q):
    order, gate = [], asyncio.Event()
    blocker = await hold(q, gate)
    victim = asyncio.create_task(q.submit(Job("victim", P_LLM, recorder(order, "victim"))))
    await asyncio.sleep(0.01)
    assert q.snapshot()["pending"] == ["victim"]
    victim.cancel()
    await asyncio.sleep(0.01)
    assert q.snapshot()["pending"] == []
    gate.set()
    await blocker
    await q.submit(Job("after", P_LLM, recorder(order, "after")))
    assert order == ["after"]


async def test_cancel_running_frees_queue(q):
    state = {"cancelled": False}

    async def run():
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise

    running = asyncio.create_task(q.submit(Job("long", P_LLM, run)))
    await asyncio.sleep(0.01)
    running.cancel()
    order = []
    await asyncio.wait_for(q.submit(Job("next", P_LLM, recorder(order, "next"))), 2)
    assert state["cancelled"] and order == ["next"]


async def test_after_runs_after_submit_returns_before_next_job(q):
    events = []

    async def run():
        events.append("run")
        return "result"

    async def after():
        await asyncio.sleep(0.05)
        events.append("after")

    assert await q.submit(Job("first", P_IMAGE, run, after)) == "result"
    assert events == ["run"]  # the caller already has its answer; the hold is still going
    await q.submit(Job("second", P_IMAGE, recorder(events, "second")))
    assert events == ["run", "after", "second"]


async def test_tune_job_preempted(q):
    state = {"cancelled": False}

    async def tune():
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise

    tuning = asyncio.create_task(q.submit(Job("tune", P_TUNE, tune)))
    await asyncio.sleep(0.01)
    order = []
    await asyncio.wait_for(q.submit(Job("chat", P_RP, recorder(order, "chat"))), 2)
    with pytest.raises(asyncio.CancelledError):
        await tuning
    assert state["cancelled"] and order == ["chat"]


async def test_snapshot_shape(q):
    gate = asyncio.Event()
    blocker = await hold(q, gate, name="block")
    a = asyncio.create_task(q.submit(Job("late", P_IMAGE, recorder([], 1))))
    b = asyncio.create_task(q.submit(Job("early", P_RP, recorder([], 1))))
    await asyncio.sleep(0.01)
    assert q.snapshot() == {"running": "block", "pending": ["early", "late"]}
    gate.set()
    await asyncio.gather(blocker, a, b)
    assert q.snapshot() == {"running": None, "pending": []}
