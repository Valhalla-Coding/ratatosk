import asyncio
import heapq
import itertools
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

P_RP, P_LLM, P_IMAGE, P_TUNE = 0, 1, 2, 10**6  # tuning stays behind every rule line


@dataclass
class Job:
    name: str
    priority: int
    run: Callable[[], Awaitable[Any]]
    after: Callable[[], Awaitable[None]] | None = None  # holds the GPU after `run` has returned its result
    preempt: bool = False  # cancel the running job if it ranks lower than this one
    owner: str = ""  # who is asking; named in the error the cancelled job's caller gets
    door: int | None = None  # the port it came in on, for the "busy" light in the UI
    on_preempt: Callable[[], Awaitable[None]] | None = None  # cleanup if THIS job is cancelled by another (e.g. tell ComfyUI to stop)


class Preempted(Exception):
    """The job was cancelled to make room for a "drop everything" request."""

    def __init__(self, by: str):
        super().__init__(f"dropped by {by}")
        self.by = by


@dataclass(eq=False)
class _Entry:
    job: Job
    fut: asyncio.Future
    task: asyncio.Task | None = None
    dead: bool = False  # cancelled by its caller or preempted
    preempted: bool = False


class GpuQueue:
    """One job at a time, lowest priority number first. Only P_TUNE jobs are preempted."""

    def __init__(self):
        self._heap: list = []
        self._seq = itertools.count()
        self._wake = asyncio.Event()
        self._running: _Entry | None = None
        self._runner: asyncio.Task | None = None

    def start(self) -> None:
        self._runner = asyncio.get_running_loop().create_task(self._loop())

    async def stop(self) -> None:
        if self._runner:
            self._runner.cancel()
            await asyncio.gather(self._runner, return_exceptions=True)

    async def submit(self, job: Job) -> Any:
        entry = _Entry(job, asyncio.get_running_loop().create_future())
        heapq.heappush(self._heap, (job.priority, next(self._seq), entry))
        self._wake.set()
        running = self._running
        if running and running.job.priority == P_TUNE and job.priority < P_TUNE:
            self._kill(running)
        elif running and job.preempt and running.job.priority > job.priority:
            self._kill(running, Preempted(job.owner))
        try:
            return await entry.fut
        except asyncio.CancelledError:
            self._kill(entry)
            raise

    def running_door(self) -> int | None:
        return self._running.job.door if self._running else None

    def snapshot(self) -> dict:
        pending = [e.job.name for _, _, e in sorted(self._heap) if not e.dead]
        return {"running": self._running.job.name if self._running else None, "pending": pending}

    @staticmethod
    def _kill(entry: _Entry, reason: Preempted | None = None) -> None:
        entry.dead = True
        entry.preempted = reason is not None
        if reason is not None and not entry.fut.done():
            entry.fut.set_exception(reason)
        else:
            entry.fut.cancel()
        if entry.task:
            entry.task.cancel()

    def _pop(self) -> _Entry | None:
        while self._heap:
            entry = heapq.heappop(self._heap)[2]
            if not entry.dead:
                return entry
        return None

    async def _loop(self) -> None:
        while True:
            entry = self._pop()
            if entry is None:
                self._wake.clear()
                await self._wake.wait()
                continue
            self._running = entry
            try:
                await self._execute(entry)
            finally:
                self._running = None

    async def _execute(self, entry: _Entry) -> None:
        entry.task = asyncio.ensure_future(entry.job.run())
        try:
            result = await entry.task
            if not entry.fut.done():
                entry.fut.set_result(result)
            if entry.job.after:
                entry.task = asyncio.ensure_future(entry.job.after())  # still killable while it holds the GPU
                await entry.task
        except asyncio.CancelledError:
            if not entry.dead:
                raise  # the runner itself is being stopped
            if entry.preempted and entry.job.on_preempt:
                try:
                    await entry.job.on_preempt()
                except Exception:  # ponytail: a failed cleanup must not stop the queue; add logging when the service logs
                    pass
        except Exception as exc:
            if not entry.fut.done():
                entry.fut.set_exception(exc)  # a failing hold after the result is dropped: the future is already done
