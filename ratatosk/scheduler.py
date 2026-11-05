import asyncio
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from ratatosk.buckets import BUCKETS, pick_bucket
from ratatosk.comfy import Comfy
from ratatosk.engine import Engine
from ratatosk.gpuqueue import P_IMAGE, P_LLM, GpuQueue, Job
from ratatosk.models import Model
from ratatosk.settings import Settings

RECENT_SECONDS = 600
IMAGE_HOLD_MAX_SECONDS = 1800  # a wedged ComfyUI must not hold the GPU forever


class GpuBusy(Exception):
    pass


class Scheduler:
    """Decides what has to move out of GPU memory before the next job runs."""

    def __init__(self, settings: Settings, queue: GpuQueue, engine: Engine, comfy: Comfy,
                 clock: Callable[[], float] = time.monotonic, poll_seconds: float = 1.0):
        self.settings, self.queue, self.engine, self.comfy = settings, queue, engine, comfy
        self.clock, self.poll_seconds = clock, poll_seconds
        # At start ComfyUI may still hold memory from before, so the first text job frees it.
        self.last_kind: str | None = "image"
        self._recent: list[tuple[float, int]] = []
        self._last_activity = clock()

    def limit_bucket(self) -> int:
        return BUCKETS[-1]

    async def run_llm(self, priority: int, model: Model, needed: int, handler: Callable[[str], Awaitable[Any]]) -> Any:
        async def run():
            now = self.clock()
            current = self.engine.loaded[1] if self.engine.loaded else None
            bucket = pick_bucket(needed, current, self._recent, now, self.limit_bucket())
            if bucket is None:
                raise ValueError("the needed context is larger than the largest bucket")
            if self.last_kind == "image":
                try:
                    await self.comfy.free()
                except Exception:  # ponytail: ComfyUI down or busy; carry on, a failed engine start will say if memory is short
                    pass
            try:
                await self.engine.ensure(Path(model.path), bucket, model.mmproj)
            except BaseException:
                self.last_kind = None
                raise
            self.last_kind = "llm"
            self._recent = [(t, n) for t, n in self._recent if now - t <= RECENT_SECONDS] + [(now, needed)]
            try:
                return await handler(self.engine.url)
            finally:
                self._last_activity = self.clock()

        return await self.queue.submit(Job(f"llm:{model.id}", priority, run))

    async def run_image(self, handler: Callable[[], Awaitable[Any]]) -> Any:
        started = asyncio.Event()

        async def run():
            started.set()
            if self.engine.pid is not None:
                await self.engine.stop()
            self.last_kind = "image"
            try:
                return await handler()
            finally:
                self._last_activity = self.clock()

        async def after():
            deadline = self.clock() + IMAGE_HOLD_MAX_SECONDS
            while await self.comfy.pending() > 0 and self.clock() < deadline:
                await asyncio.sleep(self.poll_seconds)
            self._last_activity = self.clock()

        task = asyncio.ensure_future(self.queue.submit(Job("image", P_IMAGE, run, after)))
        try:
            try:
                await asyncio.wait_for(started.wait(), self.settings.image_wait_seconds)
            except asyncio.TimeoutError:
                raise GpuBusy(f"the GPU was busy for more than {self.settings.image_wait_seconds} s") from None
            return await task
        except BaseException:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)  # leave the queue clean before reporting
            raise

    def _idle_long_enough(self) -> bool:
        return self.clock() - self._last_activity >= self.settings.idle_minutes * 60

    async def idle_check(self) -> None:
        """Stop the engine when nothing has used it for `idle_minutes`.

        The stop runs as an ordinary P_LLM job, not a preemptible P_TUNE one: a chat that arrives while the
        engine is shutting down must wait for the shutdown to finish, not cut it off half way."""
        quiet = self.queue.snapshot() == {"running": None, "pending": []}
        if self.engine.pid is None or not quiet or not self._idle_long_enough():
            return

        async def stop_if_still_idle():
            if self._idle_long_enough():  # a job may have run between the check and this job's turn
                await self.engine.stop()

        await self.queue.submit(Job("idle-stop", P_LLM, stop_if_still_idle))

    async def idle_loop(self, check_seconds: float = 30) -> None:
        while True:
            await asyncio.sleep(check_seconds)
            try:
                await self.idle_check()
            except Exception:  # ponytail: a failed check is retried at the next interval; add logging when the service logs
                pass
