import asyncio

import aiohttp


class Comfy:
    """The two ComfyUI calls the scheduler needs."""

    def __init__(self, base_url: str, session: aiohttp.ClientSession, settle_poll: float = 0.25,
                 settle_quiet: float = 1.0, settle_cap: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self._session = session
        self._settle_poll, self._settle_quiet, self._settle_cap = settle_poll, settle_quiet, settle_cap

    async def pending(self) -> int:
        """Jobs ComfyUI is running or has queued."""
        async with self._session.get(f"{self.base_url}/queue") as resp:
            data = await resp.json()
        return len(data.get("queue_running", [])) + len(data.get("queue_pending", []))

    async def _vram_free(self) -> int | None:
        try:
            async with self._session.get(f"{self.base_url}/system_stats", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                devices = (await resp.json()).get("devices") or []
            return devices[0].get("vram_free") if devices else None
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, AttributeError):
            return None

    async def free(self) -> None:
        """Ask ComfyUI to drop its models, then wait until its free memory has stopped rising.

        /free only sets a flag; ComfyUI's worker thread releases the memory a moment later, and a model
        started in between would be sized to memory that is still taken."""
        async with self._session.post(f"{self.base_url}/free", json={"unload_models": True, "free_memory": True}) as resp:
            resp.raise_for_status()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._settle_cap
        last = await self._vram_free()
        if last is None:  # ComfyUI does not report memory; nothing to wait on
            return
        last_change = loop.time()
        while loop.time() < deadline and loop.time() - last_change < self._settle_quiet:
            await asyncio.sleep(self._settle_poll)
            now = await self._vram_free()
            if now is None:
                return
            if now != last:
                last, last_change = now, loop.time()
