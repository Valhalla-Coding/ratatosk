import asyncio

import aiohttp

HELD_THRESHOLD = 256 * 2**20  # below this much held by ComfyUI itself there is nothing worth waiting for


class Comfy:
    """The two ComfyUI calls the scheduler needs."""

    def __init__(self, base_url: str, session: aiohttp.ClientSession, settle_poll: float = 0.25, settle_cap: float = 20.0):
        self.base_url = base_url.rstrip("/")
        self._session = session
        self._settle_poll, self._settle_cap = settle_poll, settle_cap

    async def interrupt(self) -> None:
        """Stop the image being made and forget what is waiting behind it ("drop everything")."""
        async with self._session.post(f"{self.base_url}/interrupt") as resp:
            resp.raise_for_status()
        async with self._session.post(f"{self.base_url}/queue", json={"clear": True}) as resp:
            resp.raise_for_status()

    async def pending(self) -> int:
        """Jobs ComfyUI is running or has queued."""
        async with self._session.get(f"{self.base_url}/queue") as resp:
            data = await resp.json()
        return len(data.get("queue_running", [])) + len(data.get("queue_pending", []))

    async def _held(self) -> int | None:
        """Bytes of GPU memory ComfyUI's own torch is holding, or None if ComfyUI does not say."""
        try:
            async with self._session.get(f"{self.base_url}/system_stats", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                devices = (await resp.json()).get("devices") or []
            held = devices[0].get("torch_vram_total") if devices else None
            return held if isinstance(held, int) else None
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, AttributeError):
            return None

    async def free(self) -> None:
        """Ask ComfyUI to drop its models, then wait until the memory it was holding is actually gone.

        /free only sets a flag; ComfyUI's worker thread releases the memory later, sometimes more than a second
        later. An engine started in between is sized by --fit to memory that is still taken, and ends up mostly
        on the CPU (measured on Idun: 3.7 tokens/s instead of 25)."""
        held = await self._held()
        async with self._session.post(f"{self.base_url}/free", json={"unload_models": True, "free_memory": True}) as resp:
            resp.raise_for_status()
        if held is None or held < HELD_THRESHOLD:  # nothing of ComfyUI's to wait for
            return
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._settle_cap
        while loop.time() < deadline:
            await asyncio.sleep(self._settle_poll)
            now = await self._held()
            if now is None or now < HELD_THRESHOLD:
                await asyncio.sleep(self._settle_poll)  # let the driver hand the pages back
                return
