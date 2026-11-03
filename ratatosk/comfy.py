import aiohttp


class Comfy:
    """The two ComfyUI calls the scheduler needs."""

    def __init__(self, base_url: str, session: aiohttp.ClientSession):
        self.base_url = base_url.rstrip("/")
        self._session = session

    async def pending(self) -> int:
        """Jobs ComfyUI is running or has queued."""
        async with self._session.get(f"{self.base_url}/queue") as resp:
            data = await resp.json()
        return len(data.get("queue_running", [])) + len(data.get("queue_pending", []))

    async def free(self) -> None:
        """Ask ComfyUI to drop its models and cached memory."""
        async with self._session.post(f"{self.base_url}/free", json={"unload_models": True, "free_memory": True}) as resp:
            resp.raise_for_status()
