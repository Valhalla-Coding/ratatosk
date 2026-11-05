import asyncio
import os
import time
from pathlib import Path
from typing import Callable

import aiohttp

from ratatosk.settings import Settings


class EngineError(Exception):
    pass


class Engine:
    """Starts, stops and health-checks one llama-server child process."""

    def __init__(self, settings: Settings, argv_override: Callable[[Path, int], list[str]] | None = None,
                 poll_seconds: float = 0.5, health_timeout: float = 300.0):
        self._settings = settings
        self._argv_override = argv_override
        self._poll = poll_seconds
        self._health_timeout = health_timeout
        self._proc: asyncio.subprocess.Process | None = None
        self._mmproj: Path | None = None
        self.loaded: tuple[Path, int] | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._settings.engine_port}"

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc and self._proc.returncode is None else None

    def argv(self, model: Path, ctx: int, mmproj: Path | None = None) -> list[str]:
        s = self._settings
        argv = [f"{s.engine_dir}/backend/llama-server", "--model", str(model), "--ctx-size", str(ctx),
                "--host", "127.0.0.1", "--port", str(s.engine_port), "--fit", "on", "--flash-attn", "on",
                "--parallel", "1", "--threads", str(s.threads)]
        if mmproj:  # measured on Idun: about 0.25 GB of GPU memory, a picture read in about 3 s
            argv += ["--mmproj", str(mmproj), "--image-max-tokens", str(s.image_max_tokens)]
        return argv

    def _env(self) -> dict:
        libs = f"{self._settings.engine_dir}/backend:{self._settings.engine_dir}/vendor"
        old = os.environ.get("LD_LIBRARY_PATH")
        return {**os.environ, "LD_LIBRARY_PATH": f"{libs}:{old}" if old else libs}

    async def ensure(self, model: Path, ctx: int, mmproj: Path | None = None) -> None:
        """Make sure llama-server is running this model (with this vision file) at this context, (re)starting it if not."""
        model = Path(model)
        mmproj = Path(mmproj) if mmproj else None
        if self.loaded == (model, ctx) and self._mmproj == mmproj and self.pid is not None:
            return
        await self.stop()
        log_path = Path(self._settings.data_dir) / "engine.log"  # kept so a failed start can be diagnosed
        log_path.parent.mkdir(parents=True, exist_ok=True)
        argv = self._argv_override(model, ctx) if self._argv_override else self.argv(model, ctx, mmproj)
        try:
            with open(log_path, "ab") as log:
                self._proc = await asyncio.create_subprocess_exec(*argv, env=self._env(), stdout=log, stderr=log)
        except OSError as exc:
            raise EngineError(f"cannot start the engine: {exc}") from exc
        try:
            await self._wait_healthy(self._proc)
        except BaseException:
            await self.stop()
            raise
        self.loaded, self._mmproj = (model, ctx), mmproj

    async def _wait_healthy(self, proc: asyncio.subprocess.Process) -> None:
        deadline = time.monotonic() + self._health_timeout
        async with aiohttp.ClientSession() as session:
            while time.monotonic() < deadline:
                if proc.returncode is not None:
                    raise EngineError(f"the engine exited with code {proc.returncode} while starting")
                try:
                    async with session.get(f"{self.url}/health", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                        if resp.status == 200:
                            return
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    pass
                await asyncio.sleep(self._poll)
        raise EngineError(f"the engine was not healthy after {self._health_timeout:.0f} s")

    async def stop(self) -> None:
        proc, self._proc, self.loaded, self._mmproj = self._proc, None, None, None
        if proc is None or proc.returncode is not None:
            return
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), 10)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()

    async def count_tokens(self, text: str) -> int | None:
        """Exact prompt length from the running engine, or None if it is not running."""
        if self.pid is None:
            return None
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.url}/tokenize", json={"content": text},
                                        timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    return len((await resp.json())["tokens"])
        except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, ValueError):
            return None
