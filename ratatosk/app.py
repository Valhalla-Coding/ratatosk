import asyncio
import signal
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
from aiohttp import web

from ratatosk.comfy import Comfy
from ratatosk.db import Db
from ratatosk.engine import Engine
from ratatosk.gateway_comfy import make_comfy_app
from ratatosk.gateway_llm import AppContext, make_llm_app
from ratatosk.gpuqueue import GpuQueue
from ratatosk.models import scan
from ratatosk.scheduler import Scheduler
from ratatosk.settings import load_settings
from ratatosk.web import make_ui_app


@dataclass
class Service:
    engine: Engine
    queue: GpuQueue
    session: aiohttp.ClientSession
    idle_task: asyncio.Task
    runners: list = field(default_factory=list)

    async def stop(self) -> None:
        self.idle_task.cancel()
        await asyncio.gather(self.idle_task, return_exceptions=True)
        for runner in self.runners:
            await runner.cleanup()
        await self.engine.stop()
        await self.queue.stop()
        await self.session.close()


async def start(settings_path: Path) -> Service:
    """Build everything and start listening on every route and the UI port."""
    settings_path = Path(settings_path)
    settings = load_settings(settings_path)
    session = aiohttp.ClientSession()
    queue, engine = GpuQueue(), Engine(settings)
    comfy = Comfy(settings.comfy_url, session)
    scheduler = Scheduler(settings, queue, engine, comfy)
    ctx = AppContext(settings, scheduler, engine, Db(Path(settings.data_dir) / "ratatosk.db", settings.keep_content),
                     lambda: scan([Path(f["path"]) for f in settings.model_folders], settings.mmproj_files))
    queue.start()
    service = Service(engine, queue, session, asyncio.get_running_loop().create_task(scheduler.idle_loop()))
    apps = [(r.port, make_llm_app(r, ctx) if r.kind == "llm" else make_comfy_app(r, ctx)) for r in settings.routes]
    apps.append((settings.ui_port, make_ui_app(ctx, comfy, settings_path)))
    try:
        for port, app in apps:
            runner = web.AppRunner(app, handler_cancellation=True)  # a client that disconnects cancels its job
            await runner.setup()
            service.runners.append(runner)
            await web.TCPSite(runner, "0.0.0.0", port).start()
    except BaseException:
        await service.stop()
        raise
    return service


async def run(settings_path: Path) -> None:
    service = await start(settings_path)
    stop = asyncio.Event()
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_running_loop().add_signal_handler(sig, stop.set)
    except NotImplementedError:  # Windows: Ctrl+C raises KeyboardInterrupt instead
        pass
    try:
        await stop.wait()
    finally:
        await service.stop()
