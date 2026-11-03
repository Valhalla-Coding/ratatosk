import socket

import aiohttp

from ratatosk.app import start
from ratatosk.settings import Route, Settings, save_settings


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def can_connect(port) -> bool:
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


async def test_start_binds_all_ports_and_stop_shuts_everything_down(tmp_path):
    ports = [free_port() for _ in range(4)]
    settings = Settings(
        routes=[Route(ports[0], "llm", "plain"), Route(ports[1], "llm", "rp"), Route(ports[2], "comfy", "plain")],
        ui_port=ports[3], data_dir=str(tmp_path), model_folders=[{"path": str(tmp_path / "none"), "label": "x"}])
    save_settings(settings, tmp_path / "settings.json")
    service = await start(tmp_path / "settings.json")
    stopped = []
    real_stop = service.engine.stop

    async def spy():
        stopped.append(True)
        await real_stop()

    service.engine.stop = spy
    try:
        assert all(can_connect(p) for p in ports)
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://127.0.0.1:{ports[0]}/v1/models") as resp:
                assert resp.status == 200
            async with session.get(f"http://127.0.0.1:{ports[3]}/api/status") as resp:
                assert resp.status == 200
    finally:
        await service.stop()
    assert stopped == [True]
    assert not any(can_connect(p) for p in ports)
