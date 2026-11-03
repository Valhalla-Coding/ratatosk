import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class Route:
    port: int
    kind: str  # "llm" or "comfy"
    profile: str = "plain"  # "plain" or "rp"


def _default_routes() -> list[Route]:
    return [Route(11434, "llm", "plain"), Route(11435, "llm", "rp"), Route(8188, "comfy", "plain")]


@dataclass
class Settings:
    routes: list[Route] = field(default_factory=_default_routes)
    ui_port: int = 5960
    engine_port: int = 8081
    comfy_url: str = "http://127.0.0.1:8189"
    engine_dir: str = "/home/master/ratatosk/engine"
    threads: int = field(default_factory=lambda: max(1, (os.cpu_count() or 2) // 2))
    model_folders: list[dict] = field(
        default_factory=lambda: [{"path": "/home/master/.lmstudio/models", "label": "SSD models"}]
    )
    default_model: str = ""
    aliases: dict = field(default_factory=dict)
    max_tokens_cap: int = 1024
    idle_minutes: int = 30
    reserve_gb: int = 5
    image_wait_seconds: int = 600
    data_dir: str = "data"


def load_settings(path: Path) -> Settings:
    path = Path(path)
    if not path.exists():
        return Settings()
    data = json.loads(path.read_text(encoding="utf-8"))
    known = {f.name for f in fields(Settings)}
    kwargs = {k: v for k, v in data.items() if k in known and k != "routes"}
    if "routes" in data:
        route_keys = {f.name for f in fields(Route)}
        kwargs["routes"] = [Route(**{k: v for k, v in r.items() if k in route_keys}) for r in data["routes"]]
    return Settings(**kwargs)


def save_settings(settings: Settings, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
