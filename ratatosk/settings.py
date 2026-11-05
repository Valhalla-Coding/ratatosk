import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class Route:
    port: int
    kind: str  # "llm" or "comfy"
    profile: str = "plain"  # "plain" or "rp"
    max_tokens: int | None = None  # longest reply on this door; None falls back to the old per-purpose caps
    allow: dict = field(default_factory=lambda: {"vision": True, "reasoning": True, "tools": True})


def default_priority(routes: list["Route"]) -> list[dict]:
    """Roleplay doors first, then other text doors, then image doors: the order Ratatosk always used."""
    rank = lambda r: 0 if r.kind == "llm" and r.profile == "rp" else 1 if r.kind == "llm" else 2
    return [{"door": r.port} for r in sorted(routes, key=rank)]


def _default_routes() -> list[Route]:
    return [Route(11434, "llm", "plain"), Route(11435, "llm", "rp"), Route(8188, "comfy", "plain")]


@dataclass
class Settings:
    routes: list[Route] = field(default_factory=_default_routes)
    ui_port: int = 5960
    engine_port: int = 8081
    comfy_url: str = "http://127.0.0.1:8189"
    engine_dir: str = field(default_factory=lambda: str(Path.home() / "ratatosk" / "engine"))
    threads: int = field(default_factory=lambda: max(1, (os.cpu_count() or 2) // 2))
    model_folders: list[dict] = field(
        default_factory=lambda: [{"path": str(Path.home() / "models"), "label": "Models"}]
    )
    default_model: str = ""
    aliases: dict = field(default_factory=dict)
    max_tokens_cap: int = 1024  # longest reply on a roleplay route
    max_tokens_cap_plain: int = 8192  # on other routes; a thinking model spends a thousand tokens or more before it answers
    idle_minutes: int = 30
    reserve_gb: int = 5
    image_wait_seconds: int = 600
    data_dir: str = "data"
    keep_content: int = 50  # how many of the newest requests keep their prompt and reply text
    mmproj_files: dict = field(default_factory=dict)  # model id -> vision file, for a model with none beside it
    image_max_tokens: int = 512  # most context one picture may take (a real camera frame measured about 500); larger ones are scaled down
    reply_reserve: int = 2048  # context kept free for a reply when sizing the engine; a 32k context halves the text speed here
    mmproj_on_gpu: bool = False  # on the 8 GB GPU the projector costs half the text speed; on the CPU a picture takes longer
    request_defaults: dict = field(default_factory=dict)  # model name or id -> request fields to add when the client left them out
    priority: list | None = None  # ordered lines, first goes first; None means default_priority(routes), [] means no lines

    def rules(self) -> list[dict]:
        return default_priority(self.routes) if self.priority is None else self.priority

    def max_tokens_for(self, route: Route) -> int:
        return route.max_tokens or (self.max_tokens_cap if route.profile == "rp" else self.max_tokens_cap_plain)


def load_settings(path: Path) -> Settings:
    path = Path(path)
    if not path.exists():
        return Settings()
    return settings_from_dict(json.loads(path.read_text(encoding="utf-8")))


def settings_from_dict(data: dict) -> Settings:
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
