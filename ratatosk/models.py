from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Model:
    id: str
    path: Path
    size: int
    folder: Path
    mmproj: Path | None = None  # the vision file that lets the model read pictures


def _vision_file(model_path: Path, override: str | None) -> Path | None:
    """The vision file set for this model, else the one lying next to it (as model downloaders put it)."""
    if override:
        return Path(override) if Path(override).is_file() else None
    siblings = sorted(model_path.parent.glob("mmproj*.gguf"))
    return siblings[0] if siblings else None


def scan(folders: list[Path], mmproj_files: dict | None = None) -> list[Model]:
    """Every .gguf under the folders (vision files are not models). A missing folder gives nothing.

    `mmproj_files` maps a model id to a vision file, for a model that has none beside it."""
    found = []
    for folder in map(Path, folders):
        if not folder.is_dir():
            continue
        for p in sorted(folder.rglob("*.gguf")):
            if "mmproj" in p.name.lower():
                continue
            found.append(Model(p.stem.lower(), p, p.stat().st_size, folder, _vision_file(p, (mmproj_files or {}).get(p.stem.lower()))))
    return found


def _find(key: str, models: list[Model]) -> Model | None:
    k = key.lower()
    return next((m for m in models if k in (m.id, str(m.path).lower())), None)


def resolve(name: str, models: list[Model], aliases: dict, default: str) -> Model | None:
    """Alias first, then the model id or path, then the default model, else None."""
    if name in aliases:
        return _find(aliases[name], models)
    return _find(name, models) or (_find(default, models) if default else None)
