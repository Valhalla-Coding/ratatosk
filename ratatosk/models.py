from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Model:
    id: str
    path: Path
    size: int
    folder: Path


def scan(folders: list[Path]) -> list[Model]:
    """Every .gguf under the folders (vision projector files excluded). A missing folder gives nothing."""
    found = []
    for folder in map(Path, folders):
        if not folder.is_dir():
            continue
        for p in sorted(folder.rglob("*.gguf")):
            if "mmproj" in p.name.lower():
                continue
            found.append(Model(p.stem.lower(), p, p.stat().st_size, folder))
    return found


def _find(key: str, models: list[Model]) -> Model | None:
    k = key.lower()
    return next((m for m in models if k in (m.id, str(m.path).lower())), None)


def resolve(name: str, models: list[Model], aliases: dict, default: str) -> Model | None:
    """Alias first, then the model id or path, then the default model, else None."""
    if name in aliases:
        return _find(aliases[name], models)
    return _find(name, models) or (_find(default, models) if default else None)
