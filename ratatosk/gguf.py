import os
import struct
from pathlib import Path

_FIXED = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9
MAX_STRING = 64 * 2**20  # a chat template is a few KB; anything this large is a corrupt file
MAX_DEPTH = 4  # arrays inside arrays; real files use one level


class _Bad(Exception):
    pass


def _take(f, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise _Bad
    return data


def _string(f) -> bytes:
    (n,) = struct.unpack("<Q", _take(f, 8))
    if n > MAX_STRING:
        raise _Bad
    return _take(f, n)


def _skip(f, kind: int, end: int, depth: int = 0) -> None:
    """Step over one value. `end` is the file size: a length that reaches past it is a corrupt file, never a big read."""
    if depth > MAX_DEPTH:
        raise _Bad
    if kind in _FIXED:
        _take(f, struct.calcsize(_FIXED[kind]))
    elif kind == _STRING:
        _string(f)
    elif kind == _ARRAY:
        (inner,) = struct.unpack("<I", _take(f, 4))
        (count,) = struct.unpack("<Q", _take(f, 8))
        if inner in _FIXED:
            size = struct.calcsize(_FIXED[inner]) * count
            if f.tell() + size > end:
                raise _Bad
            f.seek(size, 1)
        else:
            if count > end:  # every element takes at least a byte
                raise _Bad
            for _ in range(count):
                _skip(f, inner, end, depth + 1)
    else:
        raise _Bad


def read_chat_template(path: Path) -> str | None:
    """The chat template stored in a GGUF file's header, or None when there is none or the file is unreadable.

    Only the key/value block at the start is read; the tensors are never touched."""
    try:
        with open(path, "rb") as f:
            end = os.fstat(f.fileno()).st_size
            if _take(f, 4) != b"GGUF":
                return None
            (version,) = struct.unpack("<I", _take(f, 4))
            if version < 2:
                return None
            _take(f, 8)  # tensor count
            (pairs,) = struct.unpack("<Q", _take(f, 8))
            for _ in range(pairs):
                key = _string(f)
                (kind,) = struct.unpack("<I", _take(f, 4))
                if key == b"tokenizer.chat_template" and kind == _STRING:
                    return _string(f).decode("utf-8", "replace")
                _skip(f, kind, end)
    except (_Bad, OSError, struct.error, OverflowError, MemoryError, RecursionError):
        return None
    return None


def capabilities_from_template(template: str | None, has_vision: bool) -> dict:
    t = template or ""
    return {"vision": has_vision, "tools": "tools" in t, "reasoning": "enable_thinking" in t or "<think>" in t}
