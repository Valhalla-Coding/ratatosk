import struct

from ratatosk.gguf import capabilities_from_template, read_chat_template


def s(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def gguf(pairs: list[tuple[str, bytes]], version: int = 3) -> bytes:
    out = b"GGUF" + struct.pack("<I", version) + struct.pack("<Q", 0) + struct.pack("<Q", len(pairs))
    for key, value in pairs:
        out += s(key) + value
    return out + b"tensor-data-that-must-never-be-read"


STRING = lambda text: struct.pack("<I", 8) + s(text)
UINT32 = lambda n: struct.pack("<I", 4) + struct.pack("<I", n)
STR_ARRAY = lambda items: struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", len(items)) + b"".join(s(i) for i in items)
INT_ARRAY = lambda n: struct.pack("<I", 9) + struct.pack("<I", 5) + struct.pack("<Q", n) + b"\0\0\0\0" * n


def write(tmp_path, data, name="m.gguf"):
    p = tmp_path / name
    p.write_bytes(data)
    return p


def test_finds_the_template_after_other_keys_and_big_arrays(tmp_path):
    p = write(tmp_path, gguf([("general.name", STRING("x")), ("tokenizer.ggml.tokens", STR_ARRAY(["a", "bb", "ccc"])),
                              ("tokenizer.ggml.token_type", INT_ARRAY(3)), ("general.alignment", UINT32(32)),
                              ("tokenizer.chat_template", STRING("{% if tools %}<think>{% endif %}"))]))
    assert read_chat_template(p) == "{% if tools %}<think>{% endif %}"


def test_no_template_truncated_wrong_magic_and_missing_file_all_give_none(tmp_path):
    assert read_chat_template(write(tmp_path, gguf([("general.name", STRING("x"))]))) is None
    full = gguf([("tokenizer.chat_template", STRING("hello"))])
    assert read_chat_template(write(tmp_path, full[:30], "cut.gguf")) is None
    assert read_chat_template(write(tmp_path, b"NOPE" + full[4:], "magic.gguf")) is None
    assert read_chat_template(write(tmp_path, b"", "empty.gguf")) is None
    assert read_chat_template(tmp_path / "missing.gguf") is None


def test_capabilities_come_from_the_template_and_the_vision_file():
    full = "{{ tools }} {{ enable_thinking }}"
    assert capabilities_from_template(full, True) == {"vision": True, "tools": True, "reasoning": True}
    assert capabilities_from_template("{{ messages }}", False) == {"vision": False, "tools": False, "reasoning": False}
    assert capabilities_from_template(None, False) == {"vision": False, "tools": False, "reasoning": False}
    assert capabilities_from_template("<think>", False)["reasoning"] is True
