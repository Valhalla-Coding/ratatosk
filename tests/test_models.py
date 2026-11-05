from ratatosk.models import resolve, scan


def make(root, rel, size=10):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * size)
    return p


def test_scan_skips_mmproj_and_non_gguf(tmp_path):
    a = make(tmp_path, "pub/repo/Qwen-9B.Q4_K_M.gguf", 100)
    make(tmp_path, "pub/repo/mmproj-model.gguf")
    make(tmp_path, "pub/repo/notes.txt")
    models = scan([tmp_path])
    assert [(m.id, m.path, m.size, m.folder) for m in models] == [("qwen-9b.q4_k_m", a, 100, tmp_path)]


def test_scan_missing_folder_returns_empty(tmp_path):
    assert scan([tmp_path / "nope"]) == []


def test_resolve_order(tmp_path):
    a = make(tmp_path, "x/Alpha.gguf")
    b = make(tmp_path, "x/Beta.gguf")
    models = scan([tmp_path])
    aliases = {"qwen3.5-9b-obliterated": str(b)}
    assert resolve("qwen3.5-9b-obliterated", models, aliases, "alpha").path == b  # alias beats default
    assert resolve("ALPHA", models, {}, "").path == a  # case-insensitive id match
    assert resolve("unknown", models, {}, "beta").path == b  # falls back to default
    assert resolve("unknown", models, {}, "") is None


def test_scan_pairs_the_vision_file_from_the_same_folder(tmp_path):
    make(tmp_path, "pub/repo/Qwen-9B.Q4_K_M.gguf")
    projector = make(tmp_path, "pub/repo/mmproj-Qwen-9B-BF16.gguf")
    make(tmp_path, "pub/other/Plain.gguf")
    by_id = {m.id: m for m in scan([tmp_path])}
    assert by_id["qwen-9b.q4_k_m"].mmproj == projector
    assert by_id["plain"].mmproj is None  # a model with no vision file next to it can only read text
    assert "mmproj-qwen-9b-bf16" not in by_id  # the vision file itself is still not a model


def test_a_vision_file_can_be_set_for_a_model_that_has_none(tmp_path):
    make(tmp_path, "a/Obliterated.gguf")
    projector = make(tmp_path, "b/mmproj-base.gguf")  # another model's vision file, from a different folder
    models = scan([tmp_path], {"obliterated": str(projector), "gone": str(tmp_path / "missing.gguf")})
    assert models[0].mmproj == projector
    assert scan([tmp_path], {"obliterated": str(tmp_path / "missing.gguf")})[0].mmproj is None  # a missing file is ignored
