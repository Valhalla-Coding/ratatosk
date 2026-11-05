import json

from ratatosk.settings import load_settings, save_settings


def test_defaults(tmp_path):
    s = load_settings(tmp_path / "missing.json")
    assert [(r.port, r.kind, r.profile) for r in s.routes] == [
        (11434, "llm", "plain"),
        (11435, "llm", "rp"),
        (8188, "comfy", "plain"),
    ]
    assert s.ui_port == 5960
    assert s.engine_port == 8081
    assert s.comfy_url == "http://127.0.0.1:8189"
    assert s.max_tokens_cap == 1024
    assert s.idle_minutes == 30
    assert s.reserve_gb == 5
    assert s.image_wait_seconds == 600


def test_roundtrip_ignores_unknown_keys(tmp_path):
    p = tmp_path / "s.json"
    s = load_settings(p)
    s.ui_port = 6000
    s.aliases = {"a": "b"}
    save_settings(s, p)
    data = json.loads(p.read_text())
    data["bogus"] = 1
    data["routes"][0]["bogus"] = 2
    p.write_text(json.dumps(data))
    assert load_settings(p) == s


def test_door_limit_falls_back_to_the_old_caps(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"max_tokens_cap": 512, "max_tokens_cap_plain": 4096, "routes": [
        {"port": 1, "kind": "llm", "profile": "rp"},
        {"port": 2, "kind": "llm", "profile": "plain"},
        {"port": 3, "kind": "llm", "profile": "plain", "max_tokens": 100}]}))
    s = load_settings(p)
    assert [s.max_tokens_for(r) for r in s.routes] == [512, 4096, 100]


def test_door_features_default_to_allowed():
    from ratatosk.settings import Route
    assert Route(1, "llm").allow == {"vision": True, "reasoning": True, "tools": True}


def test_default_priority_matches_the_old_order(tmp_path):
    from ratatosk.settings import default_priority
    s = load_settings(tmp_path / "missing.json")
    assert s.priority is None
    assert s.rules() == default_priority(s.routes) == [{"door": 11435}, {"door": 11434}, {"door": 8188}]
    s.priority = []  # an empty list means "no lines", not "use the default"
    assert s.rules() == []
