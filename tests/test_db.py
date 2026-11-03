from ratatosk.db import Db


def test_log_and_recent(tmp_path):
    db = Db(tmp_path / "r.db")
    a = db.log_request("11434", "plain", "m", 8192, 100, 50, 1.5, "ok")
    b = db.log_request("11435", "rp", "m2", 16384, 200, 60, 2.5, "error")
    assert b > a
    rows = db.recent(1)
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == b
    assert (row["route"], row["profile"], row["model"]) == ("11435", "rp", "m2")
    assert (row["ctx"], row["prompt_tokens"], row["gen_tokens"]) == (16384, 200, 60)
    assert (row["seconds"], row["status"]) == (2.5, "error")
    assert [r["id"] for r in db.recent(5)] == [b, a]
