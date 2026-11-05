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


def test_extra_fields_are_kept_and_detail_has_the_content(tmp_path):
    db = Db(tmp_path / "r.db")
    rid = db.log_request("11435", "rp", "m", 8192, 100, 50, 2.5, "ok", client="192.168.1.5", user_agent="node",
                         path="/v1/completions", stream=1, max_tokens_asked=65536, max_tokens=1024,
                         queue_seconds=1.5, tokens_per_second=24.9, error=None, prompt_text="PROMPT", reply_text="REPLY")
    row = db.recent(1)[0]
    assert row["client"] == "192.168.1.5" and row["path"] == "/v1/completions" and row["tokens_per_second"] == 24.9
    assert "prompt_text" not in row and "reply_text" not in row  # the list stays light
    assert row["has_content"] == 1
    detail = db.get(rid)
    assert detail["prompt_text"] == "PROMPT" and detail["reply_text"] == "REPLY"
    assert (detail["max_tokens_asked"], detail["max_tokens"], detail["queue_seconds"]) == (65536, 1024, 1.5)
    assert db.get(9999) is None


def test_only_the_newest_requests_keep_their_text(tmp_path):
    db = Db(tmp_path / "r.db", keep_content=2)
    ids = [db.log_request("1", "plain", "m", 8192, 1, 1, 0.1, "ok", prompt_text=f"p{i}", reply_text=f"r{i}") for i in range(4)]
    assert [db.get(i)["prompt_text"] for i in ids] == [None, None, "p2", "p3"]
    assert [r["has_content"] for r in db.recent(4)] == [1, 1, 0, 0]  # newest first
    assert db.get(ids[0])["prompt_tokens"] == 1  # the numbers stay


def test_an_old_database_gets_the_new_columns(tmp_path):
    import sqlite3
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE requests (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        route TEXT, profile TEXT, model TEXT, ctx INTEGER, prompt_tokens INTEGER, gen_tokens INTEGER, seconds REAL, status TEXT)""")
    con.execute("INSERT INTO requests (route, profile, model, ctx, prompt_tokens, gen_tokens, seconds, status) VALUES ('1','plain','m',8192,1,1,0.1,'ok')")
    con.commit()
    con.close()
    db = Db(path)
    assert db.recent(1)[0]["has_content"] == 0 and db.recent(1)[0]["client"] is None  # the old row survives
    db.log_request("1", "plain", "m", 8192, 1, 1, 0.1, "ok", client="x", prompt_text="p")
    assert db.get(2)["client"] == "x"
