import sqlite3
from pathlib import Path

_SCHEMA = """CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    route TEXT, profile TEXT, model TEXT, ctx INTEGER,
    prompt_tokens INTEGER, gen_tokens INTEGER, seconds REAL, status TEXT)"""


class Db:
    def __init__(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(path, check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._con.execute(_SCHEMA)
        self._con.commit()

    def log_request(self, route: str, profile: str, model: str, ctx: int, prompt_tokens: int,
                    gen_tokens: int, seconds: float, status: str) -> int:
        cur = self._con.execute(
            "INSERT INTO requests (route, profile, model, ctx, prompt_tokens, gen_tokens, seconds, status)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (route, profile, model, ctx, prompt_tokens, gen_tokens, seconds, status))
        self._con.commit()
        return cur.lastrowid

    def recent(self, limit: int = 20) -> list[dict]:
        rows = self._con.execute("SELECT * FROM requests ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]
