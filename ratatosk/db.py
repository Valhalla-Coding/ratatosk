import sqlite3
from pathlib import Path

# Columns added after the first version; an older database gets them with ALTER TABLE.
_EXTRA = {"client": "TEXT", "user_agent": "TEXT", "path": "TEXT", "stream": "INTEGER", "max_tokens_asked": "INTEGER",
          "max_tokens": "INTEGER", "queue_seconds": "REAL", "tokens_per_second": "REAL", "error": "TEXT",
          "prompt_text": "TEXT", "reply_text": "TEXT"}
_TEXT = ("prompt_text", "reply_text")  # kept only for the newest requests
_SCHEMA = """CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    route TEXT, profile TEXT, model TEXT, ctx INTEGER,
    prompt_tokens INTEGER, gen_tokens INTEGER, seconds REAL, status TEXT)"""


class Db:
    def __init__(self, path: Path, keep_content: int = 50):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._keep = keep_content
        self._con = sqlite3.connect(path, check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._con.execute(_SCHEMA)
        have = {row["name"] for row in self._con.execute("PRAGMA table_info(requests)")}
        for name, kind in _EXTRA.items():
            if name not in have:
                self._con.execute(f"ALTER TABLE requests ADD COLUMN {name} {kind}")
        self._con.commit()
        self._light = "id, ts, route, profile, model, ctx, prompt_tokens, gen_tokens, seconds, status, " + ", ".join(
            n for n in _EXTRA if n not in _TEXT) + ", (prompt_text IS NOT NULL OR reply_text IS NOT NULL) AS has_content"

    def log_request(self, route: str, profile: str, model: str, ctx: int, prompt_tokens: int,
                    gen_tokens: int, seconds: float, status: str, **extra) -> int:
        unknown = set(extra) - set(_EXTRA)
        if unknown:
            raise TypeError(f"unknown request fields: {sorted(unknown)}")
        columns = ["route", "profile", "model", "ctx", "prompt_tokens", "gen_tokens", "seconds", "status", *extra]
        values = [route, profile, model, ctx, prompt_tokens, gen_tokens, seconds, status, *extra.values()]
        cur = self._con.execute(f"INSERT INTO requests ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})", values)
        self._con.execute("UPDATE requests SET prompt_text = NULL, reply_text = NULL WHERE id <= ?", (cur.lastrowid - self._keep,))
        self._con.commit()
        return cur.lastrowid

    def recent(self, limit: int = 20) -> list[dict]:
        """Newest first, without the (large) prompt and reply text; `has_content` says whether they are still kept."""
        return [dict(r) for r in self._con.execute(f"SELECT {self._light} FROM requests ORDER BY id DESC LIMIT ?", (limit,))]

    def get(self, request_id: int) -> dict | None:
        row = self._con.execute("SELECT * FROM requests WHERE id = ?", (request_id,)).fetchone()
        return dict(row) if row else None
