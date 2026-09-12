"""
Local session storage — SQLite only. Nothing in this module ever makes a
network call; all progress data stays on the user's device.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

DEFAULT_DB_PATH = Path.home() / ".vaani" / "sessions.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    exercise_name TEXT NOT NULL,
    duration_s REAL NOT NULL,
    fluency_score INTEGER NOT NULL,
    disfluency_count INTEGER NOT NULL,
    pacing_label TEXT NOT NULL
);
"""


@dataclass
class SessionRecord:
    id: int
    created_at: str
    exercise_name: str
    duration_s: float
    fluency_score: int
    disfluency_count: int
    pacing_label: str


class SessionStore:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(SCHEMA)
        self._conn.commit()

    def save_session(
        self,
        exercise_name: str,
        duration_s: float,
        fluency_score: int,
        disfluency_count: int,
        pacing_label: str,
    ) -> int:
        cur = self._conn.execute(
            """INSERT INTO sessions
               (created_at, exercise_name, duration_s, fluency_score, disfluency_count, pacing_label)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                datetime.now().isoformat(timespec="seconds"),
                exercise_name,
                duration_s,
                fluency_score,
                disfluency_count,
                pacing_label,
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def recent_sessions(self, limit: int = 10) -> list[SessionRecord]:
        rows = self._conn.execute(
            "SELECT id, created_at, exercise_name, duration_s, fluency_score, disfluency_count, pacing_label "
            "FROM sessions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [SessionRecord(*row) for row in rows]

    def close(self) -> None:
        self._conn.close()
