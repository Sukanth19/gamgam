"""SQLite storage for game metadata.

Lists (genres, tags, platforms) are stored as JSON text: we never filter on
them in SQL, so separate join tables would only add complexity.
"""

import json
import sqlite3
from pathlib import Path

from app import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    id            INTEGER PRIMARY KEY,   -- RAWG id
    slug          TEXT NOT NULL UNIQUE,
    name          TEXT NOT NULL,
    released      TEXT,
    rating        REAL,
    ratings_count INTEGER,
    playtime      INTEGER,
    added         INTEGER,               -- RAWG popularity signal
    genres        TEXT NOT NULL DEFAULT '[]',
    tags          TEXT NOT NULL DEFAULT '[]',
    platforms     TEXT NOT NULL DEFAULT '[]',
    description   TEXT,
    updated_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_games_name ON games(name COLLATE NOCASE);

-- What the user has played. One row per game: re-rating overwrites.
CREATE TABLE IF NOT EXISTS play_history (
    game_id    INTEGER PRIMARY KEY REFERENCES games(id),
    verdict    TEXT NOT NULL CHECK (verdict IN ('liked', 'disliked')),
    note       TEXT,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

_JSON_FIELDS = ("genres", "tags", "platforms")
_COLUMNS = ("id", "slug", "name", "released", "rating", "ratings_count",
            "playtime", "added", *_JSON_FIELDS, "description")


def connect(path: Path | str = config.DB_PATH) -> sqlite3.Connection:
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def upsert_games(conn: sqlite3.Connection, games: list[dict]) -> int:
    """Insert or update games by RAWG id. Returns the number of rows written.

    COALESCE keeps an existing description when the new record has none
    (list results from RAWG don't include descriptions).
    """
    cols = ", ".join(_COLUMNS)
    placeholders = ", ".join(f":{c}" for c in _COLUMNS)
    updates = ", ".join(
        f"{c} = COALESCE(excluded.{c}, games.{c})" if c == "description"
        else f"{c} = excluded.{c}"
        for c in _COLUMNS if c != "id"
    )
    sql = (f"INSERT INTO games ({cols}) VALUES ({placeholders}) "
           f"ON CONFLICT(id) DO UPDATE SET {updates}, updated_at = datetime('now')")
    rows = []
    for g in games:
        row = {c: g.get(c) for c in _COLUMNS}
        for f in _JSON_FIELDS:
            row[f] = json.dumps(g.get(f) or [])
        rows.append(row)
    with conn:
        conn.executemany(sql, rows)
    return len(rows)


def _to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    for f in _JSON_FIELDS:
        d[f] = json.loads(d[f])
    return d


def get_all_games(conn: sqlite3.Connection) -> list[dict]:
    return [_to_dict(r) for r in conn.execute("SELECT * FROM games ORDER BY added DESC")]


def get_game(conn: sqlite3.Connection, game_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM games WHERE id = ?", (game_id,)).fetchone()
    return _to_dict(row) if row else None


def ids_missing_description(conn: sqlite3.Connection) -> list[int]:
    return [r[0] for r in conn.execute(
        "SELECT id FROM games WHERE description IS NULL OR description = '' ORDER BY added DESC")]


def find_game_by_name(conn: sqlite3.Connection, name: str) -> dict | None:
    """Exact (case-insensitive) match first, then a substring match,
    preferring the most popular game when several match."""
    row = conn.execute("SELECT * FROM games WHERE name = ? COLLATE NOCASE "
                       "ORDER BY added DESC LIMIT 1", (name,)).fetchone()
    if row is None:
        row = conn.execute("SELECT * FROM games WHERE name LIKE ? "
                           "ORDER BY added DESC LIMIT 1", (f"%{name}%",)).fetchone()
    return _to_dict(row) if row else None


def count_games(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]


# --- play history -------------------------------------------------------------

def set_verdict(conn: sqlite3.Connection, game_id: int, verdict: str, note: str | None = None):
    if verdict not in ("liked", "disliked"):
        raise ValueError(f"verdict must be 'liked' or 'disliked', got {verdict!r}")
    with conn:
        conn.execute(
            "INSERT INTO play_history (game_id, verdict, note) VALUES (?, ?, ?) "
            "ON CONFLICT(game_id) DO UPDATE SET verdict = excluded.verdict, "
            "note = COALESCE(excluded.note, play_history.note), updated_at = datetime('now')",
            (game_id, verdict, note))


def remove_verdict(conn: sqlite3.Connection, game_id: int) -> bool:
    with conn:
        return conn.execute("DELETE FROM play_history WHERE game_id = ?", (game_id,)).rowcount > 0


def get_history(conn: sqlite3.Connection) -> list[dict]:
    """[{game_id, name, verdict, note, updated_at}], liked first."""
    rows = conn.execute(
        "SELECT h.game_id, g.name, h.verdict, h.note, h.updated_at FROM play_history h "
        "JOIN games g ON g.id = h.game_id ORDER BY h.verdict DESC, h.updated_at")
    return [dict(r) for r in rows]
