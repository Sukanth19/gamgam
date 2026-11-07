"""SQLite storage for game metadata.

Lists (genres, tags, platforms) are stored as JSON text: we never filter on
them in SQL, so separate join tables would only add complexity.
"""

import json
import re
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

-- Wiki pages we've fetched (phase 3). page_key = "<wiki host>:<pageid>".
CREATE TABLE IF NOT EXISTS wiki_pages (
    page_key     TEXT PRIMARY KEY,
    game_id      INTEGER NOT NULL REFERENCES games(id),
    wiki         TEXT NOT NULL,
    title        TEXT NOT NULL,
    url          TEXT,
    revid        INTEGER NOT NULL,          -- unchanged revid => skip on re-fetch
    categories   TEXT NOT NULL DEFAULT '[]',
    page_spoiler INTEGER NOT NULL DEFAULT 0,
    wikitext     TEXT,                      -- raw source, so we can re-chunk offline
    fetched_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
-- Chunks of those pages; replaced wholesale when a page's revision changes.
CREATE TABLE IF NOT EXISTS lore_chunks (
    chunk_id  TEXT PRIMARY KEY,              -- "<page_key>#<n>"
    page_key  TEXT NOT NULL REFERENCES wiki_pages(page_key) ON DELETE CASCADE,
    game_id   INTEGER NOT NULL,
    position  INTEGER NOT NULL,
    section   TEXT NOT NULL,
    kind      TEXT NOT NULL,
    text      TEXT NOT NULL,
    spoiler   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_chunks_game ON lore_chunks(game_id);

-- Full-text index over chunks for BM25 keyword search (phase 4). FTS5 ships
-- with SQLite. "porter" stems words (fighting -> fight); "remove_diacritics"
-- lets "Pokemon" match "Pokémon".
CREATE VIRTUAL TABLE IF NOT EXISTS lore_fts USING fts5(
    chunk_id UNINDEXED, game_id UNINDEXED, title, section, text,
    tokenize = 'porter unicode61 remove_diacritics 2'
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
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection):
    """Tiny migrations for databases created by older versions of the schema."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(wiki_pages)")}
    if "wikitext" not in cols:
        conn.execute("ALTER TABLE wiki_pages ADD COLUMN wikitext TEXT")
    # Chunks created before the FTS index existed: index them once.
    n_fts = conn.execute("SELECT COUNT(*) FROM lore_fts").fetchone()[0]
    if n_fts == 0 and conn.execute("SELECT 1 FROM lore_chunks LIMIT 1").fetchone():
        rebuild_fts(conn)


def rebuild_fts(conn: sqlite3.Connection):
    with conn:
        conn.execute("DELETE FROM lore_fts")
        conn.execute(
            "INSERT INTO lore_fts (chunk_id, game_id, title, section, text) "
            "SELECT c.chunk_id, c.game_id, p.title, c.section, c.text "
            "FROM lore_chunks c JOIN wiki_pages p ON p.page_key = c.page_key")


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


# --- wiki lore ----------------------------------------------------------------

def page_revids(conn: sqlite3.Connection, wiki: str) -> dict[str, int]:
    return {r[0]: r[1] for r in conn.execute(
        "SELECT page_key, revid FROM wiki_pages WHERE wiki = ?", (wiki,))}


def save_page(conn: sqlite3.Connection, page: dict, chunks: list[dict]):
    """Upsert a page and REPLACE its chunks (in one transaction)."""
    with conn:
        conn.execute(
            "INSERT INTO wiki_pages (page_key, game_id, wiki, title, url, revid, categories, "
            "page_spoiler, wikitext) VALUES (:page_key, :game_id, :wiki, :title, :url, :revid, "
            ":categories, :page_spoiler, :wikitext) "
            "ON CONFLICT(page_key) DO UPDATE SET title=excluded.title, url=excluded.url, "
            "revid=excluded.revid, categories=excluded.categories, page_spoiler=excluded.page_spoiler, "
            "wikitext=COALESCE(excluded.wikitext, wiki_pages.wikitext), fetched_at=datetime('now')",
            {"wikitext": None, **page, "categories": json.dumps(page.get("categories") or []),
             "page_spoiler": int(page.get("page_spoiler", False))})
        conn.execute("DELETE FROM lore_fts WHERE chunk_id IN "
                     "(SELECT chunk_id FROM lore_chunks WHERE page_key = ?)", (page["page_key"],))
        conn.execute("DELETE FROM lore_chunks WHERE page_key = ?", (page["page_key"],))
        conn.executemany(
            "INSERT INTO lore_chunks (chunk_id, page_key, game_id, position, section, kind, text, spoiler) "
            "VALUES (:chunk_id, :page_key, :game_id, :position, :section, :kind, :text, :spoiler)",
            [{**c, "page_key": page["page_key"], "game_id": page["game_id"],
              "spoiler": int(c.get("spoiler", False))} for c in chunks])
        # keep the keyword index in the same transaction, so they can't drift apart
        conn.executemany(
            "INSERT INTO lore_fts (chunk_id, game_id, title, section, text) VALUES (?, ?, ?, ?, ?)",
            [(c["chunk_id"], page["game_id"], page["title"], c["section"], c["text"]) for c in chunks])


def get_lore_chunks(conn: sqlite3.Connection) -> list[dict]:
    """All chunks joined with their page + game info (for embedding/citing)."""
    rows = conn.execute(
        "SELECT c.chunk_id, c.game_id, g.name AS game, p.title, p.url, c.section, c.kind, "
        "c.text, c.spoiler, p.page_spoiler, p.categories, c.position "
        "FROM lore_chunks c JOIN wiki_pages p ON p.page_key = c.page_key "
        "JOIN games g ON g.id = c.game_id ORDER BY c.chunk_id")
    out = []
    for r in rows:
        d = dict(r)
        d["categories"] = json.loads(d["categories"])
        out.append(d)
    return out


def lore_stats(conn: sqlite3.Connection) -> list[tuple]:
    return conn.execute(
        "SELECT g.name, COUNT(DISTINCT p.page_key), COUNT(c.chunk_id) FROM wiki_pages p "
        "JOIN games g ON g.id = p.game_id LEFT JOIN lore_chunks c ON c.page_key = p.page_key "
        "GROUP BY g.name ORDER BY g.name").fetchall()


def pages_with_wikitext(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT page_key, game_id, wiki, title, url, revid, categories, wikitext "
                        "FROM wiki_pages WHERE wikitext IS NOT NULL")
    return [{**dict(r), "categories": json.loads(r["categories"])} for r in rows]


def pages_missing_wikitext(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM wiki_pages WHERE wikitext IS NULL").fetchone()[0]


_STOPWORDS = frozenset("""a an and are as at be but by did do does for from had has have how i in
is it its me my of on or so that the their them then there these they this to was were what
when where which who whom why will with would you your can could should about into""".split())


def fts_query(question: str) -> str | None:
    """Turn free text into a SAFE FTS5 query.

    FTS5 has its own syntax: quotes, -, *, :, AND/OR/NOT, NEAR, ^ ... are
    operators, so raw user text can error out or mean something unintended.
    We keep only word tokens, drop stopwords, quote each one, and OR them:
    BM25 then ranks chunks matching more (and rarer) words higher.
    """
    words = [w for w in re.findall(r"\w+", question.lower()) if w not in _STOPWORDS]
    words = list(dict.fromkeys(words))            # dedupe, keep order
    return " OR ".join(f'"{w}"' for w in words) if words else None


def bm25_search(conn: sqlite3.Connection, question: str, k: int,
                game_id: int | None = None) -> list[tuple[str, float]]:
    """[(chunk_id, score)] best first. bm25() returns lower = better; columns are
    weighted: a match in the page title counts more than one in the body."""
    q = fts_query(question)
    if q is None:
        return []
    sql = ("SELECT chunk_id, bm25(lore_fts, 0, 0, 4.0, 2.0, 1.0) AS s FROM lore_fts "
           "WHERE lore_fts MATCH ?" + (" AND game_id = ?" if game_id is not None else "") +
           " ORDER BY s LIMIT ?")
    params = (q, game_id, k) if game_id is not None else (q, k)
    return [(r[0], r[1]) for r in conn.execute(sql, params)]


def get_chunks_by_ids(conn: sqlite3.Connection, chunk_ids: list[str]) -> dict[str, dict]:
    if not chunk_ids:
        return {}
    marks = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        "SELECT c.chunk_id, c.game_id, g.name AS game, p.title, p.url, c.section, c.kind, "
        f"c.text, c.spoiler, p.page_spoiler FROM lore_chunks c JOIN wiki_pages p ON p.page_key = c.page_key "
        f"JOIN games g ON g.id = c.game_id WHERE c.chunk_id IN ({marks})", chunk_ids)
    return {r["chunk_id"]: dict(r) for r in rows}


def wiki_titles_by_game(conn: sqlite3.Connection) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name, title in conn.execute(
            "SELECT g.name, p.title FROM wiki_pages p JOIN games g ON g.id = p.game_id"):
        out.setdefault(name, []).append(title)
    return out
