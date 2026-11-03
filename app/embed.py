"""Embed game profiles from SQLite into ChromaDB.

    python -m app.embed

Re-runs are cheap: a game is skipped when its stored profile_hash matches the
current profile (so only new or changed games hit the Gemini API).
"""

import argparse
import logging

from app import db
from app.embeddings import Embedder, build_profile, profile_hash
from app.vector_store import GameVectorStore


def plan_work(games: list[dict], existing: dict[int, str]):
    """Return (games, profiles, hashes) that need (re-)embedding."""
    todo = []
    for g in games:
        p = build_profile(g)
        h = profile_hash(p)
        if existing.get(g["id"]) != h:
            todo.append((g, p, h))
    return todo


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=None, help="embed at most N games (testing)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    conn = db.connect()
    games = db.get_all_games(conn)
    conn.close()
    store = GameVectorStore()
    todo = plan_work(games, store.existing_hashes())
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(games)} games in db, {len(games) - len(todo)} already embedded, {len(todo)} to embed")
    if not todo:
        return

    embedder = Embedder()
    for start, vectors in embedder.embed([p for _, p, _ in todo]):
        chunk = todo[start:start + len(vectors)]
        store.upsert([g for g, _, _ in chunk], vectors, [h for _, _, h in chunk])
        print(f"  embedded {start + len(vectors)}/{len(todo)}", flush=True)
    print(f"done. vectors in store: {store.count()}")


if __name__ == "__main__":
    main()
