"""Find games similar to a given game.

    python -m app.similar "Hollow Knight"
    python -m app.similar "Hollow Knight" -k 5

No API call needed: we reuse the game's stored vector as the query.
"""

import argparse
import sys

from app import db
from app.vector_store import GameVectorStore


def find_similar(conn, store: GameVectorStore, name: str, k: int = 10):
    game = db.find_game_by_name(conn, name)
    if game is None:
        raise LookupError(f"no game matching {name!r} in the database")
    vec = store.get_vector(game["id"])
    if vec is None:
        raise LookupError(f"{game['name']!r} has not been embedded yet (run python -m app.embed)")
    hits = store.query(vec, k + 1)                    # +1: the game itself is its own top hit
    hits = [h for h in hits if h["id"] != game["id"]][:k]
    return game, hits


def main():
    ap = argparse.ArgumentParser(description="Find games similar to a given game")
    ap.add_argument("name")
    ap.add_argument("-k", type=int, default=10)
    args = ap.parse_args()

    conn = db.connect()
    try:
        game, hits = find_similar(conn, GameVectorStore(), args.name, args.k)
    except LookupError as e:
        sys.exit(f"error: {e}")
    finally:
        conn.close()

    year = (game.get("released") or "")[:4]
    print(f"Games similar to {game['name']} ({year}):\n")
    for i, h in enumerate(hits, 1):
        print(f"{i:>2}. {h['similarity']:.3f}  {h['name']} ({h['year'] or '?'})  [{h['genres']}]")


if __name__ == "__main__":
    main()
