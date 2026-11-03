"""Fetch popular games from RAWG into SQLite.

    python -m app.fetch                 # ~2000 games
    python -m app.fetch --limit 50      # quick test

Safe to re-run: games are upserted, and descriptions are only fetched for
games that don't have one yet.
"""

import argparse
import logging

from app import config, db
from app.rawg_client import RawgClient


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=config.RAWG_DEFAULT_LIMIT)
    ap.add_argument("--skip-descriptions", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    conn = db.connect()
    client = RawgClient()
    try:
        # 1. List pages: 40 games per request.
        batch = []
        for i, game in enumerate(client.iter_popular_games(args.limit), 1):
            batch.append(game)
            if len(batch) == config.RAWG_PAGE_SIZE:
                db.upsert_games(conn, batch)
                batch = []
                print(f"listed {i} games", flush=True)
        db.upsert_games(conn, batch)
        print(f"games in db: {db.count_games(conn)}")

        # 2. Descriptions: one detail request per game, only where missing.
        if not args.skip_descriptions:
            missing = db.ids_missing_description(conn)
            print(f"fetching descriptions for {len(missing)} games...")
            for i, gid in enumerate(missing, 1):
                desc = client.get_description(gid)
                with conn:
                    conn.execute("UPDATE games SET description = ? WHERE id = ?", (desc, gid))
                if i % 50 == 0 or i == len(missing):
                    print(f"  descriptions {i}/{len(missing)}", flush=True)
    finally:
        client.close()
        conn.close()
    print("done.")


if __name__ == "__main__":
    main()
