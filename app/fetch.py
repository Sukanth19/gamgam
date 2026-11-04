"""Fetch popular games into SQLite from the source set by GAME_SOURCE (.env).

    python -m app.fetch                 # ~2000 games
    python -m app.fetch --limit 50      # quick test

Safe to re-run: games are upserted by id. For sources whose list endpoint
lacks descriptions (RAWG), descriptions are fetched only where missing.
"""

import argparse
import logging

from app import config, db


def make_client():
    if config.GAME_SOURCE == "igdb":
        from app.igdb_client import IgdbClient
        return IgdbClient()
    from app.rawg_client import RawgClient
    return RawgClient()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=config.FETCH_DEFAULT_LIMIT)
    ap.add_argument("--skip-descriptions", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    print(f"source: {config.GAME_SOURCE}  ->  {config.DB_PATH.relative_to(config.ROOT)}")
    conn = db.connect()
    client = make_client()
    try:
        # 1. List games in popularity order, saving every 100 as we go.
        batch, n = [], 0
        for n, game in enumerate(client.iter_popular_games(args.limit), 1):
            batch.append(game)
            if len(batch) == 100:
                db.upsert_games(conn, batch)
                batch = []
                print(f"listed {n} games", flush=True)
        db.upsert_games(conn, batch)
        print(f"listed {n} games; games in db: {db.count_games(conn)}")

        # 2. Descriptions: one detail request per game, only where missing.
        if client.needs_detail_fetch and not args.skip_descriptions:
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
