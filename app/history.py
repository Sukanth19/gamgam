"""Record which games you liked or disliked.

    python -m app.history like "Hollow Knight" "Celeste" "Hades"
    python -m app.history dislike "FIFA 23" --note "not into sports"
    python -m app.history list
    python -m app.history remove "FIFA 23"
"""

import argparse
import sys

from app import db


def main():
    ap = argparse.ArgumentParser(description="Record liked/disliked games")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("like", "dislike"):
        p = sub.add_parser(cmd)
        p.add_argument("names", nargs="+")
        p.add_argument("--note")
    sub.add_parser("list")
    sub.add_parser("remove").add_argument("names", nargs="+")
    args = ap.parse_args()

    conn = db.connect()
    try:
        if args.cmd == "list":
            rows = db.get_history(conn)
            if not rows:
                print('no history yet. try: python -m app.history like "Hollow Knight"')
            for r in rows:
                mark = "+" if r["verdict"] == "liked" else "-"
                print(f" {mark} {r['name']}" + (f"  ({r['note']})" if r["note"] else ""))
            return

        failed = False
        for name in args.names:
            game = db.find_game_by_name(conn, name)
            if game is None:
                print(f"  ? no game matching {name!r}", file=sys.stderr)
                failed = True
                continue
            if args.cmd == "remove":
                ok = db.remove_verdict(conn, game["id"])
                print(f"  removed {game['name']}" if ok else f"  {game['name']} wasn't in history")
            else:
                verdict = "liked" if args.cmd == "like" else "disliked"
                db.set_verdict(conn, game["id"], verdict, args.note)
                # Show the matched name so fuzzy-match surprises are visible.
                print(f"  {verdict}: {game['name']} ({(game.get('released') or '?')[:4]})")
        if failed:
            sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
