"""Fetch wiki lore, chunk it, embed it, and search it.

    python -m app.wiki fetch                       # all games in app/wikis.py
    python -m app.wiki fetch --game "Outer Wilds"  # one game
    python -m app.wiki rechunk                     # re-chunk stored pages, no network
    python -m app.wiki embed
    python -m app.wiki search "who is the pale king" [--game "Hollow Knight"] [-k 5]
    python -m app.wiki stats

Re-runs are cheap: pages whose revision hasn't changed aren't re-downloaded,
and chunks whose text hasn't changed aren't re-embedded.
"""

import argparse
import logging
import sys

import httpx

from app import db
from app.chunking import parse_page
from app.embeddings import chunk_hash, make_embedder
from app.vector_store import LoreVectorStore
from app.wiki_client import WikiClient, select_pages
from app.wikis import SKIP_TITLE_RE, WIKIS


def chunk_embed_text(c: dict) -> str:
    """What gets embedded: a short header gives the chunk its context
    ("Hornet > Lore" means much more than the paragraph alone)."""
    where = c["title"] + (f" > {c['section']}" if c["section"] else "")
    return f"{c['game']} — {where}\n{c['text']}"


def chunks_for(key: str, wikitext: str):
    parsed = parse_page(wikitext)
    chunks = [{"chunk_id": f"{key}#{i}", "position": i, "section": ch.section,
               "kind": ch.kind, "text": ch.text, "spoiler": ch.spoiler}
              for i, ch in enumerate(parsed.chunks)]
    return chunks, parsed.page_spoiler


def rechunk(conn) -> None:
    """Re-run the chunker on stored wikitext: no network needed."""
    n_pages = n_chunks = 0
    for page in db.pages_with_wikitext(conn):
        chunks, spoiler = chunks_for(page["page_key"], page["wikitext"])
        db.save_page(conn, {**page, "page_spoiler": spoiler}, chunks)
        n_pages += 1
        n_chunks += len(chunks)
    print(f"re-chunked {n_pages} pages -> {n_chunks} chunks")
    missing = db.pages_missing_wikitext(conn)
    if missing:
        print(f"  {missing} pages have no stored wikitext; run `fetch --force` once")


def fetch_wiki(conn, cfg: dict, force: bool = False) -> None:
    game = db.find_game_by_name(conn, cfg["game"])
    if game is None:
        print(f"  ! {cfg['game']!r} is not in the games db; skipping", file=sys.stderr)
        return
    host = httpx.URL(cfg["api"]).host
    client = WikiClient(cfg["api"])
    try:
        pages = select_pages(client, cfg["categories"], cfg["max_pages"], SKIP_TITLE_RE)
        revs = client.page_revisions([p["pageid"] for p in pages])
        known = db.page_revids(conn, host)
        todo = [p["pageid"] for p in pages if p["pageid"] in revs and
                (force or known.get(f"{host}:{p['pageid']}") != revs[p["pageid"]])]
        print(f"{game['name']}: {len(pages)} pages selected, {len(pages) - len(todo)} unchanged, "
              f"{len(todo)} to fetch", flush=True)
        n_chunks = 0
        for start in range(0, len(todo), 100):
            for page in client.fetch_pages(todo[start:start + 100]):
                key = f"{host}:{page['pageid']}"
                chunks, spoiler = chunks_for(key, page["wikitext"])
                db.save_page(conn, {"page_key": key, "game_id": game["id"], "wiki": host,
                                    "title": page["title"], "url": page["url"],
                                    "revid": page["revid"], "categories": page["categories"],
                                    "page_spoiler": spoiler, "wikitext": page["wikitext"]}, chunks)
                n_chunks += len(chunks)
            print(f"  fetched {min(start + 100, len(todo))}/{len(todo)} pages", flush=True)
        print(f"  {n_chunks} chunks written")
    finally:
        client.close()


def embed_lore(conn) -> None:
    chunks = db.get_lore_chunks(conn)
    store = LoreVectorStore()
    existing = store.existing_hashes()
    current = {c["chunk_id"] for c in chunks}
    stale = [i for i in existing if i not in current]      # pages that changed/shrank
    if stale:
        store.delete(stale)
    todo = []
    for c in chunks:
        text = chunk_embed_text(c)
        h = chunk_hash(text)
        if existing.get(c["chunk_id"]) != h:
            todo.append((c, text, h))
    print(f"{len(chunks)} chunks, {len(chunks) - len(todo)} already embedded, "
          f"{len(todo)} to embed, {len(stale)} stale removed")
    if not todo:
        return
    embedder = make_embedder("document")
    for start, vectors in embedder.embed([t for _, t, _ in todo]):
        part = todo[start:start + len(vectors)]
        store.upsert([c for c, _, _ in part], vectors, [h for _, _, h in part])
        done = start + len(vectors)
        if done % 640 < len(vectors) or done == len(todo):
            print(f"  embedded {done}/{len(todo)}", flush=True)
    print(f"done. lore vectors: {store.count()}")


def search(conn, query: str, k: int, game_name: str | None) -> None:
    game_id = None
    if game_name:
        game = db.find_game_by_name(conn, game_name)
        if game is None:
            sys.exit(f"error: no game matching {game_name!r}")
        game_id = game["id"]
    vec = next(make_embedder("query").embed([query]))[1][0]
    hits = LoreVectorStore().search(vec, k, game_id)
    for i, h in enumerate(hits, 1):
        where = h["title"] + (f" > {h['section']}" if h["section"] else "")
        flag = "  [spoiler]" if h["spoiler"] else ""
        print(f"{i}. {h['similarity']:.3f}  {h['game']} — {where}{flag}\n   {h['url']}")
        print(f"   {h['text'][:300]}{'...' if len(h['text']) > 300 else ''}\n")


def main():
    ap = argparse.ArgumentParser(description="Wiki lore: fetch, embed, search")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--game")
    f.add_argument("--force", action="store_true", help="re-download even unchanged pages")
    sub.add_parser("rechunk", help="re-run chunking on stored wikitext (offline)")
    sub.add_parser("embed")
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("--game")
    s.add_argument("-k", type=int, default=5)
    sub.add_parser("stats")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="  [%(levelname)s] %(message)s")

    conn = db.connect()
    try:
        if args.cmd == "fetch":
            cfgs = [w for w in WIKIS if not args.game or args.game.lower() in w["game"].lower()]
            if not cfgs:
                sys.exit(f"error: no wiki configured for {args.game!r} (see app/wikis.py)")
            for cfg in cfgs:
                fetch_wiki(conn, cfg, force=args.force)
        elif args.cmd == "rechunk":
            rechunk(conn)
        elif args.cmd == "embed":
            embed_lore(conn)
        elif args.cmd == "search":
            search(conn, args.query, args.k, args.game)
        else:
            for name, pages, chunks in db.lore_stats(conn):
                print(f"{name:<30} {pages:>5} pages {chunks:>6} chunks")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
