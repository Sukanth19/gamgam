"""Tag wiki sections with story milestones, and manage spoiler progress.

    python -m app.spoilers tag [--game "Hollow Knight"] [--limit 5]
    python -m app.spoilers status
    python -m app.spoilers progress "Hollow Knight" "City of Tears"
    python -m app.spoilers progress "Hollow Knight" off
    python -m app.spoilers fix "Witcher 3" "Kaer Morhen" "History > After the end" 5

Why an LLM? Hollow Knight's and Witcher's wikis don't mark spoilers at all,
and doing 3,000+ sections by hand isn't realistic. Gemini reads each
section's heading + first ~120 words and picks a milestone. Decisions:
- Batches of ~10 pages per call (~80 calls total) to fit the free tier.
- When unsure, pick the LATER milestone: hiding too much is the safe error.
- Untagged sections count as spoilers while protection is on, so a partial
  or failed run can never leak anything.
- Resumable: tags are saved after every call; re-runs only do what's missing.
"""

import argparse
import logging
import sys
import time

from pydantic import BaseModel

from app import config, db, llm
from app.progress import milestones_for
from app.wikis import WIKIS

log = logging.getLogger(__name__)


class SectionTag(BaseModel):
    page: int
    section: int
    milestone: int


def build_prompt(game: str, pages: list[dict]) -> str:
    ms = milestones_for(game)
    lines = [
        f"You tag wiki sections about the video game {game} for SPOILER control.",
        "Story milestones, in the order a player reaches them:",
    ]
    lines += [f"  {i}: {m.label} — {m.description}" for i, m in enumerate(ms)]
    lines += [
        "",
        "For EACH section below, give the EARLIEST milestone by which a player would already",
        "know everything the section reveals (characters' true identities, deaths, plot twists,",
        "endings, late-game areas). Rules:",
        "- Judge by the most spoilery thing the section is likely to say, using its heading too",
        "  (a section called 'Ending' or 'Fate' is late even if the sample looks harmless).",
        "- Basic gameplay info (enemy stats, how to fight, where to buy) belongs to the milestone",
        "  where that enemy/place/item is first encountered.",
        "- If unsure between two milestones, choose the LATER one.",
        "",
    ]
    for pi, page in enumerate(pages):
        lines.append(f"PAGE {pi}: {page['title']}")
        for si, sec in enumerate(page["sections"][:config.SPOILER_MAX_SECTIONS]):
            lines.append(f"  SECTION {si}: [{sec['section']}] {sec['sample']}")
    lines.append("\nReturn a JSON list of {page, section, milestone} - one per section above.")
    return "\n".join(lines)


def tag_batch(client, game: str, pages: list[dict]) -> list[tuple[str, str, int, int]] | None:
    """Returns rows for db.save_section_milestones, or None if Gemini failed."""
    n_ms = len(milestones_for(game))
    out = llm.generate_json(client, build_prompt(game, pages), list[SectionTag],
                            temperature=0.0, what="spoiler tags")
    if out is None:
        return None
    rows = []
    for t in out:
        if not (0 <= t.page < len(pages)):
            continue
        page = pages[t.page]
        secs = page["sections"][:config.SPOILER_MAX_SECTIONS]
        if not (0 <= t.section < len(secs)):
            continue
        milestone = min(max(t.milestone, 0), n_ms - 1)     # clamp out-of-range answers
        rows.append((page["page_key"], secs[t.section]["raw_section"], milestone, page["revid"]))
    return rows


def tag_game(conn, client, game_name: str, limit: int | None = None, retag: bool = False) -> bool:
    """Tag all untagged sections of one game. Returns False if the LLM gave up.
    retag: delete existing tags first (safe: untagged sections count as spoilers)."""
    game = db.find_game_by_name(conn, game_name)
    if retag:
        print(f"{game_name}: cleared {db.clear_section_milestones(conn, game['id'])} old tags")
    pages = db.sections_needing_tags(conn, game["id"], config.SPOILER_SAMPLE_WORDS)
    if limit:
        pages = pages[:limit]
    total = sum(min(len(p["sections"]), config.SPOILER_MAX_SECTIONS) for p in pages)
    print(f"{game_name}: {len(pages)} pages / {total} sections to tag", flush=True)
    done = 0
    for start in range(0, len(pages), config.SPOILER_PAGES_PER_CALL):
        batch = pages[start:start + config.SPOILER_PAGES_PER_CALL]
        rows = tag_batch(client, game_name, batch)
        if rows is None:
            print("  Gemini unavailable (busy or out of quota). Progress is saved; re-run later.")
            return False
        db.save_section_milestones(conn, rows)
        done += len(rows)
        print(f"  tagged {done}/{total} sections", flush=True)
        time.sleep(config.SPOILER_CALL_INTERVAL_S)
    return True


def main():
    ap = argparse.ArgumentParser(description="Spoiler tagging and progress")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tag")
    t.add_argument("--game")
    t.add_argument("--limit", type=int, help="only this many pages per game (testing)")
    t.add_argument("--provider", choices=["gemini", "ollama"], default=config.SPOILER_TAG_PROVIDER,
                   help="LLM for tagging (default gemini: it knows the games' plot twists)")
    t.add_argument("--retag", action="store_true", help="discard existing tags and tag again")
    sub.add_parser("status")
    f = sub.add_parser("fix", help="manually set one section's milestone (fix a leak/over-hide)")
    f.add_argument("game")
    f.add_argument("page")
    f.add_argument("section", help='section path as shown in sources, "" for the intro')
    f.add_argument("milestone", type=int)
    p = sub.add_parser("progress")
    p.add_argument("game")
    p.add_argument("milestone", help='milestone name/number, or "off"')
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="  [%(levelname)s] %(message)s")

    conn = db.connect()
    try:
        if args.cmd == "tag":
            client = llm.make_client(args.provider)
            if client is None:
                sys.exit(f"error: {args.provider} isn't available for tagging (see warnings above)")
            print(f"tagging with {args.provider}")
            for w in WIKIS:
                if args.game and args.game.lower() not in w["game"].lower():
                    continue
                if not tag_game(conn, client, w["game"], args.limit, args.retag):
                    sys.exit(2)       # non-zero, so chained commands (&&) stop too
        elif args.cmd == "fix":
            game = db.find_game_by_name(conn, args.game)
            row = conn.execute("SELECT page_key, revid FROM wiki_pages WHERE game_id = ? AND title = ?",
                               (game["id"], args.page)).fetchone() if game else None
            if row is None:
                sys.exit(f"error: no wiki page {args.page!r} for {args.game!r}")
            ok = conn.execute("SELECT 1 FROM lore_chunks WHERE page_key = ? AND section = ?",
                              (row[0], args.section)).fetchone()
            if not ok:
                secs = [r[0] for r in conn.execute("SELECT DISTINCT section FROM lore_chunks "
                                                   "WHERE page_key = ?", (row[0],))]
                sys.exit(f"error: no section {args.section!r}. Sections: {secs}")
            ms = milestones_for(game["name"])
            if not 0 <= args.milestone < len(ms):
                sys.exit(f"error: milestone must be 0-{len(ms) - 1}")
            db.save_section_milestones(conn, [(row[0], args.section, args.milestone, row[1])])
            print(f"{args.page} > {args.section or '(intro)'}: now '{ms[args.milestone].label}'")
        elif args.cmd == "status":
            for name, tagged, total in db.milestone_coverage(conn):
                game = db.find_game_by_name(conn, name)
                prog = db.get_progress(conn, game["id"])
                where = milestones_for(name)[prog].label if prog is not None else "off (show everything)"
                print(f"{name:<28} tagged {tagged:>5}/{total:<5}  protection: {where}")
        else:
            from app.progress import parse_progress
            game = db.find_game_by_name(conn, args.game)
            if game is None or not milestones_for(game["name"]):
                sys.exit(f"error: no milestones for {args.game!r}")
            ms = milestones_for(game["name"])
            if args.milestone.lower() == "off":
                idx = None
            elif args.milestone.isdigit():
                idx = int(args.milestone)
            else:
                idx = parse_progress(game["name"], args.milestone)
            if idx is not None and not (0 <= idx < len(ms)):
                idx = None
            if idx is None and args.milestone.lower() != "off":
                sys.exit("error: unknown milestone. Options:\n" +
                         "\n".join(f"  {i}: {m.label}" for i, m in enumerate(ms)))
            db.set_progress(conn, game["id"], idx)
            print(f"{game['name']}: " + (f"protection on, up to '{ms[idx].label}'" if idx is not None
                                         else "protection off"))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
