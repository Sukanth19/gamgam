"""Answer lore questions from the wiki chunks, with citations.

    python -m app.ask "who is the pale king" --game "Hollow Knight"
    python -m app.ask "why did the Nomai come here" --game "Outer Wilds" --debug

Pipeline: hybrid retrieval (vector + BM25, RRF) -> relevance gate -> grounded
Gemini answer with numbered sources -> citation check -> answer + source list.

"I don't know" can happen at two points:
1. Before the LLM: the best match is too weak (MIN_ANSWER_SIMILARITY), so the
   wiki probably doesn't cover it. Saves an API call, and avoids handing the
   model irrelevant text it might stretch into an answer.
2. The LLM itself reports the sources don't contain the answer.
"""

import argparse
import logging
import re
import sys
from dataclasses import dataclass, field

from pydantic import BaseModel

from app import config, db, llm
from app.retrieval import best_similarity, hybrid_search

IDK = "I don't know — the wiki pages I have don't cover that."


class LoreAnswer(BaseModel):
    answerable: bool
    answer: str
    citations: list[int]


@dataclass
class Result:
    answer: str
    answerable: bool
    sources: list[dict] = field(default_factory=list)    # the cited chunks, numbered
    reason: str = ""                                     # why we said "I don't know"
    hits: list[dict] = field(default_factory=list)       # everything retrieved (debug)
    revealed_at: int | None = None                       # milestone where the hidden answer appears


def build_prompt(question: str, game: str, hits: list[dict]) -> str:
    lines = [
        f"You answer questions about the lore of the video game {game}.",
        "Use ONLY the numbered sources below - no outside knowledge, even if you know the game.",
        "- After every claim, cite its source like [2]. Cite only source numbers that exist.",
        "- If the sources don't contain the answer, set answerable=false and say so briefly.",
        "- Some sources may describe other media (books, comics, board games, spin-offs); prefer",
        f"  sources about {game} itself and mention it if you rely on other media.",
        "- If sources disagree, say so. Keep the answer under ~150 words.",
        "",
        f"QUESTION: {question}",
        "",
        "SOURCES:",
    ]
    for i, h in enumerate(hits, 1):
        where = h["title"] + (f" > {h['section']}" if h["section"] else "")
        lines.append(f"[{i}] ({where})\n{h['text']}\n")
    lines.append("Return JSON: {answerable, answer, citations: [source numbers used]}.")
    return "\n".join(lines)


_CITE = re.compile(r"\[(\d+)\]")
_MULTI_CITE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)+)\]")


def split_multi_citations(text: str) -> str:
    """Models sometimes write "[1, 2]": turn it into "[1][2]" so every number
    gets checked and renumbered like a single citation."""
    return _MULTI_CITE.sub(lambda m: "".join(f"[{n}]" for n in re.findall(r"\d+", m.group(1))), text)


def check_citations(answer: str, cited: list[int], n_sources: int) -> tuple[str, list[int]]:
    """Drop citation numbers that don't match a source we actually sent."""
    valid = set(range(1, n_sources + 1))
    answer = split_multi_citations(answer)
    text = _CITE.sub(lambda m: m.group(0) if int(m.group(1)) in valid else "", answer)
    used = {int(n) for n in _CITE.findall(text)} | {c for c in cited if c in valid}
    return re.sub(r"\s+([.,;])", r"\1", text).strip(), sorted(used)


def is_visible(hit: dict, max_milestone: int | None) -> bool:
    """With protection on, only sections tagged at/before the user's progress
    are visible. Untagged sections are hidden: never guess in the leaky direction."""
    if max_milestone is None:
        return True
    return hit.get("milestone") is not None and hit["milestone"] <= max_milestone


def revealed_later(game: dict, hidden: list[dict], max_milestone: int, hits: list[dict]) -> Result:
    from app.progress import milestones_for
    ms = milestones_for(game["name"])
    best = max(hidden, key=lambda h: h["similarity"] or 0)
    at = best.get("milestone")
    where = f"around **{ms[at].label}**" if at is not None and at < len(ms) else "later on"
    text = (f"That's revealed later in the game — {where}. You're at **{ms[max_milestone].label}**, "
            "so I'm keeping it hidden. Update your progress or turn spoiler protection off to see it.")
    return Result(text, False, reason="revealed later", hits=hits, revealed_at=at)


def ask(conn, store, embedder, client, question: str, game: dict,
        max_milestone: int | None = None) -> Result:
    """max_milestone: the user's progress (None = no spoiler protection)."""
    query_vec = next(embedder.embed([question]))[1][0]
    if max_milestone is None:
        hits, hidden = hybrid_search(conn, store, query_vec, question, game["id"]), []
    else:
        # fetch a bigger pool, since some of it will be hidden
        pool = hybrid_search(conn, store, query_vec, question, game["id"],
                             k=config.SPOILER_RETRIEVE_POOL)
        hits = [h for h in pool if is_visible(h, max_milestone)][:config.ANSWER_CONTEXT_CHUNKS]
        hidden = [h for h in pool if not is_visible(h, max_milestone)]
    best = best_similarity(hits)
    # Is there a strong answer we're hiding? Then "I don't know" would be a lie.
    later = bool(hidden) and best_similarity(hidden) >= max(config.MIN_ANSWER_SIMILARITY, best)
    if not hits or best < config.MIN_ANSWER_SIMILARITY:
        if later:
            return revealed_later(game, hidden, max_milestone, hits)
        return Result(IDK, False, reason=f"best match {best:.2f} < {config.MIN_ANSWER_SIMILARITY}",
                      hits=hits)
    if client is None:
        return Result("(Gemini unavailable: no API key) - top sources below.", False,
                      sources=hits[:3], reason="no llm", hits=hits)

    out = llm.generate_json(client, build_prompt(question, game["name"], hits), LoreAnswer,
                            temperature=0.2, what="answer")
    if out is None:
        return Result("(Gemini is unavailable right now) - here are the most relevant sources.",
                      False, sources=hits[:3], reason="llm failed", hits=hits)
    if not out.answerable:
        if later:
            return revealed_later(game, hidden, max_milestone, hits)
        return Result(IDK, False, reason="model: sources don't contain the answer", hits=hits)

    text, used = check_citations(out.answer, out.citations, len(hits))
    if not used:
        # An answer with no valid citation isn't grounded in our sources: don't trust it.
        return Result(IDK, False, reason="answer had no valid citations", hits=hits)
    # Renumber so the printed source list is [1], [2], ... in order of first use.
    order = list(dict.fromkeys(int(n) for n in _CITE.findall(text))) or used
    order += [u for u in used if u not in order]
    renum = {old: new for new, old in enumerate(order, 1)}
    text = _CITE.sub(lambda m: f"[{renum[int(m.group(1))]}]", text)
    return Result(text, True, sources=[hits[i - 1] for i in order], hits=hits)


def main():
    ap = argparse.ArgumentParser(description="Ask a lore question")
    ap.add_argument("question")
    ap.add_argument("--game", required=True)
    ap.add_argument("--debug", action="store_true", help="show retrieval scores")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="  [%(levelname)s] %(message)s")
    logging.getLogger("app.llm").setLevel(logging.INFO)   # show which model answered

    conn = db.connect()
    try:
        game = db.find_game_by_name(conn, args.game)
        if game is None:
            sys.exit(f"error: no game matching {args.game!r}")
        from app.embeddings import make_embedder
        from app.vector_store import LoreVectorStore
        res = ask(conn, LoreVectorStore(), make_embedder("query"), llm.make_client(),
                  args.question, game, max_milestone=db.get_progress(conn, game["id"]))
    finally:
        conn.close()

    if args.debug:
        print(f"retrieved ({res.reason or 'answered'}):")
        for h in res.hits:
            sim = f"{h['similarity']:.3f}" if h["similarity"] is not None else "  -  "
            ms = h.get("milestone")
            print(f"  rrf {h['rrf']:.4f}  sim {sim}  vec#{h['vec_rank'] or '-':<3} "
                  f"bm25#{h['bm25_rank'] or '-':<3} m{ms if ms is not None else '?'}  "
                  f"{h['title']} > {h['section']}")
        print()
    print(res.answer)
    if res.sources:
        print("\nSources:")
        for i, s in enumerate(res.sources, 1):
            where = s["title"] + (f" > {s['section']}" if s["section"] else "")
            flag = "  (spoiler)" if s["spoiler"] or s["page_spoiler"] else ""
            print(f"  [{i}] {where}{flag}\n      {s['url']}")
        print("\nText from community wikis, CC BY-SA / CC BY; see source pages for authors.")


if __name__ == "__main__":
    main()
