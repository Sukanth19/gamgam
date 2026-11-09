"""Evaluation harness: turn "looks better" into numbers.

    python -m app.eval                          # all suites
    python -m app.eval --suite retrieval        # fast, no LLM
    python -m app.eval --suite similar          # fast, no LLM
    python -m app.eval --suite lore --suite spoilers
    python -m app.eval --set MIN_ANSWER_SIMILARITY=0.5 --tag thr050   # try a setting
    python -m app.eval --compare eval/results/A.json eval/results/B.json

Suites (question sets live in eval/*.jsonl):
- retrieval: is an expected page among the top-k chunks? hit@1/3/8 and MRR
  (mean reciprocal rank: 1/rank of the first right chunk, averaged). No LLM.
- lore: full ask() pipeline. Answer rate on answerable questions, "I don't
  know" rate on unanswerable ones, and whether citations include an expected page.
- spoilers: ask() at a given progress. LEAK = should be hidden but answered
  (the bad error); OVER-HIDE = should be answered but hidden (annoying).
  Progress is passed to ask() directly - your saved progress is never touched.
- similar: for seed games, do the top 10 include known-good matches?

Results are saved to eval/results/ with the settings used, so two runs can be
compared item by item.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from app import config, db

EVAL_DIR = config.ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"
SUITES = ("retrieval", "similar", "lore", "spoilers")

# Settings worth recording with every run (so results are reproducible/comparable).
TRACKED = ["LLM_PROVIDER", "OLLAMA_MODEL", "EMBEDDING_MODEL", "MIN_ANSWER_SIMILARITY",
           "ANSWER_CONTEXT_CHUNKS", "RETRIEVE_POOL", "RRF_K", "SPOILER_HIDDEN_MARGIN",
           "PROFILE_INCLUDE_NAME", "PROFILE_RANK_TAGS", "CHROMA_COLLECTION"]


def load(name: str) -> list[dict]:
    path = EVAL_DIR / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class Ctx:
    """Lazily built shared resources (models are slow to load)."""
    def __init__(self):
        self.conn = db.connect()
        self._emb = self._lore = self._games = self._client = None
        self.client_loaded = False

    @property
    def emb(self):
        if self._emb is None:
            from app.embeddings import make_embedder
            self._emb = make_embedder("query")
        return self._emb

    @property
    def lore(self):
        if self._lore is None:
            from app.vector_store import LoreVectorStore
            self._lore = LoreVectorStore()
        return self._lore

    @property
    def games(self):
        if self._games is None:
            from app.vector_store import GameVectorStore
            self._games = GameVectorStore()
        return self._games

    @property
    def client(self):
        if not self.client_loaded:
            from app import llm
            self._client, self.client_loaded = llm.make_client(), True
        return self._client

    def game(self, name: str) -> dict:
        g = db.find_game_by_name(self.conn, name)
        if g is None:
            raise LookupError(f"eval references unknown game {name!r}")
        return g


# --- suites -----------------------------------------------------------------------

def run_retrieval(ctx: Ctx) -> tuple[dict, list[dict]]:
    from app.retrieval import hybrid_search
    items = []
    for q in load("lore_questions"):
        if not q["answerable"]:
            continue
        g = ctx.game(q["game"])
        vec = next(ctx.emb.embed([q["question"]]))[1][0]
        hits = hybrid_search(ctx.conn, ctx.lore, vec, q["question"], g["id"])
        rank = next((i for i, h in enumerate(hits, 1) if h["title"] in q["expect_pages"]), None)
        items.append({"id": q["question"], "rank": rank, "top": [h["title"] for h in hits[:3]]})
    n = len(items)
    ranks = [it["rank"] for it in items]
    metrics = {f"hit@{k}": sum(1 for r in ranks if r and r <= k) / n for k in (1, 3, 8)}
    metrics["mrr"] = sum(1 / r for r in ranks if r) / n
    return metrics, items


def run_similar(ctx: Ctx) -> tuple[dict, list[dict]]:
    from app.similar import find_similar
    items = []
    for case in load("similar_games"):
        game, hits = find_similar(ctx.conn, ctx.games, case["game"], k=10)
        names = [h["name"] for h in hits]
        good = [n for n in names if n in case["good"]]
        bad = [n for n in names if n in case.get("bad", [])]
        items.append({"id": case["game"], "good_in_top10": len(good), "bad_in_top10": len(bad),
                      "pass": bool(good) and not bad, "top": names[:5], "bad": bad})
    n = len(items)
    metrics = {"pass_rate": sum(it["pass"] for it in items) / n,
               "good_per_query": sum(it["good_in_top10"] for it in items) / n,
               "bad_rate": sum(it["bad_in_top10"] > 0 for it in items) / n}
    return metrics, items


def _ask(ctx: Ctx, question: str, game: dict, milestone=None):
    from app.ask import ask
    return ask(ctx.conn, ctx.lore, ctx.emb, ctx.client, question, game, max_milestone=milestone)


def run_lore(ctx: Ctx) -> tuple[dict, list[dict]]:
    items = []
    for q in load("lore_questions"):
        t = time.time()
        r = _ask(ctx, q["question"], ctx.game(q["game"]))
        cited = [s["title"] for s in r.sources] if r.answerable else []
        if q["answerable"]:
            ok = r.answerable and any(p in cited for p in q["expect_pages"])
        else:
            ok = not r.answerable
        items.append({"id": q["question"], "answerable": q["answerable"], "answered": r.answerable,
                      "pass": ok, "cited": cited[:4], "reason": r.reason,
                      "seconds": round(time.time() - t, 1)})
    pos = [it for it in items if it["answerable"]]
    neg = [it for it in items if not it["answerable"]]
    metrics = {"pass_rate": sum(it["pass"] for it in items) / len(items),
               "answer_rate": sum(it["answered"] for it in pos) / max(len(pos), 1),
               "cite_correct": sum(it["pass"] for it in pos) / max(len(pos), 1),
               "idk_rate": sum(not it["answered"] for it in neg) / max(len(neg), 1),
               "avg_seconds": sum(it["seconds"] for it in items) / len(items)}
    return metrics, items


def run_spoilers(ctx: Ctx) -> tuple[dict, list[dict]]:
    from app.progress import parse_progress
    items = []
    for case in load("spoiler_questions"):
        g = ctx.game(case["game"])
        idx = parse_progress(g["name"], case["progress"])
        if idx is None:
            raise ValueError(f"can't parse progress {case['progress']!r} for {g['name']}")
        r = _ask(ctx, case["question"], g, milestone=idx)
        outcome = "answer" if r.answerable else "hidden"
        items.append({"id": f"{case['game']} @ {case['progress']}: {case['question']}",
                      "expect": case["expect"], "got": outcome, "pass": outcome == case["expect"],
                      "reason": r.reason})
    should_hide = [it for it in items if it["expect"] == "hidden"]
    should_show = [it for it in items if it["expect"] == "answer"]
    metrics = {"pass_rate": sum(it["pass"] for it in items) / len(items),
               "leak_rate": sum(it["got"] == "answer" for it in should_hide) / max(len(should_hide), 1),
               "over_hide_rate": sum(it["got"] == "hidden" for it in should_show) / max(len(should_show), 1)}
    return metrics, items


RUNNERS = {"retrieval": run_retrieval, "similar": run_similar, "lore": run_lore,
           "spoilers": run_spoilers}


# --- reporting ----------------------------------------------------------------------

def print_suite(name: str, metrics: dict, items: list[dict]):
    print(f"\n== {name} ==")
    print("  " + "   ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}"
                            for k, v in metrics.items()))
    for it in items:
        ok = it.get("pass", it.get("rank") is not None)
        if not ok:
            detail = {k: v for k, v in it.items() if k not in ("id", "pass")}
            print(f"  FAIL  {it['id']}  {detail}")


def compare(path_a: str, path_b: str):
    a, b = json.loads(Path(path_a).read_text()), json.loads(Path(path_b).read_text())
    changed = {k: (a["settings"].get(k), b["settings"].get(k)) for k in TRACKED
               if a["settings"].get(k) != b["settings"].get(k)}
    print(f"A = {path_a}\nB = {path_b}\nsettings changed: {changed or 'none'}")
    for suite in SUITES:
        if suite not in a["suites"] or suite not in b["suites"]:
            continue
        ma, mb = a["suites"][suite]["metrics"], b["suites"][suite]["metrics"]
        print(f"\n== {suite} ==")
        for k in ma:
            if k in mb:
                d = mb[k] - ma[k]
                print(f"  {k:<16} {ma[k]:6.2f} -> {mb[k]:6.2f}  ({d:+.2f})")
        ia = {it["id"]: it for it in a["suites"][suite]["items"]}
        for it in b["suites"][suite]["items"]:
            before = ia.get(it["id"])
            if before is None:
                continue
            pa = before.get("pass", before.get("rank") is not None)
            pb = it.get("pass", it.get("rank") is not None)
            if pa != pb:
                print(f"  {'FIXED ' if pb else 'BROKE '} {it['id']}")


def main():
    ap = argparse.ArgumentParser(description="Run evaluation suites")
    ap.add_argument("--suite", action="append", choices=SUITES, help="repeatable; default: all")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="override a config value for this run (read at call time)")
    ap.add_argument("--tag", default="", help="label for the results file")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()
    if args.compare:
        return compare(*args.compare)

    for kv in args.set:
        name, _, value = kv.partition("=")
        if not hasattr(config, name):
            sys.exit(f"error: config has no setting {name!r}")
        old = getattr(config, name)
        setattr(config, name, type(old)(value) if not isinstance(old, bool) else value in ("1", "true"))
        print(f"override: {name} = {getattr(config, name)!r} (was {old!r})")

    ctx = Ctx()
    report = {"when": datetime.now().isoformat(timespec="seconds"), "tag": args.tag,
              "settings": {k: getattr(config, k, None) for k in TRACKED}, "suites": {}}
    for suite in args.suite or SUITES:
        t = time.time()
        metrics, items = RUNNERS[suite](ctx)
        metrics["seconds_total"] = round(time.time() - t, 1)
        report["suites"][suite] = {"metrics": metrics, "items": items}
        print_suite(suite, metrics, items)

    # An LLM outage (crash, OOM, quota) isn't a wrong answer. Counting it as one would
    # silently corrupt comparisons, so such a run is reported but never saved.
    infra = [it["id"] for suite in report["suites"].values() for it in suite["items"]
             if it.get("reason") in ("llm failed", "no llm")]
    if infra:
        print(f"\nINVALID RUN: the LLM was unavailable for {len(infra)} item(s), e.g. {infra[0]!r}.")
        print("Not saved. Check the LLM (is Ollama running? out of memory?) and re-run.")
        sys.exit(1)

    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = RESULTS_DIR / f"{stamp}{'-' + args.tag if args.tag else ''}.json"
        out.write_text(json.dumps(report, indent=1))
        print(f"\nsaved {out.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
