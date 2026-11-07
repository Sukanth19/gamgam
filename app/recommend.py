"""Recommend games from your play history.

    python -m app.recommend               # 10 picks, explained by Gemini
    python -m app.recommend -k 5 --no-explain

How it works:
1. Taste vector (Rocchio): mean(liked vectors) - beta * mean(disliked vectors).
   Liking pulls your "taste point" towards games; disliking pushes it away.
2. Fetch the nearest ~60 games to that point, drop ones you've played.
3. MMR re-ranking picks 10 that are relevant *and* not all clones of each other.
4. For each pick, "because you liked X" = your liked game closest to it.
"""

import argparse
import logging
import sys

import numpy as np

from app import config, db
from app.vector_store import GameVectorStore


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n else v


def taste_vector(liked: list, disliked: list, beta: float = config.DISLIKE_WEIGHT) -> np.ndarray:
    if not liked:
        raise ValueError("like at least one game first")
    v = np.mean(np.asarray(liked, dtype=float), axis=0)
    if disliked:
        v = v - beta * np.mean(np.asarray(disliked, dtype=float), axis=0)
    return _unit(v)


def mmr(query: np.ndarray, candidates: list[dict], k: int,
        lam: float = config.MMR_LAMBDA) -> list[dict]:
    """Maximal Marginal Relevance: greedily pick the candidate with the best
    lam * relevance - (1 - lam) * (similarity to the closest pick so far)."""
    vecs = np.asarray([_unit(np.asarray(c["vector"], dtype=float)) for c in candidates])
    relevance = vecs @ query
    chosen: list[int] = []
    while len(chosen) < min(k, len(candidates)):
        redundancy = (vecs @ vecs[chosen].T).max(axis=1) if chosen else np.zeros(len(candidates))
        score = lam * relevance - (1 - lam) * redundancy
        score[chosen] = -np.inf
        chosen.append(int(np.argmax(score)))
    return [{**candidates[i], "taste_score": float(relevance[i])} for i in chosen]


def recommend(conn, store: GameVectorStore, k: int = 10) -> tuple[list[dict], list[dict]]:
    """Returns (picks, history). Each pick has taste_score and because (liked name)."""
    history = db.get_history(conn)
    vecs = store.get_vectors([h["game_id"] for h in history])
    liked = [h for h in history if h["verdict"] == "liked" and h["game_id"] in vecs]
    disliked = [h for h in history if h["verdict"] == "disliked" and h["game_id"] in vecs]
    taste = taste_vector([vecs[h["game_id"]] for h in liked],
                         [vecs[h["game_id"]] for h in disliked])

    played = {h["game_id"] for h in history}
    pool = store.query_with_vectors(list(taste), config.CANDIDATE_POOL + len(played))
    pool = [c for c in pool if c["id"] not in played]
    picks = mmr(taste, pool, k)

    liked_mat = np.asarray([_unit(np.asarray(vecs[h["game_id"]], dtype=float)) for h in liked])
    for p in picks:
        sims = liked_mat @ _unit(np.asarray(p.pop("vector"), dtype=float))
        p["because"] = liked[int(np.argmax(sims))]["name"]
    return picks, history


def main():
    ap = argparse.ArgumentParser(description="Recommend games from your play history")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--no-explain", action="store_true", help="skip the Gemini explanations")
    args = ap.parse_args()
    # show which Gemini model answered / which ones were busy
    logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s")
    for noisy in ("httpx", "google_genai", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    conn = db.connect()
    try:
        picks, history = recommend(conn, GameVectorStore(), args.k)
        explanations = {}
        if not args.no_explain:
            from app.explain import explain_picks
            explanations = explain_picks(conn, picks, history)
    except ValueError as e:
        sys.exit(f'error: {e} (python -m app.history like "Hollow Knight")')
    finally:
        conn.close()

    liked = [h["name"] for h in history if h["verdict"] == "liked"]
    print(f"Based on {len(liked)} liked game(s): {', '.join(liked)}\n")
    for i, p in enumerate(picks, 1):
        print(f"{i:>2}. {p['name']} ({p['year'] or '?'})  [{p['genres']}]")
        print(f"    match {p['taste_score']:.3f} · because you liked {p['because']}")
        if p["id"] in explanations:
            print(f"    {explanations[p['id']]}")
        print()


if __name__ == "__main__":
    main()
