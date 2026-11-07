"""Hybrid lore retrieval: vector search + BM25 keyword search, fused with RRF.

Why both?
- Vector search understands meaning ("why did they leave" ~ "reason for departure")
  but is fuzzy on rare proper nouns ("Ilex", "Dirtmouth").
- BM25 matches exact words, great for names, blind to paraphrase.

Reciprocal Rank Fusion: score(chunk) = sum over lists of 1 / (RRF_K + rank).
It only uses RANKS, so we never have to compare a cosine similarity with a
BM25 score (different scales). Chunks found by both lists rise to the top.
"""

from app import config, db
from app.vector_store import LoreVectorStore


def rrf(ranked_lists: list[list[str]], k: int = config.RRF_K) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores


def hybrid_search(conn, store: LoreVectorStore, query_vec: list[float], question: str,
                  game_id: int | None, k: int = config.ANSWER_CONTEXT_CHUNKS,
                  pool: int = config.RETRIEVE_POOL) -> list[dict]:
    """Top-k chunks (dicts with text/title/section/url + scoring details)."""
    vec_hits = store.search(query_vec, pool, game_id)
    bm25_hits = db.bm25_search(conn, question, pool, game_id)

    vec_ids = [h["chunk_id"] for h in vec_hits]
    bm25_ids = [cid for cid, _ in bm25_hits]
    fused = rrf([vec_ids, bm25_ids])
    top = sorted(fused, key=fused.get, reverse=True)[:k]

    sims = {h["chunk_id"]: h["similarity"] for h in vec_hits}
    chunks = db.get_chunks_by_ids(conn, top)
    out = []
    for cid in top:
        if cid not in chunks:          # in Chroma but no longer in SQLite (stale): skip
            continue
        out.append({**chunks[cid], "rrf": fused[cid], "similarity": sims.get(cid),
                    "vec_rank": vec_ids.index(cid) + 1 if cid in vec_ids else None,
                    "bm25_rank": bm25_ids.index(cid) + 1 if cid in bm25_ids else None})
    return out


def best_similarity(hits: list[dict]) -> float:
    return max((h["similarity"] for h in hits if h["similarity"] is not None), default=0.0)
