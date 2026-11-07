"""Recommendation math + explanation step, with hand-made vectors and a fake Gemini."""

import numpy as np
import pytest

from app import db, explain, llm
from app.embeddings import normalize
from app.recommend import mmr, recommend, relevance_scores, taste_vector
from app.vector_store import GameVectorStore


def test_taste_vector_averages_likes_and_pushes_away_from_dislikes():
    v = taste_vector([[1, 0, 0], [0, 1, 0]], [])
    assert np.allclose(v, normalize([1, 1, 0]))
    v2 = taste_vector([[1, 0, 0], [0, 1, 0]], [[0, 1, 0]], beta=0.25)
    assert v2[0] > v2[1] > 0                         # dislike pulled it away from axis 1
    assert np.isclose(np.linalg.norm(v2), 1.0)


def test_taste_vector_needs_a_like():
    with pytest.raises(ValueError):
        taste_vector([], [[1, 0]])


def test_mmr_avoids_near_duplicates():
    q = np.array([1.0, 0.0])
    cands = [{"id": 1, "vector": normalize([1, 0.05])},   # very relevant
             {"id": 2, "vector": normalize([1, 0.06])},   # near-clone of 1
             {"id": 3, "vector": normalize([1, -0.6])}]   # less relevant, different
    rel = np.array([np.dot(c["vector"], q) for c in cands])
    assert [c["id"] for c in mmr(cands, rel, k=2, lam=1.0)] == [1, 2]   # pure relevance
    assert [c["id"] for c in mmr(cands, rel, k=2, lam=0.5)] == [1, 3]   # diversity kicks in


def test_relevance_blend_lets_a_minority_taste_compete():
    liked = np.array([[1.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]])  # 3 platformers + 1 space game
    taste = taste_vector(liked, [])
    cand = np.array([[0.8, 0.6, 0],      # vaguely between both tastes
                     [0, 1.0, 0]])       # clearly a space game
    pure_avg = relevance_scores(cand, taste, liked, None, blend=1.0)
    blended = relevance_scores(cand, taste, liked, None, blend=0.5)
    gap_avg = pure_avg[0] - pure_avg[1]
    gap_blend = blended[0] - blended[1]
    assert gap_avg > 0.5                 # averaging buries the space game...
    assert gap_blend < gap_avg / 2       # ...the blend at least halves the gap;
    # pool expansion + MMR's redundancy penalty do the rest (see end-to-end test)


def test_recommend_gives_minority_like_a_pick():
    conn = db.connect(":memory:")
    games = [(1, "Plat A", [1, 0, 0]), (2, "Plat B", [0.98, 0.2, 0]), (3, "Plat C", [0.98, -0.2, 0]),
             (4, "Space", [0, 0, 1])] + \
            [(10 + i, f"Plat clone {i}", [1, 0.05 * (i - 3), 0.02]) for i in range(8)] + \
            [(30, "Space clone", [0.1, 0, 1])]
    db.upsert_games(conn, [{"id": i, "slug": f"g{i}", "name": n} for i, n, _ in games])
    store = GameVectorStore(path=None, collection="test_minority")
    by_id = {g["id"]: g for g in db.get_all_games(conn)}
    store.upsert([by_id[i] for i, _, _ in games], [normalize(v) for _, _, v in games], ["h"] * len(games))
    for gid in (1, 2, 3, 4):
        db.set_verdict(conn, gid, "liked")
    picks, _ = recommend(conn, store, k=5)
    assert "Space clone" in [p["name"] for p in picks]
    assert next(p for p in picks if p["name"] == "Space clone")["because"] == "Space"


GAMES = [  # id, name, vector (3 "taste" axes: metroidvania, puzzle, sports)
    (1, "Hollow Knight", [1, 0, 0]), (2, "Celeste", [0.8, 0.4, 0]),
    (3, "Ori", [0.95, 0.1, 0]), (4, "Portal", [0.1, 1, 0]),
    (5, "FIFA", [0, 0, 1]), (6, "PES", [0.05, 0, 1]), (7, "Dead Cells", [0.9, 0, 0.2]),
]


@pytest.fixture
def env():
    conn = db.connect(":memory:")
    db.upsert_games(conn, [{"id": i, "slug": n.lower(), "name": n, "genres": ["X"],
                            "description": f"{n} desc"} for i, n, _ in GAMES])
    store = GameVectorStore(path=None, collection=f"test_rec_{id(conn)}")
    games = db.get_all_games(conn)
    by_id = {g["id"]: g for g in games}
    store.upsert([by_id[i] for i, _, _ in GAMES], [normalize(v) for _, _, v in GAMES],
                 ["h"] * len(GAMES))
    db.set_verdict(conn, 1, "liked")
    db.set_verdict(conn, 4, "liked")
    db.set_verdict(conn, 5, "disliked")
    return conn, store


def test_recommend_excludes_played_and_explains_because(env):
    conn, store = env
    picks, history = recommend(conn, store, k=3)
    ids = [p["id"] for p in picks]
    assert not {1, 4, 5} & set(ids)                 # never recommend played games
    assert 6 not in ids                             # PES ~ disliked FIFA: ranked last
    because = {p["name"]: p["because"] for p in picks}
    assert because["Ori"] == "Hollow Knight"


class FakeModels:
    def __init__(self, fail_with=None):
        self.models_tried, self.fail_with = [], fail_with

    def generate_content(self, model, contents, config):
        from google.genai import errors
        self.models_tried.append(model)
        self.prompt = contents
        if self.fail_with:
            raise errors.ServerError(self.fail_with, {"error": {"code": self.fail_with,
                                                                "message": "busy", "status": "UNAVAILABLE"}})
        class R:
            parsed = [explain.Explanation(game_id=3, why=" Fast, tight platforming. "),
                      explain.Explanation(game_id=999, why="hallucinated id")]
        return R()


class FakeClient:
    def __init__(self, **kw): self.models = FakeModels(**kw)


def test_explain_grounds_prompt_and_drops_unknown_ids(env):
    conn, store = env
    picks, history = recommend(conn, store, k=3)
    client = FakeClient()
    out = explain.explain_picks(conn, picks, history, client=client)
    assert out == {3: "Fast, tight platforming."}   # id 999 isn't a pick: dropped
    p = client.models.prompt
    assert "LIKED: Hollow Knight" in p and "DISLIKED: FIFA" in p and "Use ONLY the facts" in p


def test_explain_falls_back_through_models_then_degrades(env, monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    conn, store = env
    picks, history = recommend(conn, store, k=2)
    client = FakeClient(fail_with=503)
    assert explain.explain_picks(conn, picks, history, client=client) == {}
    tried = client.models.models_tried
    assert len(set(tried)) == len(llm.config.CHAT_MODELS)         # every model tried
