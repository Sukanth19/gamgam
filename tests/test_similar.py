"""End-to-end (db -> vector store -> similar) with hand-made vectors, no API."""

from app import db
from app.embed import plan_work
from app.embeddings import build_profile, normalize, profile_hash, tag_counts
from app.similar import find_similar
from app.vector_store import GameVectorStore

GAMES = [
    {"id": 1, "slug": "hk", "name": "Hollow Knight", "released": "2017-02-24", "added": 50},
    {"id": 2, "slug": "ori", "name": "Ori and the Blind Forest", "released": "2015-03-11", "added": 40},
    {"id": 3, "slug": "celeste", "name": "Celeste", "released": "2018-01-25", "added": 30},
    {"id": 4, "slug": "fifa", "name": "FIFA 23", "released": "2022-09-30", "added": 20},
]
VECS = [normalize(v) for v in ([1, 0.1, 0], [0.9, 0.3, 0], [0.6, 0.6, 0.1], [0, 0, 1])]


def setup():
    conn = db.connect(":memory:")
    db.upsert_games(conn, GAMES)
    store = GameVectorStore(path=None, collection="test_similar")
    games = db.get_all_games(conn)
    counts = tag_counts(games)
    store.upsert(games, VECS, [profile_hash(build_profile(g, counts, len(games))) for g in games])
    return conn, store


def test_similar_excludes_self_and_orders_by_score():
    conn, store = setup()
    game, hits = find_similar(conn, store, "hollow knight", k=3)
    assert game["id"] == 1
    assert [h["name"] for h in hits] == ["Ori and the Blind Forest", "Celeste", "FIFA 23"]
    assert hits[0]["similarity"] > hits[1]["similarity"] > hits[2]["similarity"]
    assert hits[0]["year"] == 2015


def test_plan_work_skips_already_embedded_and_detects_changes():
    conn, store = setup()
    games = db.get_all_games(conn)
    assert plan_work(games, store.existing_hashes()) == []
    games[0]["description"] = "now with a description"
    todo = plan_work(games, store.existing_hashes())
    assert [g["id"] for g, _, _ in todo] == [1]
