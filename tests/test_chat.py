"""Chat handlers end to end on a temp database (no models, no network)."""

import pytest

from app import db
from app.chat import Assistant


@pytest.fixture
def assistant(tmp_path):
    path = tmp_path / "games.db"
    conn = db.connect(path)
    db.upsert_games(conn, [{"id": i, "slug": s, "name": n} for i, s, n in
                           [(1, "hk", "Hollow Knight"), (2, "celeste", "Celeste"),
                            (3, "fifa", "FIFA 23"), (4, "sv", "Stardew Valley")]])
    conn.close()
    return Assistant(db_path=path, load_models=False)


def test_like_dislike_and_history(assistant):
    r = assistant.handle("I loved Hollow Knight and Celeste")
    assert r.intent == "like" and "Hollow Knight, Celeste" in r.text and r.game == "Hollow Knight"
    assistant.handle("i hated FIFA 23")
    h = assistant.handle("show my list").text
    assert "**Liked:** Celeste, Hollow Knight" in h and "**Disliked:** FIFA 23" in h


def test_lore_for_game_without_wiki_is_declined(assistant):
    r = assistant.handle("what's the lore of Stardew Valley?")
    assert r.intent == "lore" and "don't have lore for **Stardew Valley**" in r.text


def test_recommend_without_history_asks_for_likes(assistant, monkeypatch):
    import app.recommend as rec
    monkeypatch.setattr(rec, "recommend", lambda *a, **k: (_ for _ in ()).throw(ValueError("x")))
    assert "Tell me a few games you liked" in assistant.handle("what should I play next?").text


def test_unknown_message_gets_help(assistant):
    r = assistant.handle("hello")
    assert r.intent == "help" and "Recommendations" in r.text
