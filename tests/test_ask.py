"""ask(): gating, citation checking and renumbering, with fakes (no network)."""

import pytest

from app import ask as ask_mod
from app import db, llm
from app.ask import LoreAnswer, ask, check_citations
from app.embeddings import normalize
from app.vector_store import LoreVectorStore

GAME = {"id": 1, "name": "Hollow Knight"}
TEXTS = ["The Pale King ruled Hallownest from the White Palace.",
         "The Pale King was a Wyrm who shed his shell in Kingdom's Edge.",
         "Hornet is the daughter of the Pale King and Herrah."]
VECS = [[1, 0, 0], [0.9, 0.3, 0], [0.5, 0.5, 0.5]]


class FakeEmbedder:
    def __init__(self, vec): self.vec = vec
    def embed(self, texts): yield 0, [normalize(self.vec)]


class FakeClient:
    def __init__(self, answer=None):
        self.answer, self.calls = answer, 0
        outer = self
        class Models:
            def generate_content(self, model, contents, config):
                outer.calls += 1
                outer.prompt = contents
                class R: parsed = outer.answer
                return R()
        self.models = Models()


@pytest.fixture
def env():
    conn = db.connect(":memory:")
    db.upsert_games(conn, [{"id": 1, "slug": "hk", "name": "Hollow Knight"}])
    chunks = []
    for i, t in enumerate(TEXTS):
        key = f"w:{i}"
        db.save_page(conn, {"page_key": key, "game_id": 1, "wiki": "w", "title": f"Page {i}",
                            "url": f"https://wiki/{i}", "revid": 1},
                     [{"chunk_id": f"{key}#0", "position": 0, "section": "Lore", "kind": "text", "text": t}])
    store = LoreVectorStore(path=None, collection=f"test_ask_{id(conn)}")
    chunks = db.get_lore_chunks(conn)
    store.upsert(chunks, [normalize(v) for v in VECS], ["h"] * len(chunks))
    return conn, store


def test_weak_match_says_idk_without_calling_llm(env):
    conn, store = env
    client = FakeClient()
    res = ask(conn, store, FakeEmbedder([-1, 0, 0]), client, "favourite pizza?", GAME)
    assert not res.answerable and res.answer == ask_mod.IDK
    assert client.calls == 0                                  # no API call spent


def test_answer_citations_are_checked_and_renumbered(env):
    conn, store = env
    client = FakeClient(LoreAnswer(answerable=True, citations=[2, 1, 99],
                                   answer="He ruled from the White Palace [2]. He was a Wyrm [1] [99]."))
    res = ask(conn, store, FakeEmbedder([1, 0.05, 0]), client, "who is the pale king", GAME)
    assert res.answerable
    assert "[99]" not in res.answer                           # invented source number removed
    assert res.answer.startswith("He ruled from the White Palace [1].")   # renumbered by first use
    assert [s["title"] for s in res.sources][:2] == [s["title"] for s in res.hits[1::-1]]
    assert "Use ONLY the numbered sources" in client.prompt


def test_answer_without_valid_citations_is_rejected(env):
    conn, store = env
    client = FakeClient(LoreAnswer(answerable=True, citations=[42], answer="The Pale King loved soup [42]."))
    res = ask(conn, store, FakeEmbedder([1, 0, 0]), client, "favourite food", GAME)
    assert not res.answerable and res.reason == "answer had no valid citations"


def test_model_says_not_answerable(env):
    conn, store = env
    client = FakeClient(LoreAnswer(answerable=False, citations=[], answer="Not in sources."))
    res = ask(conn, store, FakeEmbedder([1, 0, 0]), client, "favourite food", GAME)
    assert res.answer == ask_mod.IDK


def test_llm_down_still_shows_sources(env, monkeypatch):
    conn, store = env
    monkeypatch.setattr(llm, "generate_json", lambda *a, **k: None)
    res = ask(conn, store, FakeEmbedder([1, 0, 0]), FakeClient(), "who is the pale king", GAME)
    assert not res.answerable and len(res.sources) == 3


def test_combined_citations_are_split_checked_and_renumbered():
    text, used = check_citations("Because of the Eye [3, 1; 9].", [], n_sources=3)
    assert text == "Because of the Eye [3][1]." and used == [1, 3]


def test_check_citations_cleans_spacing():
    text, used = check_citations("A fact [1]. Made up [7].", [1, 7], n_sources=3)
    assert text == "A fact [1]. Made up." and used == [1]
