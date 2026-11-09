"""Spoiler control: progress parsing, tagging validation, filtering in ask()."""

import pytest

from app import db, spoilers
from app.ask import LoreAnswer, ask
from app.embeddings import normalize
from app.progress import MILESTONES, finished_index, parse_progress
from app.vector_store import LoreVectorStore
from tests.test_ask import FakeClient, FakeEmbedder


def test_every_lore_game_has_ordered_milestones():
    from app.wikis import WIKIS
    for w in WIKIS:
        assert len(MILESTONES[w["game"]]) >= 5, w["game"]


@pytest.mark.parametrize("game,text,expected", [
    ("Hollow Knight", "I just reached City of Tears", 2),
    ("Hollow Knight", "I'm in Deepnest now", 4),
    ("Hollow Knight", "I finished it", finished_index("Hollow Knight")),
    ("Portal 2", "I'm on chapter 6", 4),
    ("Outer Wilds", "no spoilers past the quantum moon", 6),
    ("The Witcher 3: Wild Hunt", "just got to Novigrad", 2),
    ("Hollow Knight", "somewhere in the middle", None),
])
def test_parse_progress(game, text, expected):
    assert parse_progress(game, text) == expected


def test_tag_batch_validates_and_clamps_model_output(monkeypatch):
    from app import llm
    pages = [{"page_key": "w:1", "title": "Pale King", "revid": 5,
              "sections": [{"section": "(intro)", "raw_section": "", "sample": "x"},
                           {"section": "Lore", "raw_section": "Lore", "sample": "y"}]}]
    monkeypatch.setattr(llm, "generate_json", lambda *a, **k: [
        spoilers.SectionTag(page=0, section=0, milestone=1),
        spoilers.SectionTag(page=0, section=1, milestone=99),   # out of range -> clamped to last
        spoilers.SectionTag(page=3, section=0, milestone=1),    # unknown page -> ignored
    ])
    rows = spoilers.tag_batch(object(), "Hollow Knight", pages)
    last = len(MILESTONES["Hollow Knight"]) - 1
    assert rows == [("w:1", "", 1, 5), ("w:1", "Lore", last, 5)]


# --- filtering in ask() -------------------------------------------------------
GAME = {"id": 1, "name": "Hollow Knight"}


# page_key -> (title, [(section, text, milestone or None if untagged, vector)])
PAGES = {
    "w:1": ("Pale King", [("", "The Pale King ruled Hallownest.", 0, [1, 0.1, 0]),
                          ("Fate", "The Pale King sealed the Radiance using the Vessel.", 6, [1, 0, 0])]),
    "w:2": ("Hornet", [("", "Hornet guards Greenpath.", None, [0, 0, 1])]),
}


@pytest.fixture
def env():
    conn = db.connect(":memory:")
    db.upsert_games(conn, [{"id": 1, "slug": "hk", "name": "Hollow Knight"}])
    vectors = {}
    for key, (title, sections) in PAGES.items():
        chunks = []
        for i, (sec, text, milestone, vec) in enumerate(sections):
            chunks.append({"chunk_id": f"{key}#{i}", "position": i, "section": sec,
                           "kind": "text", "text": text})
            vectors[f"{key}#{i}"] = normalize(vec)
            if milestone is not None:
                db.save_section_milestones(conn, [(key, sec, milestone, 1)])
        db.save_page(conn, {"page_key": key, "game_id": 1, "wiki": "w", "title": title,
                            "url": "u", "revid": 1}, chunks)
    store = LoreVectorStore(path=None, collection=f"test_spoil_{id(conn)}")
    chunks = db.get_lore_chunks(conn)
    store.upsert(chunks, [vectors[c["chunk_id"]] for c in chunks], ["h"] * len(chunks))
    return conn, store


def answer_citing_all(n=3):
    return LoreAnswer(answerable=True, citations=list(range(1, n + 1)),
                      answer=" ".join(f"Fact [{i}]." for i in range(1, n + 1)))


def test_protection_off_sees_everything(env):
    conn, store = env
    client = FakeClient(answer_citing_all())
    ask(conn, store, FakeEmbedder([1, 0, 0]), client, "what did the pale king do", GAME)
    assert "sealed the Radiance" in client.prompt


def test_protection_hides_later_and_untagged_sections(env):
    conn, store = env
    client = FakeClient(answer_citing_all(1))
    ask(conn, store, FakeEmbedder([1, 0.05, 0]), client, "who is the pale king", GAME, max_milestone=2)
    assert "ruled Hallownest" in client.prompt
    assert "sealed the Radiance" not in client.prompt          # milestone 6 > 2
    assert "Hornet guards" not in client.prompt                # untagged -> hidden


def test_wiki_spoiler_marks_override_early_tags():
    from app.ask import effective_milestone
    assert effective_milestone({"milestone": 1, "spoiler": 1, "page_spoiler": 0}, finished=5) == 5
    assert effective_milestone({"milestone": 6, "spoiler": 0, "page_spoiler": 1}, finished=5) == 6
    assert effective_milestone({"milestone": 1, "spoiler": 0, "page_spoiler": 0}, finished=5) == 1
    assert effective_milestone({"milestone": None, "spoiler": 1, "page_spoiler": 0}, finished=5) is None


def test_much_better_hidden_match_skips_llm(env):
    conn, store = env
    client = FakeClient(answer_citing_all(1))
    # hidden "Fate" scores 0.857, visible intro 0.802: gap 0.055 >= margin 0.05
    res = ask(conn, store, FakeEmbedder([1, -0.6, 0]), client, "how was the radiance sealed", GAME,
              max_milestone=2)
    assert res.reason == "revealed later"
    assert client.calls == 0                                   # weak visible text never reached the LLM


def test_hidden_answer_says_revealed_later_not_idk(env):
    conn, store = env
    client = FakeClient(LoreAnswer(answerable=False, citations=[], answer="no"))
    res = ask(conn, store, FakeEmbedder([1, 0, 0]), client, "how was the radiance sealed", GAME,
              max_milestone=2)
    assert res.reason == "revealed later" and res.revealed_at == 6
    assert "Abyss & White Palace" in res.answer or "Endings" in res.answer or "revealed later" in res.answer
    assert "sealed the Radiance" not in res.answer             # never leak the hidden text
