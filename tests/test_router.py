"""Router: ~20 realistic messages -> (intent, game), rules only (no LLM)."""

import pytest

from app.router import GameMatcher, Route, route

NAMES = ["Hollow Knight", "Hollow Knight: Silksong", "Outer Wilds", "The Outer Worlds",
         "The Witcher 3: Wild Hunt", "Portal", "Portal 2", "Inside", "Journey", "Control",
         "Celeste", "Hades", "FIFA 23", "Stardew Valley", "Doom", "Prey"]


@pytest.fixture(scope="module")
def m():
    return GameMatcher(NAMES)


@pytest.mark.parametrize("text,intent,game", [
    ("recommend me something", "recommend", None),
    ("what should I play next?", "recommend", None),
    ("I loved Celeste, what should I play next?", "recommend", "Celeste"),
    ("games like Hollow Knight", "similar", "Hollow Knight"),
    ("anything similar to stardew valley?", "similar", "Stardew Valley"),
    ("I really liked Hades", "like", "Hades"),
    ("i hated fifa 23", "dislike", "FIFA 23"),
    ("not a fan of Doom", "dislike", "Doom"),
    ("show me my list", "history", None),
    ("who is the pale king in hollow knight?", "lore", "Hollow Knight"),
    ("why did the Nomai come to the solar system in outer wilds", "lore", "Outer Wilds"),
    ("who is Caroline in Portal 2", "lore", "Portal 2"),
    ("how do I beat a leshen in witcher 3", "lore", "The Witcher 3: Wild Hunt"),
    ("tell me about hk's ending", "lore", "Hollow Knight"),
])
def test_rules(m, text, intent, game):
    r = route(text, m)
    assert (r.intent, r.game) == (intent, game)
    assert r.via == "rules"


def test_followup_uses_last_lore_game(m):
    assert route("and who is his daughter?", m, last_game="Hollow Knight").game == "Hollow Knight"


def test_wiki_entity_beats_last_game():
    from app.router import LoreEntityIndex
    ents = LoreEntityIndex({"Outer Wilds": ["Nomai", "Solanum", "Map"],
                            "Hollow Knight": ["White Lady", "Hornet", "Map"],
                            "Portal 2": ["GLaDOS", "Hornet (cut)"]})
    m = GameMatcher(NAMES)
    assert route("why did the Nomai leave?", m, last_game="Hollow Knight", entities=ents).game == "Outer Wilds"
    assert route("who is the white lady", m, last_game="Outer Wilds", entities=ents).game == "Hollow Knight"
    # "Hornet" is a title in two wikis -> no clue -> previous game is used
    assert route("who is hornet", m, last_game="Hollow Knight", entities=ents).game == "Hollow Knight"
    assert ents.guess_game("what is the map") is None                   # < 4 chars ignored


def test_keyword_vote_used_when_no_title_matches():
    m = GameMatcher(NAMES)
    vote = lambda text: "The Witcher 3: Wild Hunt" if "yennefer" in text.lower() else None
    assert route("who is Yennefer", m, last_game="Hollow Knight", keyword_vote=vote).game == \
        "The Witcher 3: Wild Hunt"
    assert route("what is love", m, last_game="Hollow Knight", keyword_vote=vote).game == "Hollow Knight"


def test_common_word_titles_need_capitalisation(m):
    assert m.find("who is inside the white palace") == []          # not the game "Inside"
    assert m.find("I loved Inside and Journey") == ["Inside", "Journey"]
    assert m.find("i need more control") == []


def test_longest_title_wins(m):
    assert m.find("is portal 2 better than portal?") == ["Portal 2"]       # lowercase single-word "portal" ignored
    assert m.find("Hollow Knight: Silksong lore") == ["Hollow Knight: Silksong"]
    assert m.find("outer wilds vs the outer worlds") == ["Outer Wilds", "The Outer Worlds"]


def test_no_rule_no_llm_falls_back(m):
    assert route("hello there", m).intent == "help"
    assert route("Outer Wilds", m) == Route("lore", ["Outer Wilds"], via="fallback")
    assert route("Celeste", m).intent == "similar"


def test_llm_used_when_rules_cannot_decide(m, monkeypatch):
    from app import llm
    from app.router import RouteChoice
    monkeypatch.setattr(llm, "generate_json",
                        lambda *a, **k: RouteChoice(intent="lore", game="the witcher 3"))
    r = route("tell me something cool", m, client=object())
    assert (r.intent, r.game, r.via) == ("lore", "The Witcher 3: Wild Hunt", "llm")
