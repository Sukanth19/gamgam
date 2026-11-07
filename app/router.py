"""Intent router: decide what a chat message wants, and which game it's about.

Two layers:
1. Rules (free, instant): find game titles in the text, look for phrases like
   "similar to", "recommend", "I loved", "who is". Handles most messages.
2. Gemini (only when the rules can't decide): returns {intent, game} as JSON.

Intents: recommend | similar | lore | like | dislike | history | progress | help
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel

from app import llm
from app.wikis import WIKIS

Intent = Literal["recommend", "similar", "lore", "like", "dislike", "history", "progress", "help"]

# Short names people actually type for the lore games.
LORE_ALIASES = {
    "hollow knight": "Hollow Knight", "hk": "Hollow Knight",
    "outer wilds": "Outer Wilds",
    "witcher 3": "The Witcher 3: Wild Hunt", "witcher": "The Witcher 3: Wild Hunt",
    "wild hunt": "The Witcher 3: Wild Hunt",
    "portal 2": "Portal 2",
}
LORE_GAMES = [w["game"] for w in WIKIS]


@dataclass
class Route:
    intent: str
    games: list[str] = field(default_factory=list)   # canonical names, in order of mention
    via: str = "rules"                                # "rules" | "llm" | "fallback"
    record: str | None = None                         # "liked"/"disliked" to save before acting

    @property
    def game(self) -> str | None:
        return self.games[0] if self.games else None


class GameMatcher:
    """Finds game titles inside free text.

    Many titles are ordinary words ("Inside", "Journey", "Control", "Prey"), so
    single-word titles only match when written with the title's capitalisation;
    multi-word titles match case-insensitively. Longest match wins, so "Portal 2"
    beats "Portal" and "Hollow Knight: Silksong" beats "Hollow Knight".
    """

    def __init__(self, names: list[str]):
        self.names = {}
        for n in names:
            key = self._norm(n)
            if len(key) >= 4 or n in LORE_GAMES:
                self.names.setdefault(key, n)
        for alias, canonical in LORE_ALIASES.items():
            self.names[alias] = canonical
        # longest first so the regex prefers "portal 2" over "portal"
        keys = sorted(self.names, key=len, reverse=True)
        self._re = re.compile(r"(?<![\w])(" + "|".join(re.escape(k) for k in keys) + r")(?![\w])")

    @staticmethod
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[:\-–—'’]", " ", s.lower())).strip()

    def find(self, text: str) -> list[str]:
        norm_text = self._norm(text)
        found = []
        for m in self._re.finditer(norm_text):
            key = m.group(1)
            name = self.names[key]
            if " " not in key and key not in LORE_ALIASES:
                # single-word title: require the original capitalisation in the raw text
                if not re.search(r"(?<![\w])" + re.escape(name) + r"(?![\w])", text):
                    continue
            if name not in found:
                found.append(name)
        return found


class LoreEntityIndex:
    """Wiki page titles -> which game they belong to ("Nomai" -> Outer Wilds,
    "White Lady" -> Hollow Knight). Lets "why did the Nomai leave?" find its
    game even when the user doesn't name it."""

    def __init__(self, titles_by_game: dict[str, list[str]]):
        self.title_game: dict[str, str] = {}
        ambiguous = set()
        for game, titles in titles_by_game.items():
            for t in titles:
                key = GameMatcher._norm(re.sub(r"\s*\(.*?\)\s*", " ", t))
                key = re.sub(r"^the ", "", key)          # "The White Lady" == "White Lady"
                if len(key) < 4:
                    continue
                if key in self.title_game and self.title_game[key] != game:
                    ambiguous.add(key)                # same title in two wikis: useless as a clue
                self.title_game[key] = game
        for key in ambiguous:
            del self.title_game[key]
        keys = sorted(self.title_game, key=len, reverse=True)
        self._re = re.compile(r"(?<![\w])(" + "|".join(map(re.escape, keys)) + r")(?![\w])") if keys else None

    def guess_game(self, text: str) -> str | None:
        """The game whose page titles cover the most characters of the text, if any."""
        if self._re is None:
            return None
        score: dict[str, int] = {}
        for m in self._re.finditer(GameMatcher._norm(text)):
            g = self.title_game[m.group(1)]
            score[g] = score.get(g, 0) + len(m.group(1))
        return max(score, key=score.get) if score else None


_RECOMMEND = re.compile(r"\b(recommend|suggest|what should i play|what (game )?(to|should i) play|"
                        r"something (new )?to play|play next|any (good )?games?)\b", re.I)
_SIMILAR = re.compile(r"\b(similar to|games? like|more like|something like|alternatives? to|"
                      r"like .+ but)\b", re.I)
_LIKE = re.compile(r"\b(i (really |absolutely )?(liked|loved|enjoyed|adored)|i (really )?(like|love) |"
                   r"(liked|loved|enjoyed|adored) it)", re.I)
# "how do I beat X" is a question, not a progress report
_QUESTION_START = re.compile(r"^\s*(how|who|what|why|where|when|which|can|could|should|do|does|did|"
                             r"is|are|was|were|will|would)\b", re.I)
_DISLIKE = re.compile(r"\b(i (really )?(disliked|hated|didn'?t (like|enjoy)|don'?t like|couldn'?t stand)|"
                      r"not a fan of)\b", re.I)
_HISTORY = re.compile(r"\b(my (history|list|likes|games)|what (have|did) i (like|liked|play|played)|"
                      r"show (me )?my)\b", re.I)
_PROGRESS = re.compile(r"\b(i'?m (now )?(at|in|on|up to|past)|i am (now )?(at|in|on|up to|past)|"
                       r"i (just )?(reached|got to|beat|finished|completed|started)|"
                       r"i'?ve (just )?(reached|beaten|finished|completed|started)|"
                       r"(no|avoid|hide) spoilers|spoilers? (on|off)|spoiler[- ]free|"
                       r"spoiler protection)\b", re.I)
_LORE = re.compile(r"\b(who|what|why|where|when|how|which|lore|story|explain|tell me about|"
                   r"happened|ending|origin)\b", re.I)


class RouteChoice(BaseModel):
    intent: Intent
    game: str | None = None


def route_with_rules(text: str, games: list[str], last_game: str | None,
                     entities: "LoreEntityIndex | None" = None,
                     keyword_vote: Callable[[str], str | None] | None = None) -> Route | None:
    has_lore_game = any(g in LORE_GAMES for g in games)
    if _HISTORY.search(text):
        return Route("history", games)
    if _PROGRESS.search(text) and not _QUESTION_START.search(text):
        lore = [g for g in games if g in LORE_GAMES]
        if lore:
            return Route("progress", lore)
        if not games and last_game in LORE_GAMES:
            return Route("progress", [last_game])
    if _DISLIKE.search(text) and games:
        return Route("dislike", games)
    if _SIMILAR.search(text) and games:
        return Route("similar", games)
    if _LIKE.search(text) and games:
        # "I loved X, what should I play next?" -> record it AND recommend
        if _RECOMMEND.search(text):
            return Route("recommend", games, record="liked")
        return Route("like", games)
    if _RECOMMEND.search(text):
        return Route("recommend", games)
    if _LORE.search(text):
        if has_lore_game:
            lore = [g for g in games if g in LORE_GAMES]
            return Route("lore", lore + [g for g in games if g not in lore])
        if games:
            # a lore-style question about a game we have no wiki for: still "lore",
            # so the user hears "no lore for X yet" rather than an unrelated answer
            return Route("lore", games)
        if not games:
            # "why did the Nomai leave?" -> a wiki page title points at the game
            guess = entities.guess_game(text) if entities else None
            if guess is None and keyword_vote is not None:
                guess = keyword_vote(text)           # "who is Yennefer" -> BM25 says Witcher
            if guess:
                return Route("lore", [guess])
            if last_game in LORE_GAMES:
                return Route("lore", [last_game])    # follow-up: "and who is his daughter?"
    return None


def route_with_llm(client, text: str, last_game: str | None) -> Route | None:
    prompt = (
        "Classify this message to a video game assistant.\n"
        "intents: recommend (wants suggestions based on their taste), similar (games like a "
        "specific game), lore (question about a game's story/characters/world/gameplay), "
        "like/dislike (telling us their opinion of a game), history (their liked/disliked list), "
        "help (greeting, unclear, or anything else).\n"
        f"Games with lore available: {', '.join(LORE_GAMES)}.\n"
        f"Game discussed previously: {last_game or 'none'}.\n"
        "Set game to the game the message is about (full title), or null.\n\n"
        f"MESSAGE: {text}"
    )
    out = llm.generate_json(client, prompt, RouteChoice, temperature=0.0, what="route")
    if out is None:
        return None
    return Route(out.intent, [out.game] if out.game else [], via="llm")


def route(text: str, matcher: GameMatcher, last_game: str | None = None, client=None,
          entities: LoreEntityIndex | None = None,
          keyword_vote: Callable[[str], str | None] | None = None) -> Route:
    games = matcher.find(text)
    r = route_with_rules(text, games, last_game, entities, keyword_vote)
    if r:
        return r
    if client is not None:
        r = route_with_llm(client, text, last_game)
        if r:
            if r.games:                              # normalise the LLM's title via our matcher
                r.games = matcher.find(r.games[0]) or r.games
            return r
    # No rule matched and no LLM: a bare game title most likely means "tell me about it".
    if games:
        return Route("lore" if games[0] in LORE_GAMES else "similar", games, via="fallback")
    return Route("help", [], via="fallback")
