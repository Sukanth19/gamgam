"""Ask Gemini to explain each recommendation in one or two sentences.

Key decisions:
- ONE call for all picks (not one per pick): the free tier counts requests.
- Grounded: the prompt contains the facts (profiles) and tells the model to
  use only those, so it doesn't invent features.
- Structured output: we ask for JSON matching a schema, so parsing can't
  break on chatty formatting.
- Never fatal: if every model is busy or the quota is gone, recommendations
  are still printed, just without explanations.
"""

import logging

from pydantic import BaseModel

from app import db, llm
from app.embeddings import clean_description, select_tags

log = logging.getLogger(__name__)


class Explanation(BaseModel):
    game_id: int
    why: str


def _brief(game: dict) -> str:
    tags = ", ".join(select_tags(game.get("tags") or [], max_tags=12))
    desc = clean_description(game.get("description"), max_chars=350)
    return f"{game['name']} | genres: {', '.join(game['genres'])} | tags: {tags} | {desc}"


def build_prompt(picks: list[dict], liked: list[dict], disliked: list[dict]) -> str:
    lines = [
        "You are a game recommendation assistant. For each recommended game, write",
        "1-2 short sentences telling the player why they might enjoy it, connecting",
        "it to the liked game named in 'because'. Use ONLY the facts given below;",
        "do not invent features, plot, or reviews. If a pick shares traits with a",
        "disliked game, you may briefly note the difference.",
        "",
        "LIKED: " + "; ".join(_brief(g) for g in liked),
    ]
    if disliked:
        lines.append("DISLIKED: " + "; ".join(g["name"] for g in disliked))
    lines += ["", "RECOMMENDATIONS:"]
    for p in picks:
        lines.append(f"- game_id={p['id']} because={p['because']} :: {_brief(p['game'])}")
    lines += ["", "Return a JSON list with one {game_id, why} object per recommendation."]
    return "\n".join(lines)


def explain_picks(conn, picks: list[dict], history: list[dict], client=None) -> dict[int, str]:
    """Returns {game_id: explanation}. Empty dict if Gemini is unavailable."""
    if not picks:
        return {}
    client = client or llm.make_client()
    if client is None:
        return {}

    games = {h["game_id"]: db.get_game(conn, h["game_id"]) for h in history}
    liked = [games[h["game_id"]] for h in history if h["verdict"] == "liked"]
    disliked = [games[h["game_id"]] for h in history if h["verdict"] == "disliked"]
    enriched = [{**p, "game": db.get_game(conn, p["id"])} for p in picks]

    result = llm.generate_json(client, build_prompt(enriched, liked, disliked),
                               list[Explanation], what="explanations")
    if result is None:
        print("(Gemini unavailable right now: showing picks without explanations)\n")
        return {}
    valid = {p["id"] for p in picks}
    return {e.game_id: e.why.strip() for e in result if e.game_id in valid}
