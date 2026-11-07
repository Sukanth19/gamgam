"""The chat brain: route a message, run the matching feature, return a Reply.

No UI code here - the Streamlit app only renders Replies. That keeps this
logic testable without a browser, and lets a CLI/other UI reuse it later.
"""

from dataclasses import dataclass, field

from app import db, llm
from app.router import LORE_GAMES, GameMatcher, LoreEntityIndex, Route, route

HELP = """I can help with:
- **Recommendations**: "what should I play next?" (based on games you've liked)
- **Similar games**: "games like Hollow Knight"
- **Your taste**: "I loved Celeste", "I hated FIFA 23", "show my list"
- **Lore**: "who is the Pale King?", "why did the Nomai come here?" — """ \
    + ", ".join(LORE_GAMES)


@dataclass
class Reply:
    text: str                                         # markdown
    intent: str = "help"
    game: str | None = None                           # becomes the conversation's "last game"
    sources: list[dict] = field(default_factory=list)  # lore citations
    via: str = ""


class Assistant:
    """Holds the expensive things (embedding models, vector stores, Gemini
    client) so they're built once. The SQLite connection is opened per message:
    it's cheap, and SQLite connections shouldn't be shared across threads
    (Streamlit runs each rerun in a thread)."""

    def __init__(self, explain: bool = True, db_path=None, load_models: bool = True):
        from app import config
        self.db_path = db_path or config.DB_PATH
        conn = db.connect(self.db_path)
        try:
            self.matcher = GameMatcher([g["name"] for g in db.get_all_games(conn)])
            self.entities = LoreEntityIndex(db.wiki_titles_by_game(conn))
        finally:
            conn.close()
        self.explain = explain
        self.games = self.lore = self.query_embedder = self.client = None
        if load_models:                      # tests skip this and plug in fakes
            from app.embeddings import make_embedder
            from app.vector_store import GameVectorStore, LoreVectorStore
            self.games = GameVectorStore()
            self.lore = LoreVectorStore()
            self.query_embedder = make_embedder("query")
            self.client = llm.make_client()

    def handle(self, text: str, last_game: str | None = None) -> Reply:
        from app.retrieval import guess_game_by_keywords
        conn = db.connect(self.db_path)
        try:
            r = route(text, self.matcher, last_game, self.client, self.entities,
                      keyword_vote=lambda q: guess_game_by_keywords(conn, q))
            handler = {"recommend": self._recommend, "similar": self._similar,
                       "lore": self._lore, "like": self._verdict, "dislike": self._verdict,
                       "history": self._history}.get(r.intent, self._help)
            reply = handler(conn, text, r)
            reply.intent, reply.via = r.intent, r.via
            return reply
        finally:
            conn.close()

    # --- handlers -----------------------------------------------------------
    def _help(self, conn, text, r: Route) -> Reply:
        return Reply(HELP)

    def _history(self, conn, text, r: Route) -> Reply:
        rows = db.get_history(conn)
        if not rows:
            return Reply("You haven't rated any games yet. Try *\"I loved Hollow Knight\"*.")
        liked = [x["name"] for x in rows if x["verdict"] == "liked"]
        disliked = [x["name"] for x in rows if x["verdict"] == "disliked"]
        parts = []
        if liked:
            parts.append("**Liked:** " + ", ".join(liked))
        if disliked:
            parts.append("**Disliked:** " + ", ".join(disliked))
        return Reply("\n\n".join(parts))

    def _record(self, conn, names: list[str], verdict: str) -> list[str]:
        saved = []
        for name in names:
            game = db.find_game_by_name(conn, name)
            if game:
                db.set_verdict(conn, game["id"], verdict)
                saved.append(game["name"])
        return saved

    def _verdict(self, conn, text, r: Route) -> Reply:
        verdict = "liked" if r.intent == "like" else "disliked"
        saved = self._record(conn, r.games, verdict)
        if not saved:
            return Reply("Which game? I couldn't find that title in my database.")
        return Reply(f"Noted — you {verdict} **{', '.join(saved)}**. "
                     "Ask *\"what should I play next?\"* when you're ready.", game=saved[0])

    def _recommend(self, conn, text, r: Route) -> Reply:
        from app.recommend import recommend
        prefix = ""
        if r.record and r.games:
            saved = self._record(conn, r.games, r.record)
            if saved:
                prefix = f"Noted — you {r.record} **{', '.join(saved)}**.\n\n"
        try:
            picks, history = recommend(conn, self.games, k=8)
        except ValueError:
            return Reply(prefix + "Tell me a few games you liked first, e.g. *\"I loved Hollow Knight\"*.")
        explanations = {}
        if self.explain and self.client is not None:
            from app.explain import explain_picks
            explanations = explain_picks(conn, picks, history, client=self.client)
        lines = [prefix + "Here's what I'd play next:\n"]
        for i, p in enumerate(picks, 1):
            lines.append(f"{i}. **{p['name']}** ({p['year'] or '?'}) — *because you liked {p['because']}*")
            if p["id"] in explanations:
                lines.append(f"   {explanations[p['id']]}")
        return Reply("\n".join(lines))

    def _similar(self, conn, text, r: Route) -> Reply:
        from app.similar import find_similar
        if not r.games:
            return Reply("Similar to which game?")
        try:
            game, hits = find_similar(conn, self.games, r.games[0], k=8)
        except LookupError as e:
            return Reply(f"Sorry — {e}.")
        lines = [f"Games similar to **{game['name']}**:\n"]
        lines += [f"{i}. **{h['name']}** ({h['year'] or '?'}) — {h['genres']}"
                  for i, h in enumerate(hits, 1)]
        return Reply("\n".join(lines), game=game["name"])

    def _lore(self, conn, text, r: Route) -> Reply:
        from app.ask import ask
        lore_game = next((g for g in r.games if g in LORE_GAMES), None)
        if lore_game is None:
            asked = f" for **{r.games[0]}**" if r.games else ""
            return Reply(f"I don't have lore{asked} yet. I know: {', '.join(LORE_GAMES)}.",
                         game=r.game)
        game = db.find_game_by_name(conn, lore_game)
        res = ask(conn, self.lore, self.query_embedder, self.client, text, game)
        sources = [{"title": s["title"], "section": s["section"], "url": s["url"],
                    "spoiler": bool(s["spoiler"] or s["page_spoiler"])} for s in res.sources]
        return Reply(f"*({lore_game})*\n\n{res.answer}", game=lore_game, sources=sources)
