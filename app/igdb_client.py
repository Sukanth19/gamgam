"""IGDB API client (https://api-docs.igdb.com).

Auth: IGDB is owned by Twitch. You register a free Twitch developer app, then
trade its client id + secret for an app access token (OAuth "client
credentials" flow). The token lasts ~60 days; we simply request one per run.

Queries are POSTed in IGDB's own query language ("Apicalypse"), e.g.
    fields name, genres.name; where total_rating_count > 0; limit 500;
`genres.name` expands a reference inline, like a SQL join.
"""

from collections.abc import Iterator
from datetime import datetime, timezone

import httpx

from app import config
from app.http_retry import ThrottledHttp

FIELDS = ",".join([
    "name", "slug", "first_release_date", "total_rating", "total_rating_count",
    "summary", "storyline", "genres.name", "themes.name", "player_perspectives.name",
    "keywords.name", "platforms.name",
])


class IgdbClient:
    # Descriptions come back in the list query, no per-game detail calls.
    needs_detail_fetch = False

    def __init__(self, client_id: str = config.TWITCH_CLIENT_ID,
                 client_secret: str = config.TWITCH_CLIENT_SECRET, *,
                 min_interval: float = config.IGDB_MIN_INTERVAL_S,
                 max_retries: int = config.IGDB_MAX_RETRIES,
                 transport: httpx.BaseTransport | None = None):
        if not client_id or not client_secret:
            raise RuntimeError("TWITCH_CLIENT_ID / TWITCH_CLIENT_SECRET are not set (see .env.example)")
        auth = ThrottledHttp(base_url=config.TWITCH_AUTH_BASE_URL, min_interval=0,
                             max_retries=max_retries, transport=transport)
        try:
            # Credentials go in the form body, not the URL, so they never show
            # up in logs or error messages.
            token = auth.request("POST", "/oauth2/token", data={
                "client_id": client_id, "client_secret": client_secret,
                "grant_type": "client_credentials"}).json()["access_token"]
        finally:
            auth.close()
        self._http = ThrottledHttp(
            base_url=config.IGDB_BASE_URL, min_interval=min_interval, max_retries=max_retries,
            transport=transport,
            headers={"Client-ID": client_id, "Authorization": f"Bearer {token}"})

    def close(self):
        self._http.close()

    def query(self, endpoint: str, body: str) -> list[dict]:
        return self._http.request("POST", f"/{endpoint}", content=body).json()

    def iter_popular_games(self, limit: int = config.FETCH_DEFAULT_LIMIT) -> Iterator[dict]:
        """Yield up to `limit` games, most-rated first (number of ratings is a
        good popularity proxy). game_type 0/8/9 = main games, remakes,
        remasters; this skips DLCs, bundles, mods, etc."""
        offset = 0
        while offset < limit:
            n = min(config.IGDB_PAGE_SIZE, limit - offset)
            rows = self.query("games", (
                f"fields {FIELDS};"
                " where total_rating_count > 0 & game_type = (0,8,9);"
                f" sort total_rating_count desc; limit {n}; offset {offset};"))
            yield from (parse_game(r) for r in rows)
            if len(rows) < n:
                return
            offset += n


def _names(raw: dict, key: str) -> list[str]:
    return [x["name"] for x in raw.get(key) or [] if "name" in x]


def parse_game(raw: dict) -> dict:
    """Map an IGDB game onto the same shape RAWG games use, so db/embeddings
    don't care where the data came from."""
    ts = raw.get("first_release_date")
    released = datetime.fromtimestamp(ts, timezone.utc).date().isoformat() if ts else None
    rating = raw.get("total_rating")          # IGDB: 0-100
    desc = "\n\n".join(p for p in (raw.get("summary"), raw.get("storyline")) if p)
    return {
        "id": raw["id"],
        "slug": raw["slug"],
        "name": raw["name"],
        "released": released,
        "rating": round(rating / 20, 2) if rating is not None else None,  # -> 0-5 like RAWG
        "ratings_count": raw.get("total_rating_count"),
        "playtime": None,                     # not in IGDB's games endpoint
        "added": raw.get("total_rating_count"),  # our popularity sort key
        "genres": _names(raw, "genres"),
        # themes ("Horror", "Open world") and perspectives first: they're curated;
        # keywords are user-submitted and noisier, so they go last.
        "tags": _names(raw, "themes") + _names(raw, "player_perspectives") + _names(raw, "keywords"),
        "platforms": _names(raw, "platforms"),
        "description": desc or None,
    }
