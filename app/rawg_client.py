"""Minimal RAWG API client with pagination, retries, and rate limiting.

Docs: https://api.rawg.io/docs/
"""

from collections.abc import Iterator

import httpx

from app import config
from app.http_retry import ThrottledHttp


class RawgClient:
    # The list endpoint has no descriptions -> fetch.py calls get_description per game.
    needs_detail_fetch = True

    def __init__(self, api_key: str = config.RAWG_API_KEY, *,
                 min_interval: float = config.RAWG_MIN_INTERVAL_S,
                 max_retries: int = config.RAWG_MAX_RETRIES,
                 transport: httpx.BaseTransport | None = None):
        if not api_key:
            raise RuntimeError("RAWG_API_KEY is not set (see .env.example)")
        self.api_key = api_key
        self._http = ThrottledHttp(base_url=config.RAWG_BASE_URL, min_interval=min_interval,
                                   max_retries=max_retries, transport=transport)

    def close(self):
        self._http.close()

    def _get(self, path: str, params: dict | None = None) -> dict:
        return self._http.request("GET", path, params={**(params or {}), "key": self.api_key}).json()

    def iter_popular_games(self, limit: int = config.FETCH_DEFAULT_LIMIT) -> Iterator[dict]:
        """Yield up to `limit` games ordered by popularity (`-added` = most
        users have added it to their library)."""
        page, yielded = 1, 0
        while yielded < limit:
            data = self._get("/games", {"ordering": "-added", "page": page,
                                        "page_size": config.RAWG_PAGE_SIZE})
            for raw in data.get("results", []):
                yield parse_game(raw)
                yielded += 1
                if yielded >= limit:
                    return
            if not data.get("next"):
                return
            page += 1

    def get_description(self, game_id: int) -> str:
        data = self._get(f"/games/{game_id}")
        return data.get("description_raw") or ""


def parse_game(raw: dict) -> dict:
    """Flatten a RAWG game object into the shape db.upsert_games expects."""
    return {
        "id": raw["id"],
        "slug": raw["slug"],
        "name": raw["name"],
        "released": raw.get("released"),
        "rating": raw.get("rating"),
        "ratings_count": raw.get("ratings_count"),
        "playtime": raw.get("playtime"),
        "added": raw.get("added"),
        "genres": [g["name"] for g in raw.get("genres") or []],
        # RAWG mixes in non-English tags; keep English only.
        "tags": [t["name"] for t in raw.get("tags") or [] if t.get("language", "eng") == "eng"],
        "platforms": [p["platform"]["name"] for p in raw.get("platforms") or []],
        "description": raw.get("description_raw"),
    }
