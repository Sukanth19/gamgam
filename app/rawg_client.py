"""Minimal RAWG API client with pagination, retries, and rate limiting.

Docs: https://api.rawg.io/docs/
"""

import logging
import random
import time
from collections.abc import Iterator

import httpx

from app import config

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class RawgClient:
    def __init__(self, api_key: str = config.RAWG_API_KEY, *,
                 min_interval: float = config.RAWG_MIN_INTERVAL_S,
                 max_retries: int = config.RAWG_MAX_RETRIES,
                 transport: httpx.BaseTransport | None = None):
        if not api_key:
            raise RuntimeError("RAWG_API_KEY is not set (see .env.example)")
        self.api_key = api_key
        self.min_interval = min_interval
        self.max_retries = max_retries
        self._last_request = 0.0
        self._http = httpx.Client(base_url=config.RAWG_BASE_URL, timeout=30,
                                  transport=transport,
                                  headers={"User-Agent": "gamgam-learning-project"})

    def close(self):
        self._http.close()

    # --- low level -----------------------------------------------------------
    def _throttle(self):
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _get(self, path: str, params: dict | None = None) -> dict:
        params = {**(params or {}), "key": self.api_key}
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                resp = self._http.get(path, params=params)
            except httpx.TransportError as e:   # timeouts, connection resets...
                err = f"network error: {e}"
            else:
                if resp.status_code < 400:
                    return resp.json()
                if resp.status_code not in RETRYABLE_STATUS:
                    resp.raise_for_status()      # 401/404 etc: retrying won't help
                err = f"HTTP {resp.status_code}"
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    if attempt == self.max_retries:
                        break
                    log.warning("%s on %s, server asked to wait %ss", err, path, retry_after)
                    time.sleep(int(retry_after))
                    continue
            if attempt == self.max_retries:
                break
            delay = min(60, 2 ** attempt) + random.uniform(0, 1)  # exp backoff + jitter
            log.warning("%s on %s, retry %d/%d in %.1fs", err, path,
                        attempt + 1, self.max_retries, delay)
            time.sleep(delay)
        raise RuntimeError(f"RAWG request {path} failed after {self.max_retries} retries: {err}")

    # --- high level ----------------------------------------------------------
    def iter_popular_games(self, limit: int = config.RAWG_DEFAULT_LIMIT) -> Iterator[dict]:
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
        """The list endpoint has no description; the detail endpoint does."""
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
