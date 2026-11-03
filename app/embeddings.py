"""Text profiles for games + batched Gemini embeddings with 429 backoff."""

import hashlib
import html
import json
import logging
import math
import random
import re
import time

from app import config

log = logging.getLogger(__name__)

TASK_TYPE = "SEMANTIC_SIMILARITY"   # comparing game profile <-> game profile

# Store-feature tags present on huge numbers of games. They say nothing about
# what the game *is*, and would make unrelated games look similar.
NOISE_TAGS = {
    "singleplayer", "multiplayer", "steam achievements", "full controller support",
    "steam cloud", "steam trading cards", "steam leaderboards", "steam workshop",
    "partial controller support", "controller support", "achievements", "cloud saves",
    "stats", "in-app purchases", "includes level editor", "steam-trading-cards",
    "remote play together", "remote play on tv", "steamvr collectibles", "valve anti-cheat enabled",
    "captions available", "commentary available", "great soundtrack", "atmospheric",
}

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
# RAWG often appends a Spanish translation after a line containing just "Español".
_SPANISH_RE = re.compile(r"(^|\n)\s*Español\s*(\n|$)")


def clean_description(text: str | None, max_chars: int = config.PROFILE_MAX_DESC_CHARS) -> str:
    if not text:
        return ""
    m = _SPANISH_RE.search(text)
    if m:
        text = text[:m.start()]
    text = html.unescape(_TAG_RE.sub(" ", text))
    text = _WS_RE.sub(" ", text).strip()
    if len(text) > max_chars:
        cut = text.rfind(" ", 0, max_chars)        # don't cut mid-word
        text = text[:cut if cut > 0 else max_chars].rstrip(" ,.;:") + "..."
    return text


def build_profile(game: dict) -> str:
    """One text blob per game; this is what actually gets embedded."""
    parts = [f"{game['name']}."]
    if game.get("genres"):
        parts.append("Genres: " + ", ".join(game["genres"]) + ".")
    tags = [t for t in game.get("tags") or [] if t.lower() not in NOISE_TAGS]
    tags = tags[:config.PROFILE_MAX_TAGS]
    if tags:
        parts.append("Tags: " + ", ".join(tags) + ".")
    desc = clean_description(game.get("description"))
    if desc:
        parts.append(desc)
    return "\n".join(parts)


def profile_hash(profile: str) -> str:
    """Cache key: if the profile text, model or settings change, re-embed."""
    key = f"{config.EMBEDDING_MODEL}|{config.EMBEDDING_DIM}|{TASK_TYPE}|{profile}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def normalize(vec: list[float]) -> list[float]:
    """Truncated (non-3072) gemini-embedding-001 vectors aren't unit length."""
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


def _retry_delay_from_error(err) -> float | None:
    """Gemini 429 responses often include RetryInfo, e.g. "retryDelay": "37s"."""
    m = re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', json.dumps(getattr(err, "details", None) or {}))
    return float(m.group(1)) if m else None


class Embedder:
    def __init__(self, client=None, *, batch_size: int = config.EMBED_BATCH_SIZE,
                 min_interval: float = config.EMBED_MIN_INTERVAL_S,
                 max_retries: int = config.EMBED_MAX_RETRIES):
        if client is None:
            from google import genai
            if not config.GEMINI_API_KEY:
                raise RuntimeError("GEMINI_API_KEY is not set (see .env.example)")
            client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.client = client
        self.batch_size = batch_size
        self.min_interval = min_interval
        self.max_retries = max_retries
        self._last = 0.0

    def _embed_once(self, texts: list[str]) -> list[list[float]]:
        from google.genai import errors, types
        cfg = types.EmbedContentConfig(task_type=TASK_TYPE,
                                       output_dimensionality=config.EMBEDDING_DIM)
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                resp = self.client.models.embed_content(
                    model=config.EMBEDDING_MODEL, contents=texts, config=cfg)
                return [normalize(e.values) for e in resp.embeddings]
            except errors.APIError as e:
                if e.code != 429 and (e.code or 0) < 500:
                    raise                      # bad request / auth: don't retry
                if attempt == self.max_retries:
                    raise
                delay = _retry_delay_from_error(e) or min(90, 2 ** (attempt + 1))
                delay += random.uniform(0, 1)
                log.warning("Gemini %s, retry %d/%d in %.0fs", e.code,
                            attempt + 1, self.max_retries, delay)
                time.sleep(delay)
        raise AssertionError("unreachable")

    def embed(self, texts: list[str]):
        """Yield (start_index, vectors) per batch so callers can save progress
        incrementally - a crash mid-run loses at most one batch."""
        for start in range(0, len(texts), self.batch_size):
            yield start, self._embed_once(texts[start:start + self.batch_size])
