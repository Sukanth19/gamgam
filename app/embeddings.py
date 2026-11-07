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

# Tags about the store/platform/marketing, not about what the game *is*
# ("steam achievements", "the game awards 2016", "pax west 2016", ...). Left
# in, they make unrelated games look similar. Patterns, not an exact list,
# because IGDB keywords are user-submitted and come in endless variants.
NOISE_TAG_RE = re.compile(
    r"steam|achievement|trophies|controller|cloud save|trading card|leaderboard|"
    r"game awards|nominee|award|\bpax\b|\be3\b|gamescom|available on|"
    r"digital distribution|downloadable content|\bdlc\b|bink video|greatest hits|"
    r"soundtrack|crowdfunding|kickstarter|greenlight|early access|"
    r"in-app purchase|microtransaction|remote play|anti-cheat|captions|commentary|"
    r"^sequel$|^singleplayer$|^multiplayer$|^stats$|^online$",
    re.IGNORECASE)

# IGDB "player perspectives" describe the camera, not the game. "Bird view /
# Isometric" alone was linking Stardew Valley to Pillars of Eternity.
CAMERA_TAGS = {"first person", "third person", "bird view / isometric", "side view"}

# Tags on more than this share of games are too generic to help ("action").
MAX_TAG_SHARE = 0.30

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


def tag_counts(games: list[dict]) -> dict[str, int]:
    """How many games carry each tag (lowercased). Used to rank tags."""
    counts: dict[str, int] = {}
    for g in games:
        for t in {t.lower() for t in g.get("tags") or []}:
            counts[t] = counts.get(t, 0) + 1
    return counts


def select_tags(tags: list[str], counts: dict[str, int] | None = None,
                n_games: int = 0, max_tags: int = config.PROFILE_MAX_TAGS) -> list[str]:
    """Drop junk, then keep the most *useful* tags.

    Without corpus stats we just keep the original order. With them, a tag is
    useful when several games share it (so it's a real, agreed-on concept, not
    a one-off) but not so many that it's meaningless - so we rank tags
    shared by >=2 games and <=30% of games by how many games share them.
    """
    seen, clean = set(), []
    for t in tags:
        k = t.lower().strip()
        if k and k not in seen and k not in CAMERA_TAGS and not NOISE_TAG_RE.search(k):
            seen.add(k)
            clean.append(t)
    if counts and n_games:
        limit = MAX_TAG_SHARE * n_games
        def score(t):
            c = counts.get(t.lower(), 0)
            return c if 2 <= c <= limit else -1   # one-offs and near-universal tags last
        clean.sort(key=score, reverse=True)       # stable: ties keep original order
    return clean[:max_tags]


def _dedupe_paragraphs(text: str) -> str:
    """IGDB summary and storyline often repeat each other; keep the first copy."""
    norm = lambda s: _WS_RE.sub(" ", s).strip().lower()
    out: list[str] = []
    for para in text.split("\n\n"):
        n = norm(para)
        if not n:
            continue
        if any(n[:80] in norm(p) for p in out):        # already covered by an earlier one
            continue
        # if this paragraph contains an earlier one, it replaces it
        out = [p for p in out if norm(p)[:80] not in n]
        out.append(para)
    return "\n\n".join(out)


def build_profile(game: dict, counts: dict[str, int] | None = None, n_games: int = 0) -> str:
    """One text blob per game; this is what actually gets embedded.
    Pass `counts`/`n_games` (from tag_counts) to rank tags by usefulness."""
    parts = [f"{game['name']}."]
    if game.get("genres"):
        parts.append("Genres: " + ", ".join(game["genres"]) + ".")
    tags = select_tags(game.get("tags") or [], counts, n_games)
    if tags:
        parts.append("Tags: " + ", ".join(tags) + ".")
    desc = clean_description(_dedupe_paragraphs(game.get("description") or ""))
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


class QuotaExhausted(RuntimeError):
    """Raised when Gemini asks for a very long wait (daily quota used up)."""

    def __init__(self, wait_s: float):
        self.wait_s = wait_s
        super().__init__(f"Gemini quota exhausted, server asked to wait {wait_s / 3600:.1f}h")


def _retry_delay_from_error(err) -> float | None:
    """Gemini 429 responses often include RetryInfo, e.g. "retryDelay": "37s"."""
    m = re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', json.dumps(getattr(err, "details", None) or {}))
    return float(m.group(1)) if m else None


class GeminiEmbedder:
    def __init__(self, client=None, *, task_type: str = TASK_TYPE,
                 batch_size: int = config.EMBED_BATCH_SIZE,
                 min_interval: float = config.EMBED_MIN_INTERVAL_S,
                 max_retries: int = config.EMBED_MAX_RETRIES):
        if client is None:
            from google import genai
            if not config.GEMINI_API_KEY:
                raise RuntimeError("GEMINI_API_KEY is not set (see .env.example)")
            client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.client = client
        self.task_type = task_type
        self.batch_size = batch_size
        self.min_interval = min_interval
        self.max_retries = max_retries
        self._last = 0.0

    def _embed_once(self, texts: list[str]) -> list[list[float]]:
        from google.genai import errors, types
        cfg = types.EmbedContentConfig(task_type=self.task_type,
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
                server_delay = _retry_delay_from_error(e)
                if server_delay and server_delay > config.EMBED_MAX_WAIT_S:
                    raise QuotaExhausted(server_delay) from e
                delay = server_delay or min(90, 2 ** (attempt + 1))
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


class LocalEmbedder:
    """Open-source embedding model on CPU via fastembed (ONNX runtime, no
    PyTorch). First run downloads the model once into data/models/."""

    def __init__(self, model_name: str = config.LOCAL_EMBEDDING_MODEL,
                 batch_size: int = config.LOCAL_EMBED_BATCH_SIZE, prefix: str = ""):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model_name, cache_dir=str(config.MODELS_DIR))
        self.batch_size = batch_size
        self.prefix = prefix

    def embed(self, texts: list[str]):
        for start in range(0, len(texts), self.batch_size):
            batch = [self.prefix + t for t in texts[start:start + self.batch_size]]
            yield start, [normalize(v.tolist()) for v in self.model.embed(batch)]


def make_embedder(kind: str = "similarity"):
    """Pick the embedder from EMBED_PROVIDER (.env).

    kind: "similarity" (game <-> game), "document" (lore chunks being stored)
    or "query" (a question searching those chunks). Retrieval models embed
    questions and passages slightly differently, which helps them match a short
    question to a long answer passage.
    """
    if config.EMBED_PROVIDER == "gemini":
        task = {"similarity": TASK_TYPE, "document": "RETRIEVAL_DOCUMENT",
                "query": "RETRIEVAL_QUERY"}[kind]
        return GeminiEmbedder(task_type=task)
    return LocalEmbedder(prefix=config.LOCAL_QUERY_PREFIX if kind == "query" else "")


def chunk_hash(text: str) -> str:
    """Cache key for a lore chunk (document-mode embedding)."""
    key = f"{config.EMBEDDING_MODEL}|{config.EMBEDDING_DIM}|document|{text}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]
