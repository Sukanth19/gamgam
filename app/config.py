"""Central settings. Everything tunable lives here, not scattered in modules."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# --- API keys (from .env) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
RAWG_API_KEY = os.getenv("RAWG_API_KEY", "")
TWITCH_CLIENT_ID = os.getenv("TWITCH_CLIENT_ID", "")
TWITCH_CLIENT_SECRET = os.getenv("TWITCH_CLIENT_SECRET", "")

# --- Game data source: "igdb" (default) or "rawg" ---
# Each source has its own ids, so each gets its own db + vector collection.
GAME_SOURCE = os.getenv("GAME_SOURCE", "igdb").strip().lower()
if GAME_SOURCE not in ("igdb", "rawg"):
    raise RuntimeError(f"GAME_SOURCE must be 'igdb' or 'rawg', got {GAME_SOURCE!r}")

# --- Models ---
# Embedding provider: "local" (default) runs an open model on your CPU via
# fastembed - free, no rate limits. "gemini" uses gemini-embedding-001 (free
# tier ~1000 texts/day). Vectors from different models live in different
# spaces, so each provider gets its own Chroma collection.
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "local").strip().lower()
if EMBED_PROVIDER not in ("local", "gemini"):
    raise RuntimeError(f"EMBED_PROVIDER must be 'local' or 'gemini', got {EMBED_PROVIDER!r}")
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
LOCAL_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"   # 768 dims, ~210 MB, MIT license
EMBEDDING_MODEL = GEMINI_EMBEDDING_MODEL if EMBED_PROVIDER == "gemini" else LOCAL_EMBEDDING_MODEL
# gemini-embedding-001 outputs 3072 dims by default; it is trained so that
# truncated vectors still work well (Matryoshka). 768 is 4x smaller with
# little quality loss, and matches bge-base's native size.
EMBEDDING_DIM = 768
# Chat models, tried in order: free-tier models often return 503 "high demand",
# so we fall back to the next one instead of failing.
CHAT_MODELS = ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-2.5-flash", "gemini-flash-lite-latest"]
CHAT_MODEL = CHAT_MODELS[0]
CHAT_RETRIES_PER_MODEL = 2

# --- Recommendations ---
DISLIKE_WEIGHT = 0.5       # Rocchio beta: how hard dislikes push the taste vector away
CANDIDATE_POOL = 60        # nearest neighbours of the taste vector fetched before MMR
PER_LIKE_POOL = 20         # + nearest neighbours of EACH liked game (multi-interest)
# relevance = TASTE_BLEND * sim(taste vector) + (1 - TASTE_BLEND) * sim(closest liked game)
#             - DISLIKE_WEIGHT * sim(closest disliked game)
# Pure average (1.0) lets your majority taste outvote the rest; pure max (0.0)
# just returns each liked game's clones. A blend keeps both.
TASTE_BLEND = 0.5
MMR_LAMBDA = 0.7           # 1.0 = pure relevance, lower = more variety

# --- Paths ---
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / f"games_{GAME_SOURCE}.db"
CHROMA_DIR = DATA_DIR / "chroma"
CHROMA_COLLECTION = f"games_{GAME_SOURCE}_{EMBED_PROVIDER}"
MODELS_DIR = DATA_DIR / "models"   # local model downloads (git-ignored with data/)

FETCH_DEFAULT_LIMIT = 2000

# --- RAWG ---
RAWG_BASE_URL = "https://api.rawg.io/api"
RAWG_PAGE_SIZE = 40          # RAWG max
RAWG_MIN_INTERVAL_S = 0.25   # min gap between requests (be polite)
RAWG_MAX_RETRIES = 5

# --- IGDB (via Twitch auth) ---
TWITCH_AUTH_BASE_URL = "https://id.twitch.tv"
IGDB_BASE_URL = "https://api.igdb.com/v4"
IGDB_PAGE_SIZE = 500         # IGDB max
IGDB_MIN_INTERVAL_S = 0.3    # IGDB allows 4 requests/second
IGDB_MAX_RETRIES = 5

LOCAL_EMBED_BATCH_SIZE = 64  # local: no rate limit, batch size just sets progress granularity

# --- Gemini embedding rate limiting ---
EMBED_BATCH_SIZE = 50        # texts per embed request
EMBED_MIN_INTERVAL_S = 1.0   # min gap between embed requests
EMBED_MAX_RETRIES = 8
# If Gemini asks us to wait longer than this, it's the *daily* quota, not the
# per-minute one: stop and let the user re-run later instead of sleeping for hours.
EMBED_MAX_WAIT_S = 300

# --- Text profile ---
PROFILE_MAX_TAGS = 15          # tested 30: no better
# Rank tags by how many games share them? Tested by eye on 8 games: it pushed
# generic mechanics (crafting, mining) ahead of IGDB's curated themes and put
# DayZ/Rust next to Stardew Valley. Off until the phase-7 eval can measure it.
PROFILE_RANK_TAGS = False
PROFILE_MAX_DESC_CHARS = 1500

# --- Wikis / lore (phase 3) ---
WIKI_MIN_INTERVAL_S = 1.0    # ~1 request/second: polite to community wikis
WIKI_MAX_RETRIES = 5
WIKI_BATCH_SIZE = 20         # pages per content request (big pages -> server may split; we follow `continue`)
CHUNK_WORDS = 250            # target chunk size; small = precise retrieval, big = more context
CHUNK_OVERLAP_WORDS = 40     # shared words between neighbouring chunks so facts aren't cut in half
CHUNK_MIN_WORDS = 12         # drop near-empty sections
LORE_COLLECTION = f"lore_{EMBED_PROVIDER}"
# bge models retrieve better when queries (not documents) carry this instruction.
LOCAL_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
