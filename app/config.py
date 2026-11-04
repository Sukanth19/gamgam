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
CHAT_MODEL = "gemini-2.5-flash"  # used from phase 2 onwards

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
PROFILE_MAX_TAGS = 15
PROFILE_MAX_DESC_CHARS = 1500
