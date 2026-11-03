"""Central settings. Everything tunable lives here, not scattered in modules."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# --- API keys (from .env) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
RAWG_API_KEY = os.getenv("RAWG_API_KEY", "")

# --- Models ---
EMBEDDING_MODEL = "gemini-embedding-001"
# gemini-embedding-001 outputs 3072 dims by default; it is trained so that
# truncated vectors still work well (Matryoshka). 768 is 4x smaller with
# little quality loss for this use case.
EMBEDDING_DIM = 768
CHAT_MODEL = "gemini-2.5-flash"  # used from phase 2 onwards

# --- Paths ---
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "games.db"
CHROMA_DIR = DATA_DIR / "chroma"
CHROMA_COLLECTION = "games"

# --- RAWG ---
RAWG_BASE_URL = "https://api.rawg.io/api"
RAWG_PAGE_SIZE = 40          # RAWG max
RAWG_DEFAULT_LIMIT = 2000
RAWG_MIN_INTERVAL_S = 0.25   # min gap between requests (be polite)
RAWG_MAX_RETRIES = 5

# --- Gemini embedding rate limiting ---
EMBED_BATCH_SIZE = 50        # texts per embed request
EMBED_MIN_INTERVAL_S = 1.0   # min gap between embed requests
EMBED_MAX_RETRIES = 8

# --- Text profile ---
PROFILE_MAX_TAGS = 15
PROFILE_MAX_DESC_CHARS = 1500
