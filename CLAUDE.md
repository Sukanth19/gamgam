# gamgam — game recommender + game lore q&a

A learning project: a Python RAG app built on the Gemini API (free tier).
The user is learning, so briefly explain the key decisions when changing code.

## Roadmap
1. [built — code + tests done, awaiting first live run] Fetch game metadata from the RAWG API into SQLite, embed it, "similar games" search
2. Play history (liked/disliked) -> taste-vector recommendations, with Gemini explaining each pick
3. Scrape Fandom wikis (MediaWiki API) for 3-5 games, chunk by section, embed
4. Lore Q&A: hybrid BM25 + vector search, answers with citations, says "I don't know" when unsure
5. Intent router + Streamlit chat UI
6. Spoiler control (filter lore by how far the user has played)
7. Evaluation harness

## Stack
- Python 3.12 (managed with uv; `uv sync`, `uv run ...`), google-genai SDK, ChromaDB (local persistent), SQLite, httpx; Streamlit later
- Embeddings: `gemini-embedding-001`. Model names live in `app/config.py`, never hardcoded elsewhere.
- API keys in `.env` (git-ignored), loaded with python-dotenv; `.env.example` documents them.

## Layout
- `app/config.py` – settings, paths, model names, rate limits
- `app/db.py` – SQLite schema + upserts (`data/games.db`)
- `app/rawg_client.py` – RAWG API client (pagination, retries, rate limiting); `python -m app.fetch`
- `app/embeddings.py` – text profiles + batched Gemini embedding with 429 backoff; `python -m app.embed`
- `app/vector_store.py` – ChromaDB wrapper (`data/chroma/`, cosine distance)
- `app/similar.py` – `python -m app.similar "Hollow Knight"`
- `tests/` – pytest (`uv run pytest`)

## Conventions
- Re-runs must be idempotent: upsert by RAWG id, skip already-embedded games (profile hash cache).
- Respect free-tier rate limits for both RAWG and Gemini.
- Commit messages: all lowercase, no AI attribution/co-author lines.
