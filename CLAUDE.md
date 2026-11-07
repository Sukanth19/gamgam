# gamgam — game recommender + game lore q&a

A learning project: a Python RAG app built on the Gemini API (free tier).
The user is learning, so briefly explain the key decisions when changing code.

## Roadmap
1. [done] Fetch game metadata (IGDB by default; RAWG optional) into SQLite, embed it, "similar games" search
2. [done] Play history (liked/disliked) -> taste-vector recommendations, with Gemini explaining each pick. Known gap: one averaged taste vector drowns minority tastes (multi-interest retrieval would fix)
3. Scrape Fandom wikis (MediaWiki API) for 3-5 games, chunk by section, embed
4. Lore Q&A: hybrid BM25 + vector search, answers with citations, says "I don't know" when unsure
5. Intent router + Streamlit chat UI
6. Spoiler control (filter lore by how far the user has played)
7. Evaluation harness

## Stack
- Python 3.12 (managed with uv; `uv sync`, `uv run ...`), google-genai SDK, ChromaDB (local persistent), SQLite, httpx; Streamlit later
- Embeddings: `EMBED_PROVIDER` (.env) = `local` (default; `BAAI/bge-base-en-v1.5` on CPU via fastembed, no limits, ~7 min for 2000 games) or `gemini` (`gemini-embedding-001`; free tier ran out at ~1000 texts/day). Each provider has its own Chroma collection `games_<source>_<provider>`. Model names live in `app/config.py`, never hardcoded elsewhere.
- API keys in `.env` (git-ignored), loaded with python-dotenv; `.env.example` documents them.

## Layout
- `app/config.py` – settings, paths, model names, rate limits
- `app/db.py` – SQLite schema + upserts (`data/games_<source>.db`)
- `app/http_retry.py` – shared throttling + retry/backoff for API clients
- `app/igdb_client.py` – IGDB client (Twitch OAuth, Apicalypse queries) — default source
- `app/rawg_client.py` – RAWG client (RAWG was down with Cloudflare 522 errors on 2026-11-03/04)
- `app/fetch.py` – `python -m app.fetch`, picks the client from `GAME_SOURCE`
- `app/embeddings.py` – text profiles (junk/camera tags filtered, summary/storyline deduped) + local/Gemini embedders
- `app/embed.py` – `python -m app.embed`
- `app/vector_store.py` – ChromaDB wrapper (`data/chroma/`, cosine distance)
- `app/similar.py` – `python -m app.similar "Hollow Knight"`
- `app/history.py` – `python -m app.history like|dislike|list|remove "Game"` (play_history table)
- `app/recommend.py` – Rocchio taste vector + MMR diversity + "because you liked X"; `python -m app.recommend`
- `app/explain.py` – one grounded Gemini call (JSON schema) explaining all picks; falls back across `CHAT_MODELS`, never fatal
- `tests/` – pytest (`uv run pytest`)

## Conventions
- `GAME_SOURCE` (.env) selects igdb|rawg; ids differ per source, so each source has its own db file and chroma collection. Clients all emit the same flat game dict.
- Re-runs must be idempotent: upsert by source id, skip already-embedded games (profile hash cache).
- Respect free-tier rate limits (IGDB 4 req/s, RAWG, Gemini). Long Gemini 429 waits (>5 min) mean daily quota: stop, don't sleep.
- Profile changes re-embed everything (~7 min local). Judge them on a fixed set of seed games, not one example; tag ranking (`PROFILE_RANK_TAGS`) made results worse by eye.
- Commit messages: all lowercase, no AI attribution/co-author lines.
