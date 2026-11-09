# gamgam — game recommender + game lore q&a

A learning project: a Python RAG app. Runtime LLM is local (Ollama, qwen3:8b on the RTX 4060); Gemini is only used for the one-time spoiler tagging.
The user is learning, so briefly explain the key decisions when changing code.

## Roadmap
1. [done] Fetch game metadata (IGDB by default; RAWG optional) into SQLite, embed it, "similar games" search
2. [done] Play history (liked/disliked) -> taste-vector recommendations, with Gemini explaining each pick. Multi-interest: candidates from the taste vector + each liked game's neighbours; relevance blends both (`TASTE_BLEND`). Profiles are name-free since phase 7 (similar eval 8/12 -> 10/12); Outer Wilds -> The Outer Worlds persists: it comes from description text, not the title
3. [done] Wiki lore (MediaWiki API) for Hollow Knight, Outer Wilds, Witcher 3, Portal 2: 775 pages -> ~5.1k chunks, ~18 min to embed locally. Spoilers: Portal marks inline `{{spoiler|..}}`, Outer Wilds whole pages; HK and Witcher don't mark spoilers (phase 6 needs another signal, e.g. area/quest categories)
4. [done] Lore Q&A: hybrid BM25 (SQLite FTS5) + vector search fused with RRF, grounded Gemini answers with checked/renumbered citations, "I don't know" via similarity gate (0.55, calibrated on 16 questions) + model's answerable flag. Eval seed set: `eval/lore_questions.jsonl`
5. [done] Intent router (rules -> wiki-title entities -> BM25 game vote -> Gemini fallback) + Streamlit chat UI (`uv run streamlit run streamlit_app.py`)
6. [done] Spoiler control: per-game milestones (`app/progress.py`), every wiki section tagged by Gemini from a 120-word sample (`python -m app.spoilers tag`), lore filtered by the user's progress. Untagged = hidden; wiki `{{spoiler}}` marks hide until the main story is finished; if hidden matches clearly beat visible ones -> "revealed later". Errs toward over-hiding. Eval cases: `eval/spoiler_questions.jsonl`
7. [in progress] Evaluation harness: `python -m app.eval [--suite S] [--set NAME=VAL] [--tag T] | --compare A B` (results in eval/results/; runs with LLM outages are refused). Baseline: lore 42/42, retrieval hit@1 0.76 / hit@3 0.94, similar 10/12 (name-free profiles; was 8/12 with names), spoilers leak 2/16, over-hide 5/16. Section-sampled spoiler tags miss content deep in long sections (Crones > Biography tagged Velen but reveals post-Kaer Morhen events) -> next: chunk-level tagging

## Stack
- Python 3.12 (managed with uv; `uv sync`, `uv run ...`), ChromaDB (local persistent), SQLite, httpx, Streamlit; google-genai SDK (optional)
- LLM: `LLM_PROVIDER` (.env) = `ollama` (default; `OLLAMA_MODEL=qwen3:8b`, schema-constrained JSON, thinking off; 16/16 on eval/lore_questions) or `gemini`. Spoiler tagging uses `SPOILER_TAG_PROVIDER` (default gemini: qwen3:8b tagged 27% of HK sections 2+ milestones too early)
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
- `app/wikis.py` – which wikis/categories to fetch (add a game = add an entry; any MediaWiki api.php)
- `app/wiki_client.py` – MediaWiki API client (batched page fetch, follows `continue`, ~1 req/s)
- `app/chunking.py` – wikitext -> chunks: infobox facts, keeps quote/dialogue/spoiler template text, HTML headings, skips gallery/trivia/bugs, ~250-word chunks w/ overlap
- `app/wiki.py` – `python -m app.wiki fetch [--game X] [--force] | rechunk | embed | search "q" [--game X] | stats`
- `app/llm.py` – `make_client(provider)` + `generate_json(client, prompt, Schema)`: Ollama or Gemini (model fallback, timeout); never raises
- `app/progress.py` – story milestones per game + `parse_progress`
- `app/spoilers.py` – `python -m app.spoilers tag [--game X] [--retag] [--provider] | status | progress "Game" "milestone"|off`
- `app/retrieval.py` – hybrid search: vector + BM25 (`lore_fts`), Reciprocal Rank Fusion
- `app/ask.py` – `python -m app.ask "question" --game X [--debug]`
- `app/router.py` – intent + game detection; single-word titles ("Inside", "Journey") need title-case to match
- `app/chat.py` – `Assistant.handle(text, last_game) -> Reply`; all chat logic, no UI code
- `streamlit_app.py` – UI only (chat, sidebar likes/dislikes, sources w/ spoiler expander); usage stats off in `.streamlit/config.toml`
- `app/eval.py` – evaluation suites: retrieval (no LLM), similar (no LLM), lore, spoilers
- `eval/` – lore_questions, spoiler_questions, similar_games (.jsonl; `"draft": true` = not yet reviewed by the user); results/
- `tests/` – pytest (`uv run pytest`)

## Conventions
- `GAME_SOURCE` (.env) selects igdb|rawg; ids differ per source, so each source has its own db file and chroma collection. Clients all emit the same flat game dict.
- Re-runs must be idempotent: upsert by source id, skip already-embedded games (profile hash cache).
- Respect free-tier rate limits (IGDB 4 req/s, RAWG, Gemini). Long Gemini 429 waits (>5 min) mean daily quota: stop, don't sleep.
- Profile changes re-embed everything (~7 min local). Judge them on a fixed set of seed games, not one example; tag ranking (`PROFILE_RANK_TAGS`) made results worse by eye.
- Keep raw source data (wikitext in `wiki_pages`) separate from derived data (chunks, vectors): chunker changes -> `rechunk` + `embed`, no re-download.
- Lore embeddings use retrieval mode: documents plain, queries get the bge query prefix (`make_embedder("query")`).
- FTS5 queries are built by `db.fts_query` (quoted word tokens OR'd): never pass raw user text to MATCH.
- Citations: answers may only cite sources that were sent; `check_citations` drops others and splits "[1, 2]".
- UI smoke test: `streamlit.testing.v1.AppTest` drives streamlit_app.py headlessly (loads real models; not in the unit suite).
- Resume an interrupted tag run WITHOUT `--retag` (it clears that game's tags first).
- Don't run two model-heavy jobs at once (embed + eval + Ollama OOM-killed Ollama on 14 GB RAM).
- Fix individual spoiler mis-tags with `python -m app.spoilers fix "Game" "Page" "Section" N`.
- Commit messages: all lowercase, no AI attribution/co-author lines.
