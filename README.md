# gamgam

A game recommender and lore Q&A in one chat. Tell it what you liked and it suggests what to play next and explains why. Ask about a game's story and it answers from the game's wiki, cites its sources, says "I don't know" when the wiki doesn't cover it, and keeps spoilers hidden past the point you've reached.

Built as a learning project for retrieval-augmented generation (RAG) on the Gemini API free tier.

## Features

- **Recommendations**: like or dislike games, and gamgam builds a taste vector from them. Picks come from that vector plus each liked game's neighbours, so a minority taste still gets suggestions. Results are diversified with MMR, and Gemini writes a short "because you liked X" for each one
- **Similar games**: nearest neighbours of any game in a ~2,000-game catalogue pulled from IGDB
- **Lore Q&A** for Hollow Knight, Outer Wilds, The Witcher 3 and Portal 2 (775 wiki pages, ~5.1k chunks)
- **Hybrid search**: BM25 (SQLite FTS5) and vector search, merged with Reciprocal Rank Fusion
- **Grounded answers**: Gemini may only cite the passages it was given. Citations are checked and renumbered, and anything it didn't receive is dropped
- **"I don't know"**: a similarity gate (calibrated on a question set) plus the model's own answerable flag, so it declines instead of guessing
- **Spoiler control**: set how far you've played and lore past that point is filtered out. It tells you when the answer is revealed later instead of spoiling it
- **Intent router**: works out whether you want a recommendation or lore, and which game you mean, using title matching, wiki entities and a BM25 vote, with Gemini as a fallback
- **Streamlit chat UI** with a sidebar for likes, dislikes and progress, and expandable sources

## How it works

```
                 ┌──────────────┐
  you ──────────▶│ intent router │
                 └──────┬───────┘
         recommend      │       lore question
        ┌───────────────┴────────────────┐
        ▼                                ▼
  taste vector (likes − dislikes)   BM25 (FTS5) + vector search
  + per-like neighbours                 │ RRF fusion
        │ MMR diversity                 │ spoiler filter (your progress)
        ▼                                │ similarity gate
  Gemini explains each pick              ▼
                                   Gemini answer with checked citations
```

- **Game data** comes from IGDB (or RAWG) into SQLite. Each game becomes a text profile (genres, themes, summary, with store and award junk filtered out) that's embedded into ChromaDB.
- **Embeddings** run locally by default (`BAAI/bge-base-en-v1.5` via fastembed, on the CPU, no limits). Gemini embeddings are an option, but the free tier runs out at about 1,000 texts a day.
- **Wiki lore** is fetched through the MediaWiki API. Raw wikitext is stored separately from chunks, so the chunker can change without re-downloading. Chunks are ~250 words with overlap. They keep infobox facts, quotes and spoiler templates, and skip galleries, trivia and bug lists.
- **Spoiler tags**: Portal and Outer Wilds mark spoilers on their wikis, but Hollow Knight and The Witcher don't. Gemini tags each wiki section with the story milestone it belongs to. When unsure it picks the later milestone, and untagged sections count as spoilers, so a partial run never leaks anything.

## Tech stack

Python 3.12, uv, google-genai (Gemini), ChromaDB, SQLite + FTS5, fastembed, httpx, mwparserfromhell, Streamlit, pytest

## Running it

### 1. Install

```bash
uv sync
cp .env.example .env
```

### 2. Keys

| Variable | Where to get it |
| --- | --- |
| `GEMINI_API_KEY` | [Google AI Studio](https://aistudio.google.com/apikey) (free tier is enough) |
| `TWITCH_CLIENT_ID`, `TWITCH_CLIENT_SECRET` | IGDB uses Twitch login: create a free app at [dev.twitch.tv](https://dev.twitch.tv/console/apps) |
| `RAWG_API_KEY` | Only if `GAME_SOURCE=rawg` |

### 3. Build the data

```bash
uv run python -m app.fetch          # ~2,000 games from IGDB into data/
uv run python -m app.embed          # embed them (~7 min on CPU)
uv run python -m app.wiki fetch     # download wiki lore
uv run python -m app.wiki embed     # embed lore chunks (~18 min on CPU)
uv run python -m app.spoilers tag   # tag sections with story milestones (resumable)
```

Everything is idempotent: re-running skips work that's already done.

### 4. Chat

```bash
uv run streamlit run streamlit_app.py
```

Or use the CLIs directly:

```bash
uv run python -m app.similar "Hollow Knight"
uv run python -m app.history like "Outer Wilds"
uv run python -m app.recommend
uv run python -m app.ask "who is the radiance?" --game "Hollow Knight"
uv run python -m app.spoilers progress "Hollow Knight" "City of Tears"
```

## Adding a game's lore

Add an entry to `app/wikis.py` with the wiki's `api.php` URL and the categories to fetch (any MediaWiki wiki works), then run `app.wiki fetch --game "<name>"` and `app.wiki embed`. For spoiler control, add its story milestones to `app/progress.py` and run `app.spoilers tag --game "<name>"`.

## Tests

```bash
uv run pytest
```

## Roadmap

- [x] Game catalogue, embeddings and similar-games search
- [x] Taste-vector recommendations with explanations
- [x] Wiki lore pipeline
- [x] Lore Q&A with hybrid search and checked citations
- [x] Intent router and Streamlit chat
- [x] Spoiler control
- [ ] Evaluation harness (`eval/`)

Known gap: game names are part of each profile, so look-alike titles can match (Outer Wilds → The Outer Worlds).
