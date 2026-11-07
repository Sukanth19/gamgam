from app import db
from app.retrieval import rrf


def test_rrf_rewards_items_found_by_both_lists():
    scores = rrf([["a", "b", "c"], ["c", "d"]], k=60)
    assert max(scores, key=scores.get) == "c"           # rank 3 + rank 1 beats a single rank 1
    assert scores["a"] > scores["b"] == scores["d"]          # same rank -> same score, whichever list


def test_fts_query_neutralises_operators_and_stopwords():
    assert db.fts_query('Who is "Ilex" -- AND the NEAR(ghost*)?') == '"ilex" OR "near" OR "ghost"'
    assert db.fts_query("who is the") is None


def test_bm25_search_filters_by_game_and_ranks_title_matches_high():
    conn = db.connect(":memory:")
    db.upsert_games(conn, [{"id": 1, "slug": "a", "name": "Outer Wilds"},
                           {"id": 2, "slug": "b", "name": "Hollow Knight"}])
    def page(key, game, title, text):
        db.save_page(conn, {"page_key": key, "game_id": game, "wiki": "w", "title": title,
                            "url": "u", "revid": 1},
                     [{"chunk_id": f"{key}#0", "position": 0, "section": "", "kind": "text", "text": text}])
    page("w:1", 1, "Ilex", "A Nomai child who played the anglerfish game.")
    page("w:2", 1, "Solanum", "Solanum played games with her friend Ilex as a child.")
    page("w:3", 2, "Ilex", "Unrelated bug also called Ilex for testing the game filter.")
    hits = db.bm25_search(conn, "who is Ilex?", 5, game_id=1)
    assert [cid for cid, _ in hits] == ["w:1#0", "w:2#0"]   # title match first; other game excluded
    # re-saving a page must not leave duplicate rows in the keyword index
    page("w:1", 1, "Ilex", "Rewritten text about the Nomai child Ilex.")
    assert conn.execute("SELECT COUNT(*) FROM lore_fts").fetchone()[0] == 3
