"""Offline MediaWiki client tests (MockTransport, no network)."""

import httpx
import pytest

from app import db, http_retry
from app.wiki_client import WikiClient, select_pages
from app.wikis import SKIP_TITLE_RE


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(http_retry.time, "sleep", lambda s: None)


def client_with(handler):
    return WikiClient("https://example.fandom.com/api.php", min_interval=0,
                      transport=httpx.MockTransport(handler))


def test_requests_hit_api_php_without_trailing_slash():
    seen = []
    def handler(req):
        seen.append(req.url.path)
        return httpx.Response(200, json={"query": {"pages": []}})
    client_with(handler).category_pages("Lore")
    assert seen == ["/api.php"]


def test_fetch_pages_follows_continue_and_merges_categories():
    def handler(req):
        p = req.url.params
        if "clcontinue" not in p:   # first response: content for page 1, categories cut short
            return httpx.Response(200, json={
                "continue": {"clcontinue": "1|B", "continue": "||"},
                "query": {"pages": [{"pageid": 1, "title": "Hornet", "fullurl": "https://x/Hornet",
                                     "categories": [{"title": "Category:NPCs"}],
                                     "revisions": [{"revid": 7, "slots": {"main": {"content": "text"}}}]}]}})
        return httpx.Response(200, json={"query": {"pages": [
            {"pageid": 1, "title": "Hornet", "categories": [{"title": "Category:Bosses"}]}]}})
    pages = client_with(handler).fetch_pages([1])
    assert len(pages) == 1
    assert pages[0]["categories"] == ["NPCs", "Bosses"]
    assert pages[0]["revid"] == 7 and pages[0]["url"] == "https://x/Hornet"


def test_api_error_is_raised():
    def handler(req):
        return httpx.Response(200, json={"error": {"code": "badvalue", "info": "bad things"}})
    with pytest.raises(RuntimeError, match="bad things"):
        client_with(handler).category_pages("Lore")


def test_select_pages_round_robin_longest_first_skips_translations():
    class Fake:
        def category_pages(self, cat):
            return {"Big": [{"pageid": i, "title": f"Big {i}", "length": i} for i in range(1, 50)],
                    "Small": [{"pageid": 100, "title": "GLaDOS", "length": 5},
                              {"pageid": 101, "title": "GLaDOS/fr", "length": 9}]}[cat]
    chosen = select_pages(Fake(), ["Big", "Small"], max_pages=4, skip_title_re=SKIP_TITLE_RE)
    titles = [p["title"] for p in chosen]
    assert titles[0] == "Big 49"               # longest first
    assert "GLaDOS" in titles                  # small category still gets a slot
    assert "GLaDOS/fr" not in titles           # translation skipped
    assert len(titles) == 4


def test_save_page_replaces_chunks_on_new_revision():
    conn = db.connect(":memory:")
    db.upsert_games(conn, [{"id": 1, "slug": "hk", "name": "Hollow Knight"}])
    page = {"page_key": "w:1", "game_id": 1, "wiki": "w", "title": "Hornet", "url": "u", "revid": 1}
    db.save_page(conn, page, [{"chunk_id": f"w:1#{i}", "position": i, "section": "", "kind": "text",
                               "text": f"old {i}"} for i in range(3)])
    db.save_page(conn, {**page, "revid": 2, "page_spoiler": True},
                 [{"chunk_id": "w:1#0", "position": 0, "section": "Lore", "kind": "text", "text": "new"}])
    chunks = db.get_lore_chunks(conn)
    assert [c["text"] for c in chunks] == ["new"]          # old chunks gone, not appended
    assert chunks[0]["page_spoiler"] == 1 and chunks[0]["game"] == "Hollow Knight"
    assert db.page_revids(conn, "w") == {"w:1": 2}
