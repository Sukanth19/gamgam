"""MediaWiki API client (works for Fandom and any other MediaWiki site).

Docs: https://www.mediawiki.org/wiki/API:Main_page

Key decisions:
- Use the API, not HTML scraping: stable, structured, and what wikis ask bots to use.
- Fetch page content in BATCHES (many titles per request), ~1 request/second.
- Large batches can be cut short by the server; MediaWiki then returns a
  `continue` token, and we keep requesting until everything has arrived.
"""

import re
from collections.abc import Iterator

import httpx

from app import config
from app.http_retry import ThrottledHttp


class WikiClient:
    def __init__(self, api_url: str, *, min_interval: float = config.WIKI_MIN_INTERVAL_S,
                 transport: httpx.BaseTransport | None = None):
        self.api_url = api_url
        # base_url = scheme+host, request path = "/api.php": passing the full URL as
        # base_url with an empty path makes httpx add a trailing slash -> 403/404.
        url = httpx.URL(api_url)
        self._path = url.path
        self._http = ThrottledHttp(base_url=str(url.copy_with(path="/", query=None)),
                                   min_interval=min_interval,
                                   max_retries=config.WIKI_MAX_RETRIES, transport=transport)
        self._http.http.follow_redirects = True

    def close(self):
        self._http.close()

    def _query(self, params: dict) -> Iterator[dict]:
        """Run an action=query request, following `continue` tokens."""
        base = {"action": "query", "format": "json", "formatversion": 2, **params}
        cont: dict = {}
        while True:
            data = self._http.request("GET", self._path, params={**base, **cont}).json()
            if "error" in data:
                raise RuntimeError(f"{self.api_url}: {data['error'].get('info', data['error'])}")
            yield data
            if "continue" not in data:
                return
            cont = data["continue"]

    def category_pages(self, category: str) -> list[dict]:
        """Articles in a category: [{pageid, title, length}] (namespace 0 only)."""
        pages = {}
        for data in self._query({"generator": "categorymembers", "gcmtitle": f"Category:{category}",
                                 "gcmnamespace": 0, "gcmlimit": 500, "prop": "info"}):
            for p in data.get("query", {}).get("pages", []):
                pages[p["pageid"]] = {"pageid": p["pageid"], "title": p["title"],
                                      "length": p.get("length", 0)}
        return list(pages.values())

    def page_revisions(self, pageids: list[int]) -> dict[int, int]:
        """Current revision id per page, cheap (no content). Used to skip unchanged pages."""
        out = {}
        for start in range(0, len(pageids), 50):
            ids = "|".join(map(str, pageids[start:start + 50]))
            for data in self._query({"pageids": ids, "prop": "info"}):
                for p in data.get("query", {}).get("pages", []):
                    if "lastrevid" in p:
                        out[p["pageid"]] = p["lastrevid"]
        return out

    def fetch_pages(self, pageids: list[int]) -> list[dict]:
        """Full pages: [{pageid, title, revid, url, wikitext, categories}]."""
        out: dict[int, dict] = {}
        for start in range(0, len(pageids), config.WIKI_BATCH_SIZE):
            ids = "|".join(map(str, pageids[start:start + config.WIKI_BATCH_SIZE]))
            for data in self._query({"pageids": ids, "prop": "revisions|categories|info",
                                     "rvprop": "ids|content", "rvslots": "main",
                                     "inprop": "url", "cllimit": "max", "clshow": "!hidden"}):
                for p in data.get("query", {}).get("pages", []):
                    page = out.setdefault(p["pageid"], {"pageid": p["pageid"], "title": p["title"],
                                                        "url": p.get("fullurl"), "categories": [],
                                                        "revid": None, "wikitext": None})
                    page["categories"] += [c["title"].split(":", 1)[1] for c in p.get("categories", [])]
                    if p.get("revisions"):
                        rev = p["revisions"][0]
                        page["revid"] = rev["revid"]
                        page["wikitext"] = rev["slots"]["main"]["content"]
        return [p for p in out.values() if p["wikitext"] is not None]


def select_pages(client: WikiClient, categories: list[str], max_pages: int,
                 skip_title_re: str) -> list[dict]:
    """Walk categories in order; longest pages first within each; cap total."""
    skip = re.compile(skip_title_re)
    chosen: dict[int, dict] = {}
    per_cat = [sorted(client.category_pages(c), key=lambda p: -p["length"]) for c in categories]
    # Round-robin across categories so a huge first category can't eat the whole budget.
    i = 0
    while len(chosen) < max_pages and any(i < len(pc) for pc in per_cat):
        for pc in per_cat:
            if i < len(pc) and len(chosen) < max_pages:
                p = pc[i]
                if p["pageid"] not in chosen and not skip.search(p["title"]):
                    chosen[p["pageid"]] = p
        i += 1
    return list(chosen.values())
