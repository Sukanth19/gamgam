"""Offline tests for the RAWG client using httpx's MockTransport (no network)."""

import httpx
import pytest

from app import http_retry, rawg_client
from app.rawg_client import RawgClient

RAW = {"id": 1, "slug": "hk", "name": "Hollow Knight", "released": "2017-02-24",
       "rating": 4.4, "ratings_count": 10, "playtime": 20, "added": 99,
       "genres": [{"name": "Action"}],
       "tags": [{"name": "Metroidvania", "language": "eng"},
                {"name": "Метроидвания", "language": "rus"}],
       "platforms": [{"platform": {"name": "PC"}}]}


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(http_retry.time, "sleep", lambda s: None)


def client_with(handler):
    return RawgClient("test-key", min_interval=0, max_retries=3,
                      transport=httpx.MockTransport(handler))


def test_parse_game_flattens_and_keeps_english_tags():
    g = rawg_client.parse_game(RAW)
    assert g["genres"] == ["Action"]
    assert g["tags"] == ["Metroidvania"]
    assert g["platforms"] == ["PC"]


def test_pagination_follows_next_and_respects_limit():
    def handler(req):
        page = int(req.url.params["page"])
        results = [{**RAW, "id": page * 100 + i, "slug": f"g{page}-{i}"} for i in range(3)]
        return httpx.Response(200, json={"results": results,
                                         "next": "more" if page < 5 else None})
    games = list(client_with(handler).iter_popular_games(limit=7))
    assert len(games) == 7
    assert games[0]["id"] == 100 and games[-1]["id"] == 300


def test_retries_on_429_then_succeeds():
    calls = []
    def handler(req):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(429)
        return httpx.Response(200, json={"description_raw": "ok"})
    assert client_with(handler).get_description(1) == "ok"
    assert len(calls) == 3


def test_does_not_retry_on_401():
    calls = []
    def handler(req):
        calls.append(1)
        return httpx.Response(401)
    with pytest.raises(httpx.HTTPStatusError):
        client_with(handler).get_description(1)
    assert len(calls) == 1
