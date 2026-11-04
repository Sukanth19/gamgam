"""Offline tests for the IGDB client (MockTransport, no network)."""

import httpx
import pytest

from app import http_retry, igdb_client
from app.igdb_client import IgdbClient

RAW = {"id": 14593, "slug": "hollow-knight", "name": "Hollow Knight",
       "first_release_date": 1487894400, "total_rating": 88.0, "total_rating_count": 2500,
       "summary": "Forge your own path.", "storyline": "Beneath Dirtmouth lies Hallownest.",
       "genres": [{"id": 8, "name": "Platform"}],
       "themes": [{"id": 1, "name": "Action"}],
       "player_perspectives": [{"id": 4, "name": "Side view"}],
       "keywords": [{"id": 5, "name": "metroidvania"}],
       "platforms": [{"id": 6, "name": "PC (Microsoft Windows)"}]}


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(http_retry.time, "sleep", lambda s: None)


def test_parse_game_maps_to_common_shape():
    g = igdb_client.parse_game(RAW)
    assert g["released"] == "2017-02-24"
    assert g["rating"] == 4.4                       # 88/100 -> 4.4/5
    assert g["genres"] == ["Platform"]
    assert g["tags"] == ["Action", "Side view", "metroidvania"]
    assert g["description"] == "Forge your own path.\n\nBeneath Dirtmouth lies Hallownest."


def test_parse_game_tolerates_missing_fields():
    g = igdb_client.parse_game({"id": 1, "slug": "x", "name": "X"})
    assert g["released"] is None and g["rating"] is None
    assert g["genres"] == [] and g["description"] is None


def test_auth_then_paginated_queries():
    bodies = []
    def handler(req):
        if req.url.host == "id.twitch.tv":
            assert req.url.path == "/oauth2/token"           # exact path, no trailing slash
            assert "secret" not in str(req.url)              # secret only in the body
            assert b"grant_type=client_credentials" in req.content
            return httpx.Response(200, json={"access_token": "tok", "expires_in": 5000})
        assert req.headers["Authorization"] == "Bearer tok"
        assert req.headers["Client-ID"] == "cid"
        body = req.content.decode()
        bodies.append(body)
        offset = int(body.split("offset ")[1].rstrip(";"))
        limit = int(body.split("limit ")[1].split(";")[0])
        n = min(limit, 1200 - offset)              # pretend IGDB has 1200 games
        return httpx.Response(200, json=[{**RAW, "id": offset + i} for i in range(n)])

    client = IgdbClient("cid", "secret", min_interval=0, transport=httpx.MockTransport(handler))
    games = list(client.iter_popular_games(limit=1100))
    assert len(games) == 1100
    assert len({g["id"] for g in games}) == 1100
    assert [b.split("offset ")[1] for b in bodies] == ["0;", "500;", "1000;"]
    assert "limit 100;" in bodies[-1]               # last page only asks for what's left


def test_missing_credentials_raise():
    with pytest.raises(RuntimeError, match="TWITCH_CLIENT_ID"):
        IgdbClient("", "")
