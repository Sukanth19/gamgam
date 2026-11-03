import pytest

from app import db


@pytest.fixture
def conn():
    c = db.connect(":memory:")
    yield c
    c.close()


def make_game(**overrides):
    g = {"id": 1, "slug": "hollow-knight", "name": "Hollow Knight", "released": "2017-02-24",
         "rating": 4.4, "ratings_count": 3000, "playtime": 20, "added": 9000,
         "genres": ["Action", "Platformer"], "tags": ["Metroidvania"], "platforms": ["PC"],
         "description": "Bugs and swords."}
    g.update(overrides)
    return g


def test_insert_and_read_back_roundtrips_lists(conn):
    db.upsert_games(conn, [make_game()])
    g = db.get_game(conn, 1)
    assert g["name"] == "Hollow Knight"
    assert g["genres"] == ["Action", "Platformer"]
    assert g["platforms"] == ["PC"]


def test_rerun_updates_instead_of_duplicating(conn):
    db.upsert_games(conn, [make_game()])
    db.upsert_games(conn, [make_game(rating=4.6, tags=["Metroidvania", "Souls-like"])])
    assert db.count_games(conn) == 1
    g = db.get_game(conn, 1)
    assert g["rating"] == 4.6
    assert g["tags"] == ["Metroidvania", "Souls-like"]


def test_upsert_without_description_keeps_existing_one(conn):
    db.upsert_games(conn, [make_game()])
    db.upsert_games(conn, [make_game(description=None)])
    assert db.get_game(conn, 1)["description"] == "Bugs and swords."
    assert db.ids_missing_description(conn) == []


def test_find_game_by_name_exact_then_substring(conn):
    db.upsert_games(conn, [make_game(),
                           make_game(id=2, slug="hollow-knight-silksong",
                                     name="Hollow Knight: Silksong", added=100)])
    assert db.find_game_by_name(conn, "hollow knight")["id"] == 1
    assert db.find_game_by_name(conn, "silksong")["id"] == 2
    assert db.find_game_by_name(conn, "zelda") is None
