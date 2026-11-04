import math

from app import embeddings
from app.embeddings import build_profile, clean_description, profile_hash


def game(**kw):
    g = {"id": 1, "name": "Hollow Knight", "genres": ["Action", "Platformer"],
         "tags": ["Singleplayer", "Metroidvania", "Steam Achievements", "Souls-like"],
         "description": "Explore a <b>vast</b> ruined kingdom &amp; fight bugs."}
    g.update(kw)
    return g


def test_profile_contains_name_genres_tags_and_clean_description():
    p = build_profile(game())
    assert p.splitlines()[0] == "Hollow Knight."
    assert "Genres: Action, Platformer." in p
    assert "Explore a vast ruined kingdom & fight bugs." in p
    assert "<b>" not in p and "&amp;" not in p


def test_profile_drops_noise_tags():
    p = build_profile(game())
    assert "Tags: Metroidvania, Souls-like." in p
    assert "Steam Achievements" not in p and "Singleplayer" not in p


def test_profile_handles_missing_fields():
    p = build_profile({"name": "Mystery", "genres": [], "tags": [], "description": None})
    assert p == "Mystery."


def test_clean_description_strips_spanish_translation_and_truncates():
    text = "An English description.\n\nEspañol\nUna descripción en español."
    assert clean_description(text) == "An English description."
    long = clean_description("word " * 1000, max_chars=50)
    assert len(long) <= 53 and long.endswith("...") and "wor..." not in long


def test_profile_hash_changes_with_text():
    assert profile_hash("a") == profile_hash("a")
    assert profile_hash("a") != profile_hash("b")


def test_normalize_gives_unit_vector():
    v = embeddings.normalize([3.0, 4.0])
    assert math.isclose(math.hypot(*v), 1.0)


class FakeModels:
    def __init__(self, fail_times=0, retry_delay=None):
        self.calls, self.fail_times, self.retry_delay = [], fail_times, retry_delay

    def embed_content(self, model, contents, config):
        from google.genai import errors
        self.calls.append(len(contents))
        if len(self.calls) <= self.fail_times:
            details = ([{"@type": "type.googleapis.com/google.rpc.RetryInfo",
                         "retryDelay": f"{self.retry_delay}s"}] if self.retry_delay else [])
            raise errors.ClientError(429, {"error": {"code": 429, "message": "quota",
                                                     "status": "RESOURCE_EXHAUSTED",
                                                     "details": details}})
        class E:  # mimic SDK response shape
            def __init__(s, v): s.values = v
        class R:
            embeddings = [E([1.0, 1.0]) for _ in contents]
        return R()


class FakeClient:
    def __init__(self, **kw): self.models = FakeModels(**kw)


def test_embedder_gives_up_on_daily_quota_instead_of_sleeping(monkeypatch):
    import pytest
    slept = []
    monkeypatch.setattr(embeddings.time, "sleep", slept.append)
    e = embeddings.Embedder(FakeClient(fail_times=1, retry_delay=76979), min_interval=0)
    with pytest.raises(embeddings.QuotaExhausted):
        list(e.embed(["a"]))
    assert slept == []                              # never slept 21 hours


def test_embedder_batches_and_retries_429(monkeypatch):
    monkeypatch.setattr(embeddings.time, "sleep", lambda s: None)
    client = FakeClient(fail_times=2)
    e = embeddings.Embedder(client, batch_size=2, min_interval=0)
    out = list(e.embed(["a", "b", "c"]))
    assert [start for start, _ in out] == [0, 2]
    assert [len(v) for _, v in out] == [2, 1]
    assert client.models.calls == [2, 2, 2, 1]   # two 429s, then success, then last batch
