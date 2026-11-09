"""The harness itself: metric maths and the invalid-run guard (no models)."""

import json

import pytest

from app import eval as ev


def fake_ctx():
    class C:
        def game(self, name): return {"id": 1, "name": name}
    return C()


def test_spoiler_metrics_separate_leaks_from_over_hiding(monkeypatch):
    cases = [{"game": "Hollow Knight", "progress": "greenpath", "question": q, "expect": e}
             for q, e in [("a", "hidden"), ("b", "hidden"), ("c", "answer"), ("d", "answer")]]
    monkeypatch.setattr(ev, "load", lambda name: cases)
    answered = {"a": True, "b": False, "c": True, "d": False}   # a leaks, d over-hides
    class R:
        def __init__(self, q): self.answerable, self.reason = answered[q], ""
    monkeypatch.setattr(ev, "_ask", lambda ctx, q, g, milestone=None: R(q))
    metrics, items = ev.run_spoilers(fake_ctx())
    assert metrics["leak_rate"] == 0.5 and metrics["over_hide_rate"] == 0.5
    assert metrics["pass_rate"] == 0.5


def test_llm_outage_makes_run_invalid_and_unsaved(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(ev, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(ev, "Ctx", lambda: None)
    monkeypatch.setattr(ev, "RUNNERS", {"lore": lambda ctx: (
        {"pass_rate": 0.0}, [{"id": "q", "pass": False, "reason": "llm failed"}])})
    monkeypatch.setattr("sys.argv", ["eval", "--suite", "lore"])
    with pytest.raises(SystemExit) as e:
        ev.main()
    assert e.value.code == 1
    assert "INVALID RUN" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []                       # nothing saved
