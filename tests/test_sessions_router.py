"""FastAPI-level tests for the session lifecycle API. Uses a fresh app with
an in-memory checkpointer (no real sqlite file, no network) and stubs every
LLM-calling agent, so this exercises routing/validation/HTTP-status logic
end to end without depending on app.main's real AsyncSqliteSaver setup.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

from app import graph as graph_module
from app.agents import critic, interpreter, question_generator, report_synthesizer
from app.routers import sessions
from schemas import CriticVerdict, Interpretation, Report


def _make_client() -> TestClient:
    app = FastAPI()
    app.include_router(sessions.router)
    app.state.graph = graph_module.build_graph(MemorySaver())
    return TestClient(app)


@pytest.fixture(autouse=True)
def stub_agents(monkeypatch):
    async def fake_interpret(transcript, mood_tag, feedback=None):
        return Interpretation(summary="stub summary", themes=["stub_theme"])
    monkeypatch.setattr(interpreter, "interpret", fake_interpret)

    async def fake_critique(interpretation, transcript):
        return CriticVerdict(approved=True, reason="fine")
    monkeypatch.setattr(critic, "critique", fake_critique)

    # One question, then done — keeps the lifecycle test short.
    calls = {"n": 0}

    async def fake_generate_question(interpretation, qa_turns):
        calls["n"] += 1
        return "Question 1?" if calls["n"] == 1 else None
    monkeypatch.setattr(question_generator, "generate_question", fake_generate_question)

    async def fake_synthesize(transcript, qa_turns):
        return Report(self_image_reframe="reframe", good_things=["thing"], actionables=["do x"])
    monkeypatch.setattr(report_synthesizer, "synthesize", fake_synthesize)


def test_full_session_lifecycle_through_api():
    client = _make_client()

    create_resp = client.post("/sessions", json={"mood_selection": "stressed", "transcript": "Work is a lot."})
    assert create_resp.status_code == 200
    body = create_resp.json()
    session_id = body["session_id"]
    assert body["status"] == "active"
    assert body["question"] == "Question 1?"
    assert body["qa_turns"] == []

    answer_resp = client.post(f"/sessions/{session_id}/answer", json={"answer": "It was fine."})
    assert answer_resp.status_code == 200
    body2 = answer_resp.json()
    assert body2["status"] == "completed"
    assert body2["question"] is None
    assert body2["report"]["self_image_reframe"] == "reframe"
    assert len(body2["qa_turns"]) == 1
    assert body2["qa_turns"][0]["answer"] == "It was fine."


def test_get_session_reflects_current_state_without_advancing():
    client = _make_client()
    create_resp = client.post("/sessions", json={"mood_selection": "stressed", "transcript": "Work is a lot."})
    session_id = create_resp.json()["session_id"]

    get_resp = client.get(f"/sessions/{session_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["question"] == "Question 1?"
    assert get_resp.json()["status"] == "active"


def test_get_unknown_session_returns_404():
    client = _make_client()
    resp = client.get("/sessions/does-not-exist")
    assert resp.status_code == 404


def test_answer_unknown_session_returns_404():
    client = _make_client()
    resp = client.post("/sessions/does-not-exist/answer", json={"answer": "x"})
    assert resp.status_code == 404


def test_answer_after_completion_returns_409():
    client = _make_client()
    create_resp = client.post("/sessions", json={"mood_selection": "stressed", "transcript": "Work is a lot."})
    session_id = create_resp.json()["session_id"]

    first = client.post(f"/sessions/{session_id}/answer", json={"answer": "fine"})
    assert first.json()["status"] == "completed"

    second = client.post(f"/sessions/{session_id}/answer", json={"answer": "more"})
    assert second.status_code == 409


def test_crisis_transcript_short_circuits_through_api():
    client = _make_client()
    resp = client.post(
        "/sessions", json={"mood_selection": "sad", "transcript": "I want to kill myself."}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "crisis_terminated"
    assert body["crisis_message"]
    assert body["question"] is None
    assert body["report"] is None


def test_create_session_rejects_empty_transcript():
    client = _make_client()
    resp = client.post("/sessions", json={"mood_selection": "stressed", "transcript": ""})
    assert resp.status_code == 422


def test_create_session_rejects_invalid_mood_selection():
    client = _make_client()
    resp = client.post("/sessions", json={"mood_selection": "furious", "transcript": "text"})
    assert resp.status_code == 422
