"""Tests the LangGraph wiring itself: crisis short-circuit, the critic's
bounded retry-once loop, and the interrupt/resume question loop (including
its 3-question cap and early-exit when the model signals it has enough).
All LLM-calling agents are stubbed via monkeypatch — this tests
orchestration logic, never makes a real Anthropic call.
"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from app import graph as graph_module
from app.agents import critic, interpreter, question_generator, report_synthesizer
from app.config import settings
from schemas import CriticVerdict, Interpretation, Report, SessionStatus


def _new_graph():
    return graph_module.build_graph(MemorySaver())


def _approve_critic(monkeypatch):
    async def fake_critique(interpretation, transcript):
        return CriticVerdict(approved=True, reason="fine")
    monkeypatch.setattr(critic, "critique", fake_critique)


def _stub_interpret(monkeypatch, calls: list):
    async def fake_interpret(transcript, mood_tag, feedback=None):
        calls.append(feedback)
        return Interpretation(summary="stub summary", themes=["stub_theme"])
    monkeypatch.setattr(interpreter, "interpret", fake_interpret)


def _no_more_questions(monkeypatch):
    async def fake_generate_question(interpretation, qa_turns):
        return None
    monkeypatch.setattr(question_generator, "generate_question", fake_generate_question)


def _stub_report(monkeypatch):
    async def fake_synthesize(transcript, qa_turns):
        return Report(self_image_reframe="reframe", good_things=["thing"], actionables=["do x"])
    monkeypatch.setattr(report_synthesizer, "synthesize", fake_synthesize)


async def test_crisis_short_circuits_before_interpreter(monkeypatch):
    calls = []
    _stub_interpret(monkeypatch, calls)

    graph = _new_graph()
    response = await graph_module.start_session(
        graph, "s-crisis", "sad", "I don't want to be alive anymore, I keep thinking about ending it all."
    )

    assert response.status == SessionStatus.CRISIS_TERMINATED
    assert response.crisis_message is not None
    assert response.question is None
    assert response.report is None
    assert calls == []  # interpreter never invoked


async def test_normal_flow_reaches_first_question(monkeypatch):
    calls = []
    _stub_interpret(monkeypatch, calls)
    _approve_critic(monkeypatch)

    async def fake_generate_question(interpretation, qa_turns):
        return "What happened right before that?"
    monkeypatch.setattr(question_generator, "generate_question", fake_generate_question)

    graph = _new_graph()
    response = await graph_module.start_session(graph, "s-normal", "stressed", "Work has been a lot lately.")

    assert response.status == SessionStatus.ACTIVE
    assert response.question == "What happened right before that?"
    assert response.qa_turns == []
    assert len(calls) == 1


async def test_critic_retries_once_then_proceeds(monkeypatch):
    calls = []
    _stub_interpret(monkeypatch, calls)
    _no_more_questions(monkeypatch)
    _stub_report(monkeypatch)

    critique_calls = []

    async def fake_critique(interpretation, transcript):
        critique_calls.append(1)
        approved = len(critique_calls) > 1  # reject first, approve on retry
        return CriticVerdict(approved=approved, reason="reason")
    monkeypatch.setattr(critic, "critique", fake_critique)

    graph = _new_graph()
    response = await graph_module.start_session(graph, "s-retry", "stressed", "Some transcript text.")

    assert len(calls) == 2  # interpreter called again after rejection
    assert len(critique_calls) == 2
    assert calls[1] is not None  # retry was given the rejection reason as feedback
    assert response.status == SessionStatus.COMPLETED
    assert response.report is not None


async def test_critic_force_approves_after_max_retries(monkeypatch):
    calls = []
    _stub_interpret(monkeypatch, calls)
    _no_more_questions(monkeypatch)
    _stub_report(monkeypatch)

    async def always_rejecting(interpretation, transcript):
        return CriticVerdict(approved=False, reason="never good enough")
    monkeypatch.setattr(critic, "critique", always_rejecting)

    graph = _new_graph()
    response = await graph_module.start_session(graph, "s-force-approve", "stressed", "Some transcript text.")

    # Bounded by settings.max_critic_attempts (1 retry) -> 2 interpreter calls total, never infinite.
    assert len(calls) == settings.max_critic_attempts + 1
    assert response.status == SessionStatus.COMPLETED  # still reaches a report, doesn't hang


async def test_question_loop_respects_three_question_cap(monkeypatch):
    calls = []
    _stub_interpret(monkeypatch, calls)
    _approve_critic(monkeypatch)
    _stub_report(monkeypatch)

    async def always_asks(interpretation, qa_turns):
        return f"Question #{len(qa_turns) + 1}?"
    monkeypatch.setattr(question_generator, "generate_question", always_asks)

    graph = _new_graph()
    response = await graph_module.start_session(graph, "s-cap", "stressed", "Some transcript text.")
    assert response.status == SessionStatus.ACTIVE
    assert response.question == "Question #1?"

    for expected_next_question in ["Question #2?", "Question #3?", None]:
        response = await graph_module.resume_session(graph, "s-cap", "an answer")
        assert response is not None
        if expected_next_question is None:
            assert response.status == SessionStatus.COMPLETED
            assert response.report is not None
            assert len(response.qa_turns) == settings.max_questions
        else:
            assert response.status == SessionStatus.ACTIVE
            assert response.question == expected_next_question


async def test_question_loop_stops_early_when_model_signals_done(monkeypatch):
    calls = []
    _stub_interpret(monkeypatch, calls)
    _approve_critic(monkeypatch)
    _stub_report(monkeypatch)

    question_calls = []

    async def one_question_then_done(interpretation, qa_turns):
        question_calls.append(1)
        return "Only question?" if len(question_calls) == 1 else None
    monkeypatch.setattr(question_generator, "generate_question", one_question_then_done)

    graph = _new_graph()
    response = await graph_module.start_session(graph, "s-early-stop", "stressed", "Some transcript text.")
    assert response.question == "Only question?"

    response = await graph_module.resume_session(graph, "s-early-stop", "an answer")
    assert response is not None
    assert response.status == SessionStatus.COMPLETED
    assert len(response.qa_turns) == 1
    assert response.report is not None


async def test_unknown_session_returns_none(monkeypatch):
    graph = _new_graph()
    assert await graph_module.get_session(graph, "does-not-exist") is None
