"""LangGraph orchestration for one coaching session.

    mood_router -> safety_gate -> [crisis] -> crisis_response -> END
                                -> [else]  -> interpreter -> critic
                                                 (bounded retry: reject -> interpreter, max 1)
                                              -> question_generator <-> ask_question
                                                 (LLM proposes a question; ask_question is the
                                                  ONLY node that calls interrupt(), up to 3 rounds)
                                              -> report_synthesizer -> END

Question asking is deliberately split into two nodes. LangGraph re-runs a
node's code from the top on resume, up to the point of its (already
resolved) interrupt() call — so anything with a side effect (like an LLM
call) that sits *before* interrupt() in the same node would silently
re-execute on every resume. `question_generator` (LLM call, decides the next
question or that none is needed) and `ask_question` (nothing but a state
read and `interrupt()`) are separate nodes specifically so the LLM call is
checkpointed once and never re-run on resume.

The 3-question cap and the critic's one-retry bound are both enforced here
in plain Python (`_route_after_*`), never left to a model to self-limit —
per AGENTS.md, an LLM is not the final authority on an application rule.
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.agents import critic, interpreter, question_generator, report_synthesizer
from app.agents.mood_router import route_mood
from app.config import settings
from app.safety import CRISIS_RESOURCES_MESSAGE, detect_crisis
from schemas import (
    Interpretation,
    MoodSelection,
    MoodTag,
    QATurn,
    Report,
    SessionResponse,
    SessionStatus,
)
from telemetry import get_tracer

_tracer = get_tracer("monologue-coach.graph")


class MonologueState(TypedDict, total=False):
    session_id: str
    mood_selection: str
    transcript: str
    mood_tag: dict
    safety_verdict: dict
    interpretation: dict
    critic_verdict: dict
    critic_attempts: int
    qa_turns: list[dict]
    question_count: int
    pending_question: str | None
    questions_done: bool
    status: str
    crisis_message: str | None
    report: dict | None


def create_initial_state(session_id: str, mood_selection: str, transcript: str) -> MonologueState:
    return MonologueState(
        session_id=session_id,
        mood_selection=mood_selection,
        transcript=transcript,
        critic_attempts=0,
        qa_turns=[],
        question_count=0,
        status=SessionStatus.ACTIVE.value,
    )


# --- Nodes -----------------------------------------------------------------


async def _node_mood_router(state: MonologueState) -> MonologueState:
    with _tracer.start_as_current_span("graph.mood_router"):
        mood_tag = route_mood(MoodSelection(state["mood_selection"]), state["transcript"])
        return {"mood_tag": mood_tag.model_dump(mode="json")}


async def _node_safety_gate(state: MonologueState) -> MonologueState:
    with _tracer.start_as_current_span("graph.safety_gate") as span:
        verdict = detect_crisis(state["transcript"])
        span.set_attribute("safety_triggered", verdict.is_crisis)
        return {"safety_verdict": verdict.model_dump()}


def _route_after_safety(state: MonologueState) -> str:
    return "crisis_response" if state["safety_verdict"]["is_crisis"] else "interpreter"


async def _node_crisis_response(state: MonologueState) -> MonologueState:
    with _tracer.start_as_current_span("graph.crisis_response"):
        return {
            "status": SessionStatus.CRISIS_TERMINATED.value,
            "crisis_message": CRISIS_RESOURCES_MESSAGE,
        }


async def _node_interpreter(state: MonologueState) -> MonologueState:
    with _tracer.start_as_current_span("graph.interpreter"):
        mood_tag = MoodTag(**state["mood_tag"])
        prior_verdict = state.get("critic_verdict")
        feedback = [prior_verdict["reason"]] if prior_verdict and not prior_verdict["approved"] else None
        interpretation = await interpreter.interpret(state["transcript"], mood_tag, feedback=feedback)
        return {"interpretation": interpretation.model_dump()}


async def _node_critic(state: MonologueState) -> MonologueState:
    with _tracer.start_as_current_span("graph.critic") as span:
        interpretation = Interpretation(**state["interpretation"])
        verdict = await critic.critique(interpretation, state["transcript"])
        attempts = state.get("critic_attempts", 0) + 1
        span.set_attribute("critic_rejected", not verdict.approved)
        span.set_attribute("critic_attempts", attempts)
        return {"critic_verdict": verdict.model_dump(), "critic_attempts": attempts}


def _route_after_critic(state: MonologueState) -> str:
    verdict = state["critic_verdict"]
    if not verdict["approved"] and state["critic_attempts"] <= settings.max_critic_attempts:
        return "interpreter"
    return "question_generator"


async def _node_question_generator(state: MonologueState) -> MonologueState:
    with _tracer.start_as_current_span("graph.question_generator") as span:
        interpretation = Interpretation(**state["interpretation"])
        qa_turns = [QATurn(**t) for t in state.get("qa_turns", [])]
        question = await question_generator.generate_question(interpretation, qa_turns)
        span.set_attribute("proposed_question", bool(question))
        if question is None:
            return {"questions_done": True, "pending_question": None}
        return {"questions_done": False, "pending_question": question}


def _route_after_question_generator(state: MonologueState) -> str:
    if state.get("questions_done") or state.get("question_count", 0) >= settings.max_questions:
        return "report_synthesizer"
    return "ask_question"


async def _node_ask_question(state: MonologueState) -> MonologueState:
    question_count = state.get("question_count", 0)
    question = state["pending_question"]
    assert question is not None  # guarded by _route_after_question_generator
    answer = interrupt({"question": question, "turn": question_count + 1})
    qa_turns = [QATurn(**t) for t in state.get("qa_turns", [])]
    qa_turns.append(QATurn(turn_index=question_count + 1, question=question, answer=answer))
    return {
        "qa_turns": [t.model_dump() for t in qa_turns],
        "question_count": question_count + 1,
    }


async def _node_report_synthesizer(state: MonologueState) -> MonologueState:
    with _tracer.start_as_current_span("graph.report_synthesizer"):
        qa_turns = [QATurn(**t) for t in state.get("qa_turns", [])]
        report = await report_synthesizer.synthesize(state["transcript"], qa_turns)
        return {"report": report.model_dump(), "status": SessionStatus.COMPLETED.value}


def build_graph(checkpointer):
    graph = StateGraph(MonologueState)
    graph.add_node("mood_router", _node_mood_router)
    graph.add_node("safety_gate", _node_safety_gate)
    graph.add_node("crisis_response", _node_crisis_response)
    graph.add_node("interpreter", _node_interpreter)
    graph.add_node("critic", _node_critic)
    graph.add_node("question_generator", _node_question_generator)
    graph.add_node("ask_question", _node_ask_question)
    graph.add_node("report_synthesizer", _node_report_synthesizer)

    graph.add_edge(START, "mood_router")
    graph.add_edge("mood_router", "safety_gate")
    graph.add_conditional_edges(
        "safety_gate", _route_after_safety,
        {"crisis_response": "crisis_response", "interpreter": "interpreter"},
    )
    graph.add_edge("crisis_response", END)
    graph.add_edge("interpreter", "critic")
    graph.add_conditional_edges(
        "critic", _route_after_critic,
        {"interpreter": "interpreter", "question_generator": "question_generator"},
    )
    graph.add_conditional_edges(
        "question_generator", _route_after_question_generator,
        {"report_synthesizer": "report_synthesizer", "ask_question": "ask_question"},
    )
    graph.add_edge("ask_question", "question_generator")
    graph.add_edge("report_synthesizer", END)

    return graph.compile(checkpointer=checkpointer)


# --- Session-level helpers (used by the FastAPI routes) --------------------


def _config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


def _pending_question(snapshot) -> str | None:
    for task in snapshot.tasks:
        for intr in task.interrupts:
            return intr.value.get("question")
    return None


def session_response_from_snapshot(session_id: str, snapshot) -> SessionResponse | None:
    values = snapshot.values
    if not values:
        return None
    report = Report(**values["report"]) if values.get("report") else None
    qa_turns = [QATurn(**t) for t in values.get("qa_turns", [])]
    return SessionResponse(
        session_id=session_id,
        status=SessionStatus(values.get("status", SessionStatus.ACTIVE.value)),
        question=_pending_question(snapshot),
        crisis_message=values.get("crisis_message"),
        report=report,
        qa_turns=qa_turns,
    )


async def start_session(graph, session_id: str, mood_selection: str, transcript: str) -> SessionResponse:
    config = _config(session_id)
    initial_state = create_initial_state(session_id, mood_selection, transcript)
    await graph.ainvoke(initial_state, config=config)
    snapshot = await graph.aget_state(config)
    response = session_response_from_snapshot(session_id, snapshot)
    assert response is not None
    return response


async def resume_session(graph, session_id: str, answer: str) -> SessionResponse | None:
    config = _config(session_id)
    await graph.ainvoke(Command(resume=answer), config=config)
    snapshot = await graph.aget_state(config)
    return session_response_from_snapshot(session_id, snapshot)


async def get_session(graph, session_id: str) -> SessionResponse | None:
    snapshot = await graph.aget_state(_config(session_id))
    return session_response_from_snapshot(session_id, snapshot)
