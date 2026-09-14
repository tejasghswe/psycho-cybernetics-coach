"""Shared Pydantic contracts used by the graph nodes, the FastAPI routes, and
the eval harness. `Session`/`Turn`-shaped API responses are assembled in code
from LangGraph checkpoint state (see app/graph.py) rather than persisted as a
separate hand-rolled table — the checkpointer is the single source of truth
for session state, so there is exactly one place that can drift from reality.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class MoodSelection(str, Enum):
    STRESSED = "stressed"
    SAD = "sad"
    ANNOYED = "annoyed"
    BETTER = "better"
    OTHER = "other"


class SessionStatus(str, Enum):
    ACTIVE = "active"
    CRISIS_TERMINATED = "crisis_terminated"
    COMPLETED = "completed"


class MoodTag(BaseModel):
    """Mood Router output: the user's own mood label plus a cheap triage
    flag (distress-intensity heuristic on the raw transcript, not a
    diagnosis)."""

    label: MoodSelection
    triage_flag: bool


class SafetyVerdict(BaseModel):
    is_crisis: bool
    reason: str
    matched_phrases: list[str] = Field(default_factory=list)


class Interpretation(BaseModel):
    summary: str
    themes: list[str] = Field(default_factory=list)


class CriticVerdict(BaseModel):
    approved: bool
    reason: str


class QATurn(BaseModel):
    turn_index: int
    question: str
    answer: str


class Report(BaseModel):
    self_image_reframe: str = Field(
        description=(
            "A specific, rehearsable success-image instruction reframing one "
            "self-belief the user voiced — a mental-rehearsal instruction, "
            "not a generic affirmation."
        )
    )
    good_things: list[str] = Field(
        default_factory=list,
        description="1-3 things the user mentioned that went okay, pulled directly from the transcript/answers.",
    )
    actionables: list[str] = Field(
        default_factory=list,
        description="1-3 concrete, specific actionables for today/this week.",
    )


# --- API request/response models ----------------------------------------


class CreateSessionRequest(BaseModel):
    mood_selection: MoodSelection
    transcript: str = Field(min_length=1, max_length=8000)


class AnswerRequest(BaseModel):
    answer: str = Field(min_length=1, max_length=2000)


class SessionResponse(BaseModel):
    session_id: str
    status: SessionStatus
    question: str | None = None
    crisis_message: str | None = None
    report: Report | None = None
    qa_turns: list[QATurn] = Field(default_factory=list)
