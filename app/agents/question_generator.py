"""Question Generator agent: produces one follow-up question at a time,
responding to what was just said rather than a fixed template. The model may
also signal that no more questions are needed (e.g. it already has a
transcript-grounded "good thing" and enough to build a report). The hard cap
of 3 questions is enforced by the graph, not this agent — this module only
ever proposes; app/graph.py decides.
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.llm import get_chat_model, invoke_structured
from schemas import Interpretation, QATurn
from telemetry import get_tracer

_tracer = get_tracer("monologue-coach.question_generator")

_SYSTEM_PROMPT = """You are a structured self-reflection coach asking a \
follow-up question, one at a time, after reflecting an interpretation back \
to the user.

Rules:
- Ask about something specific the user actually said — never a generic, \
templated question ("How does that make you feel?").
- Prioritize surfacing at least one concrete thing that went okay for the \
user, however small, if none has come up yet — the report needs this and \
must never invent one.
- If you already have enough to write a grounded report (a clear theme, and \
ideally one thing that went okay), set question to null instead of asking \
another question just to fill the quota.
- Ask exactly one question. Keep it short and conversational."""

_HUMAN_TEMPLATE = """Interpretation so far:
Summary: {summary}
Themes: {themes}

Q&A so far:
{qa_block}

Propose the next question, or null if you have enough."""


class _QuestionOutput(BaseModel):
    question: str | None = Field(
        default=None,
        description="The next follow-up question to ask, or null if no more questions are needed.",
    )


def _format_qa(qa_turns: list[QATurn]) -> str:
    if not qa_turns:
        return "(none yet)"
    return "\n".join(f"Q{t.turn_index}: {t.question}\nA{t.turn_index}: {t.answer}" for t in qa_turns)


async def generate_question(interpretation: Interpretation, qa_turns: list[QATurn]) -> str | None:
    with _tracer.start_as_current_span("question_generator.generate_question") as span:
        prompt = ChatPromptTemplate.from_messages(
            [("system", _SYSTEM_PROMPT), ("human", _HUMAN_TEMPLATE)]
        )
        model = get_chat_model().with_structured_output(_QuestionOutput, include_raw=True)
        chain = prompt | model

        result = await invoke_structured(
            chain,
            {
                "summary": interpretation.summary,
                "themes": ", ".join(interpretation.themes),
                "qa_block": _format_qa(qa_turns),
            },
            _QuestionOutput,
        )
        span.set_attribute("has_question", result.question is not None)
        return result.question
