"""Interpreter agent: reflects back an interpretation of the user's freeform
transcript. Must be strictly grounded — no invented facts, no clinical
labels, no personality-typing — and reflective/specific/tentative in tone.
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from app.llm import get_chat_model, invoke_structured
from schemas import Interpretation, MoodTag
from telemetry import get_tracer

_tracer = get_tracer("monologue-coach.interpreter")

_SYSTEM_PROMPT = """You are a structured self-reflection coach, not a \
therapist and not a diagnostic tool. The user just spoke freeform for a few \
minutes about how they're feeling. Reflect back an interpretation of what \
they said.

Rules:
- Every claim must be traceable to something the user actually said. Do not \
invent facts, events, or feelings they didn't express.
- Never use clinical labels, diagnoses, or personality-typing (no "you have \
anxiety", no MBTI/Enneagram-style typing).
- Tone: reflective, specific, and tentative ("it sounds like...", "it seems \
like..."), not generic pop-psychology filler.
- Identify 2-4 concrete themes actually present in what they said."""

_HUMAN_TEMPLATE = """The user selected mood: {mood_label}

Transcript:
{transcript}
{feedback_block}
Produce the interpretation now."""


async def interpret(transcript: str, mood_tag: MoodTag, feedback: list[str] | None = None) -> Interpretation:
    with _tracer.start_as_current_span("interpreter.interpret") as span:
        span.set_attribute("mood_label", mood_tag.label.value)

        feedback_block = ""
        if feedback:
            feedback_block = (
                "\nYour previous attempt was rejected for these reasons — "
                "regenerate without repeating them:\n- " + "\n- ".join(feedback) + "\n"
            )

        prompt = ChatPromptTemplate.from_messages(
            [("system", _SYSTEM_PROMPT), ("human", _HUMAN_TEMPLATE)]
        )
        model = get_chat_model().with_structured_output(Interpretation, include_raw=True)
        chain = prompt | model

        interpretation = await invoke_structured(
            chain,
            {
                "mood_label": mood_tag.label.value,
                "transcript": transcript,
                "feedback_block": feedback_block,
            },
            Interpretation,
        )
        span.set_attribute("themes", ",".join(interpretation.themes))
        return interpretation
