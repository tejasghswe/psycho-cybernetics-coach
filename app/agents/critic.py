"""Critic agent: checks the Interpreter's output before it's shown to the
user or used downstream. Two things, both LLM-judged since they're about
prose quality/faithfulness rather than a bright-line rule: is every claim
traceable to the transcript, and is the tone non-judgmental and
appropriately tentative rather than generic pop-psychology.
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from app.llm import get_chat_model, invoke_structured
from schemas import CriticVerdict, Interpretation
from telemetry import get_tracer

_tracer = get_tracer("monologue-coach.critic")

_SYSTEM_PROMPT = """You are a strict editor reviewing an AI coach's \
interpretation of a user's freeform transcript, before it's shown to them.

Reject (approved=false) if ANY of these hold:
- The interpretation states something as fact that isn't traceable to the \
transcript.
- It uses a clinical label, diagnosis, or personality-typing.
- The tone is judgmental, or reads as generic pop-psychology rather than \
specific to what this user actually said.
- The tone isn't appropriately tentative (should read like "it sounds \
like...", not flat assertions).

Otherwise approve. Give a one-sentence reason either way."""

_HUMAN_TEMPLATE = """Original transcript:
{transcript}

Interpretation to review:
Summary: {summary}
Themes: {themes}

Give your verdict now."""


async def critique(interpretation: Interpretation, transcript: str) -> CriticVerdict:
    with _tracer.start_as_current_span("critic.critique") as span:
        prompt = ChatPromptTemplate.from_messages(
            [("system", _SYSTEM_PROMPT), ("human", _HUMAN_TEMPLATE)]
        )
        model = get_chat_model().with_structured_output(CriticVerdict, include_raw=True)
        chain = prompt | model

        verdict = await invoke_structured(
            chain,
            {
                "transcript": transcript,
                "summary": interpretation.summary,
                "themes": ", ".join(interpretation.themes),
            },
            CriticVerdict,
        )
        span.set_attribute("approved", verdict.approved)
        return verdict
