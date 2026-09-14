"""Report Synthesizer agent: produces the final structured report from the
full transcript plus all Q&A turns. Two required, specific fields (not
free-text filler) grounded in named frameworks:

- Psycho-cybernetics reframe: one self-belief the user voiced, reframed as a
  concrete, rehearsable "success image" — a mental-rehearsal instruction,
  not an affirmation.
- Three Good Things: 1-3 things that went okay, pulled directly from what
  the user said. If nothing surfaced, that's a signal the Question
  Generator should have fished for one — this agent must not invent one.
"""
from __future__ import annotations

import re

from langchain_core.prompts import ChatPromptTemplate

from app.llm import get_chat_model, invoke_structured
from schemas import QATurn, Report
from telemetry import get_tracer

_TAG_LEAK_PATTERN = re.compile(r"</?\w+>")

_tracer = get_tracer("monologue-coach.report_synthesizer")

_SYSTEM_PROMPT = """You are a structured self-reflection coach writing the \
final report after a short conversation with the user.

Produce:
1. self_image_reframe: Identify ONE specific self-belief the user voiced \
(e.g. "I always mess up presentations"). Reframe it per psycho-cybernetics \
as a concrete, rehearsable "success image" — a specific mental-rehearsal \
instruction the user can practice (e.g. "Before your next presentation, \
spend two minutes picturing yourself pausing calmly after a stumble and \
continuing"), NOT a generic affirmation like "I am capable." The imagined \
future moment itself may include invented sensory/narrative detail (that's \
the point of a rehearsal scene) — but never invent a REASON, MECHANISM, or \
PAST EXPERIENCE to explain something the user only stated as an outcome \
(e.g. if they said a report "went well," don't claim they "started it \
early" unless they said that), and never swap in a stronger or different \
feeling word than the one they actually used (e.g. they said "less scary" — \
don't upgrade that to "calm" or "settled").
2. good_things: 1-3 things the user mentioned that went okay, however \
small, quoted or closely paraphrased from what they actually said. If \
nothing like this appears anywhere in the transcript or answers, return an \
empty list — never invent one.
3. actionables: 1-3 concrete, specific actions for today/this week, tied to \
what was actually discussed. No generic filler like "try journaling" unless \
it's specifically tied to something the user said.

Ground every field strictly in the transcript and answers below."""

_HUMAN_TEMPLATE = """Transcript:
{transcript}

Q&A:
{qa_block}

Produce the report now."""


def _format_qa(qa_turns: list[QATurn]) -> str:
    if not qa_turns:
        return "(no follow-up questions were asked)"
    return "\n".join(f"Q{t.turn_index}: {t.question}\nA{t.turn_index}: {t.answer}" for t in qa_turns)


def _looks_malformed(report: Report) -> bool:
    """Pydantic guarantees the right shape, not that the content is actually
    what it claims to be. A truncated/mis-parsed structured-output call has
    been observed (via the grounding eval, see evals/run_evals.py) to spill
    other fields' content — including literal closing tags — into
    self_image_reframe while leaving good_things/actionables empty. Catch
    that class of failure so it can be retried instead of silently shipped.
    """
    if _TAG_LEAK_PATTERN.search(report.self_image_reframe):
        return True
    return len(report.self_image_reframe) > 700 and not report.good_things and not report.actionables


async def synthesize(transcript: str, qa_turns: list[QATurn]) -> Report:
    with _tracer.start_as_current_span("report_synthesizer.synthesize") as span:
        prompt = ChatPromptTemplate.from_messages(
            [("system", _SYSTEM_PROMPT), ("human", _HUMAN_TEMPLATE)]
        )
        model = get_chat_model().with_structured_output(Report, include_raw=True)
        chain = prompt | model
        inputs = {"transcript": transcript, "qa_block": _format_qa(qa_turns)}

        report = await invoke_structured(chain, inputs, Report)
        malformed = _looks_malformed(report)
        span.set_attribute("malformed_output_detected", malformed)
        if malformed:
            # One bounded retry, same inputs — a fresh sample is a different
            # generation, not a repeat of the same failure.
            report = await invoke_structured(chain, inputs, Report)

        span.set_attribute("good_things_count", len(report.good_things))
        span.set_attribute("actionables_count", len(report.actionables))
        return report
