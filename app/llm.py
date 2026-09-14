"""Thin, single-place wrapper around the chat model used by every agent node.
Centralizing this makes it trivial to swap models (e.g. a cheaper one for
evals) without touching agent logic.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, TypeVar

from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ValidationError

from app.config import settings

logger = logging.getLogger(__name__)

_T = TypeVar("_T", bound=BaseModel)


@lru_cache(maxsize=4)
def get_chat_model(model: str | None = None) -> ChatAnthropic:
    # Note: `temperature` is intentionally omitted — newer Claude models
    # (e.g. claude-sonnet-5) reject it as a deprecated parameter.
    return ChatAnthropic(
        model=model or settings.anthropic_model,
        anthropic_api_key=settings.anthropic_api_key,
        # 2048 rather than a tighter budget: the Report Synthesizer's
        # self_image_reframe is a multi-sentence rehearsal script and a
        # truncated tool call has been observed to spill unparsed content
        # into a single field instead of failing cleanly (see
        # report_synthesizer._looks_malformed).
        max_tokens=2048,
    )


async def invoke_structured(chain: Runnable, inputs: Any, schema: type[_T], *, retries: int = 2) -> _T:
    """Run a `model.with_structured_output(schema, include_raw=True)` chain
    (piped after a prompt, or called bare), retrying on malformed tool-call
    output instead of letting it reach the caller as an unhandled exception
    or, worse, silently-wrong data.

    `include_raw=True` is required on the chain's structured-output step:
    rather than letting langchain raise on a parse failure, it returns
    {"raw": AIMessage, "parsed": schema | None, "parsing_error": Exception |
    None}, which lets this function attempt one targeted recovery before
    giving up. Two distinct malformations have been observed in practice
    (see evals/run_evals.py's grounding eval and report_synthesizer's
    _looks_malformed):
      1. A list-typed field emitted as tag-delimited text instead of JSON.
      2. The whole set of fields wrapped in an extra top-level "parameters"
         key, as if the tool schema were OpenAI-function-calling-shaped —
         recovered here by re-validating against that inner dict.
    Per AGENTS.md, a tool/parsing failure is an expected failure mode worth
    a bounded retry, not a reason to crash the session — a fresh generation
    is a different sample, not a repeat of the same malformed one.
    """
    last_error: BaseException | None = None
    for attempt in range(retries + 1):
        result = await chain.ainvoke(inputs)
        parsed = result.get("parsed")
        if parsed is not None:
            return parsed

        raw = result.get("raw")
        tool_calls = getattr(raw, "tool_calls", None) or []
        if tool_calls:
            args = tool_calls[0].get("args", {})
            wrapped = args.get("parameters")
            if isinstance(wrapped, dict):
                try:
                    return schema.model_validate(wrapped)
                except ValidationError:
                    pass

        last_error = result.get("parsing_error")
        logger.warning(
            "Structured output for %s failed to parse (attempt %d/%d): %s",
            schema.__name__, attempt + 1, retries + 1, last_error,
        )

    if last_error is not None:
        raise last_error
    raise RuntimeError(
        f"Structured output for {schema.__name__} failed after {retries + 1} attempts "
        "with no parsing_error captured."
    )
