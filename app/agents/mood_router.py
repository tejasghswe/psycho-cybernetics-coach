"""Mood Router: cheap, deterministic — the user already told us their mood
via the dropdown, so this node just carries that label through and adds a
light triage flag from a small intensity-word heuristic on the raw
transcript. No LLM call needed.
"""
from __future__ import annotations

import re

from schemas import MoodSelection, MoodTag

_INTENSITY_PATTERNS = [
    re.compile(r"\bcan'?t (stop|handle|cope|breathe)\b", re.I),
    re.compile(r"\boverwhelm(ed|ing)?\b", re.I),
    re.compile(r"\bpanic(king|ked)?\b", re.I),
    re.compile(r"\bbreaking down\b", re.I),
    re.compile(r"\bfalling apart\b", re.I),
    re.compile(r"\bcrying\b", re.I),
    re.compile(r"\bhopeless\b", re.I),
    re.compile(r"\bexhaust(ed|ing)\b", re.I),
]


def route_mood(mood_selection: MoodSelection, transcript: str) -> MoodTag:
    triage_flag = any(p.search(transcript) for p in _INTENSITY_PATTERNS)
    return MoodTag(label=mood_selection, triage_flag=triage_flag)
