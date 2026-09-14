"""Deterministic Safety Gate. Crisis detection is intentionally rule-based,
not model-judged: this is the one hard, non-negotiable requirement in the
spec, and an LLM call is both a reliability risk (latency, API failure, an
off-day on phrasing) and an unauditable one for a safety-critical gate. Bias
is deliberately toward false positives (broad phrase matching) over false
negatives, per the spec.

This mirrors the existing repo convention (see the sibling handoff_copilot
project's app/security.py) of keeping safety/security decisions in
deterministic, testable code rather than delegating them to a model.
"""
from __future__ import annotations

import re

from schemas import SafetyVerdict

# Negation contractions are normalized to a canonical two-word form before
# matching (see _normalize) so every pattern below only has to spell out one
# form. Discovered the hard way: an earlier version of the "don't want to be
# here anymore" pattern only matched the contraction "don't", and silently
# missed the fully-spelled-out "do not want to be here anymore" — a common,
# textbook crisis phrasing. Given this gate's bias-toward-false-positives
# requirement, catching only one grammatical form of a phrase is a real
# recall gap, not a cosmetic one.
_CONTRACTION_EXPANSIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bdon'?t\b", re.I), "do not"),
    (re.compile(r"\bcan'?t\b", re.I), "can not"),
    (re.compile(r"\bcannot\b", re.I), "can not"),
    (re.compile(r"\bwon'?t\b", re.I), "will not"),
    (re.compile(r"\bwouldn'?t\b", re.I), "would not"),
]


def _normalize(text: str) -> str:
    for pattern, replacement in _CONTRACTION_EXPANSIONS:
        text = pattern.sub(replacement, text)
    return text


# Deliberately broad. Each pattern is meant to catch a family of phrasings,
# not just one exact string — over-triggering here is the accepted tradeoff.
_CRISIS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bkill(ing)? myself\b", re.I),
    re.compile(r"\bsuicid\w*\b", re.I),
    re.compile(r"\bend(ing)? (my|it all)\b", re.I),
    re.compile(r"\bend my life\b", re.I),
    re.compile(r"\bnot (be|being) (here|alive) anymore\b", re.I),
    re.compile(r"\bdo not want to (be alive|be here|live|wake up|exist) anymore\b", re.I),
    re.compile(r"\bwish(ed|ing)? i (was|were) dead\b", re.I),
    re.compile(r"\bbetter off (dead|without me)\b", re.I),
    re.compile(r"\beveryone('s| is)? better off without me\b", re.I),
    re.compile(r"\bno (point|reason) (in|to) (living|going on|continuing)\b", re.I),
    re.compile(r"\bno point anymore\b", re.I),
    re.compile(r"\bself.?harm\w*\b", re.I),
    re.compile(r"\bhurt(ing)? myself\b", re.I),
    re.compile(r"\bcut(ting)? myself\b", re.I),
    re.compile(r"\bwant to die\b", re.I),
    re.compile(r"\bplan to (kill|hurt|harm) myself\b", re.I),
    re.compile(r"\bcan not (go on|keep going) (like this )?anymore\b", re.I),
    re.compile(r"\bgive up on (life|everything)\b", re.I),
    re.compile(r"\boverdose\b", re.I),
    re.compile(r"\ball (my|the) pills\b", re.I),
    re.compile(r"\bjump(ing)? (off|in front of)\b", re.I),
]

CRISIS_RESOURCES_MESSAGE = (
    "It sounds like you might be going through something really painful right "
    "now, and I want to make sure you get support that's actually equipped for "
    "this — more than I am.\n\n"
    "If you're in the US, you can call or text 988 (Suicide & Crisis Lifeline), "
    "available 24/7. If you're outside the US, the International Association "
    "for Suicide Prevention maintains a list of crisis centers at "
    "https://www.iasp.info/resources/Crisis_Centres/. If you're in immediate "
    "danger, please contact local emergency services.\n\n"
    "You don't have to go through this alone, and reaching out is a strong, "
    "reasonable thing to do."
)


def detect_crisis(text: str) -> SafetyVerdict:
    normalized = _normalize(text)
    matched = sorted({m.group(0) for p in _CRISIS_PATTERNS if (m := p.search(normalized))})
    if matched:
        return SafetyVerdict(
            is_crisis=True,
            reason=f"Matched crisis-language pattern(s): {', '.join(matched)}",
            matched_phrases=matched,
        )
    return SafetyVerdict(is_crisis=False, reason="No crisis-language patterns matched.")
