"""Unit tests for the deterministic Safety Gate keyword detector."""
from app.safety import detect_crisis

_CLEARLY_FINE = [
    "Work has been busy but I'm managing okay, just tired.",
    "I had a good weekend, went hiking with a friend and felt recharged.",
    "I'm a bit annoyed at my coworker for missing a deadline, that's really it.",
]

_CLEARLY_CONCERNING = [
    "I don't want to be alive anymore, I keep thinking about ending it all.",
    "Sometimes I think everyone would be better off without me.",
    "I've been thinking about how I would kill myself.",
    "I just want to end my life, there's no point anymore.",
    "I've been cutting myself when things get bad.",
    # Fully-spelled-out negation, not the contraction — a real gap an
    # earlier version of the pattern list missed (see safety.py's
    # _normalize docstring).
    "I do not want to be here anymore and I have been thinking about how I would do it.",
    "I can not go on like this anymore.",
]


def test_clean_transcripts_do_not_trigger():
    for text in _CLEARLY_FINE:
        verdict = detect_crisis(text)
        assert verdict.is_crisis is False, text


def test_crisis_phrases_trigger():
    for text in _CLEARLY_CONCERNING:
        verdict = detect_crisis(text)
        assert verdict.is_crisis is True, text
        assert verdict.matched_phrases


def test_verdict_reports_matched_phrases():
    verdict = detect_crisis("I want to kill myself.")
    assert verdict.is_crisis is True
    assert any("kill" in p for p in verdict.matched_phrases)


def test_empty_transcript_does_not_trigger():
    verdict = detect_crisis("")
    assert verdict.is_crisis is False


def test_contraction_and_expanded_negation_both_trigger():
    # Regression: "don't want to be here anymore" matched, but the
    # fully-spelled-out "do not want to be here anymore" silently didn't,
    # until detect_crisis started normalizing contractions first.
    contracted = detect_crisis("I don't want to be here anymore.")
    expanded = detect_crisis("I do not want to be here anymore.")
    assert contracted.is_crisis is True
    assert expanded.is_crisis is True
