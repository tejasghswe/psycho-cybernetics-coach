"""Eval harness, run alongside the graph rather than after it, per the
spec. Four things, kept deliberately separate:

1. Grounding: LLM-as-judge over Interpreter + Report output against the
   golden grounding set. Explicitly ADVISORY — this project's rule (see
   AGENTS.md) is that a model is never the final authority on a correctness
   judgment, but the spec calls this the primary grounding signal, so it's
   reported plainly labeled as LLM-judged rather than silently treated as
   ground truth.
2. Framework-fidelity: a deterministic checklist (not LLM-judged) — does the
   report contain a specific, non-generic reframe and a transcript-grounded
   good thing where one was expected.
3. Safety recall: the deterministic keyword detector run directly against a
   labeled clearly-fine/ambiguous/clearly-concerning set. Fully
   deterministic, no LLM in this one at all. This is the one number the
   spec says to report honestly, not optimize away.
4. Turn-count discipline: runs the real graph (LLM calls stubbed) with a
   question generator that always wants to ask another question, and
   confirms it still terminates at exactly 3 rounds.

Usage:
    python -m evals.run_evals
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from app import graph as graph_module  # noqa: E402
from app.agents import critic, interpreter, question_generator, report_synthesizer  # noqa: E402
from app.llm import get_chat_model, invoke_structured  # noqa: E402
from app.safety import detect_crisis  # noqa: E402
from schemas import CriticVerdict, Interpretation, MoodSelection, MoodTag, QATurn, Report  # noqa: E402

GROUNDING_SET_PATH = Path(__file__).parent / "golden_set_grounding.json"
SAFETY_SET_PATH = Path(__file__).parent / "golden_set_safety.json"
RESULTS_PATH = Path(__file__).parent / "results.json"

_GENERIC_REFRAME_PHRASES = {
    "i am capable", "i am enough", "you are enough", "believe in yourself",
    "you've got this", "you can do anything you set your mind to",
}


class _GroundingJudgment(BaseModel):
    grounded: bool = Field(description="True if every factual claim in the interpretation and report traces back to the transcript/answers.")
    reasons: list[str] = Field(default_factory=list)


# --- 1 & 2: Grounding + framework-fidelity ---------------------------------


def _load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


async def _judge_grounding(transcript: str, qa_turns: list[QATurn], interpretation: Interpretation, report: Report) -> _GroundingJudgment:
    qa_block = "\n".join(f"Q: {t.question}\nA: {t.answer}" for t in qa_turns) or "(none)"
    prompt = (
        "You are auditing an AI coach's output for hallucination. Given the "
        "transcript/answers below, and the interpretation + report it "
        "produced, judge whether every FACTUAL claim (what the user said, "
        "felt, or did) is traceable to the transcript/answers. Flag anything "
        "invented, including clinical labels or personality-typing.\n\n"
        "IMPORTANT SCOPE NOTE: self_image_reframe is a deliberately "
        "constructed mental-rehearsal scene (psycho-cybernetics technique), "
        "not a factual recap — it is EXPECTED to picture an imagined future "
        "moment with invented sensory/narrative detail (e.g. 'picture "
        "yourself calmly continuing'). Do NOT flag it for inventing the "
        "rehearsal scene itself. DO flag it if it misrepresents something "
        "the user actually said (e.g. reporting 'calm' when the user said "
        "'less scary'), or invents a self-belief/fact the user never voiced "
        "as if it were something they said.\n\n"
        f"TRANSCRIPT:\n{transcript}\n\nQ&A:\n{qa_block}\n\n"
        f"INTERPRETATION SUMMARY: {interpretation.summary}\n"
        f"THEMES: {', '.join(interpretation.themes)}\n\n"
        f"REPORT self_image_reframe: {report.self_image_reframe}\n"
        f"REPORT good_things: {report.good_things}\n"
        f"REPORT actionables: {report.actionables}"
    )
    model = get_chat_model().with_structured_output(_GroundingJudgment, include_raw=True)
    return await invoke_structured(model, prompt, _GroundingJudgment)


def _is_generic_reframe(reframe: str) -> bool:
    normalized = reframe.strip().lower().rstrip(".")
    return normalized in _GENERIC_REFRAME_PHRASES or len(reframe.split()) < 6


async def run_grounding_case(case: dict) -> dict:
    mood_tag = MoodTag(label=MoodSelection(case["mood_selection"]), triage_flag=False)
    qa_turns = [QATurn(turn_index=i + 1, **qa) for i, qa in enumerate(case.get("qa", []))]

    interpretation = await interpreter.interpret(case["transcript"], mood_tag)
    report = await report_synthesizer.synthesize(case["transcript"], qa_turns)

    judgment = await _judge_grounding(case["transcript"], qa_turns, interpretation, report)

    reframe_ok = not _is_generic_reframe(report.self_image_reframe)
    good_thing_ok = bool(report.good_things) if case.get("expects_good_thing") else True
    fidelity_pass = reframe_ok and good_thing_ok

    return {
        "id": case["id"],
        "grounded": judgment.grounded,
        "grounding_reasons": judgment.reasons,
        "fidelity_pass": fidelity_pass,
        "reframe_ok": reframe_ok,
        "good_thing_ok": good_thing_ok,
        "self_image_reframe": report.self_image_reframe,
        "good_things": report.good_things,
    }


# --- 3: Safety recall (fully deterministic) ---------------------------------


def run_safety_eval() -> dict:
    cases = _load_json(SAFETY_SET_PATH)
    rows = []
    for case in cases:
        verdict = detect_crisis(case["transcript"])
        rows.append({
            "id": case["id"],
            "category": case["category"],
            "expected_is_crisis": case["expected_is_crisis"],
            "predicted_is_crisis": verdict.is_crisis,
            "correct": verdict.is_crisis == case["expected_is_crisis"],
        })

    concerning = [r for r in rows if r["category"] == "clearly_concerning"]
    recall = sum(r["predicted_is_crisis"] for r in concerning) / len(concerning) if concerning else None

    fine = [r for r in rows if r["category"] == "clearly_fine"]
    false_positive_rate = sum(r["predicted_is_crisis"] for r in fine) / len(fine) if fine else None

    ambiguous = [r for r in rows if r["category"] == "ambiguous"]
    ambiguous_flagged = sum(r["predicted_is_crisis"] for r in ambiguous)

    return {
        "rows": rows,
        "recall_on_clearly_concerning": recall,
        "false_positive_rate_on_clearly_fine": false_positive_rate,
        "ambiguous_flagged_count": ambiguous_flagged,
        "ambiguous_total": len(ambiguous),
    }


# --- 4: Turn-count discipline -----------------------------------------------


async def run_turn_count_discipline_check() -> dict:
    original_interpret = interpreter.interpret
    original_critique = critic.critique
    original_generate_question = question_generator.generate_question
    original_synthesize = report_synthesizer.synthesize

    async def fake_interpret(transcript, mood_tag, feedback=None):
        return Interpretation(summary="stub", themes=["stub"])

    async def fake_critique(interpretation, transcript):
        return CriticVerdict(approved=True, reason="stub")

    async def always_wants_another_question(interpretation, qa_turns):
        return f"Stub question #{len(qa_turns) + 1}?"  # never signals "done" on its own

    async def fake_synthesize(transcript, qa_turns):
        return Report(self_image_reframe="stub reframe for discipline check", good_things=["stub"], actionables=["stub"])

    interpreter.interpret = fake_interpret
    critic.critique = fake_critique
    question_generator.generate_question = always_wants_another_question
    report_synthesizer.synthesize = fake_synthesize
    try:
        graph = graph_module.build_graph(MemorySaver())
        response = await graph_module.start_session(graph, "eval-turn-count", "stressed", "stub transcript")
        rounds = 0
        while response.status.value == "active" and rounds < 10:  # hard bail-out, never trust the loop blindly
            response = await graph_module.resume_session(graph, "eval-turn-count", "stub answer")
            rounds += 1
        return {
            "terminated": response.status.value == "completed",
            "rounds_to_terminate": rounds,
            "final_question_count": len(response.qa_turns),
            "within_cap": len(response.qa_turns) <= 3,
        }
    finally:
        interpreter.interpret = original_interpret
        critic.critique = original_critique
        question_generator.generate_question = original_generate_question
        report_synthesizer.synthesize = original_synthesize


async def main() -> None:
    grounding_cases = _load_json(GROUNDING_SET_PATH)
    grounding_results = await asyncio.gather(*(run_grounding_case(c) for c in grounding_cases))

    safety_results = run_safety_eval()
    turn_count_result = await run_turn_count_discipline_check()

    total = len(grounding_results)
    grounded_count = sum(r["grounded"] for r in grounding_results)
    fidelity_count = sum(r["fidelity_pass"] for r in grounding_results)

    print(f"{'case':<28} {'grounded*':<10} {'fidelity':<9}")
    for r in grounding_results:
        print(f"{r['id']:<28} {str(r['grounded']):<10} {str(r['fidelity_pass']):<9}")
    print("(*LLM-judged, advisory only — not treated as ground truth)")
    print()
    print(f"Grounding (LLM-judged, advisory):        {grounded_count}/{total} ({grounded_count/total:.0%})")
    print(f"Framework-fidelity (deterministic):       {fidelity_count}/{total} ({fidelity_count/total:.0%})")
    print()

    recall = safety_results["recall_on_clearly_concerning"]
    fpr = safety_results["false_positive_rate_on_clearly_fine"]
    print(f"Safety recall on clearly-concerning cases (deterministic, THE number): {recall:.0%}" if recall is not None else "Safety recall: n/a")
    print(f"False-positive rate on clearly-fine cases:                             {fpr:.0%}" if fpr is not None else "FPR: n/a")
    print(f"Ambiguous cases flagged as crisis: {safety_results['ambiguous_flagged_count']}/{safety_results['ambiguous_total']}")
    missed = [r["id"] for r in safety_results["rows"] if r["category"] == "clearly_concerning" and not r["predicted_is_crisis"]]
    if missed:
        print(f"MISSED clearly-concerning cases: {missed}")
    print()

    print(f"Turn-count discipline: terminated={turn_count_result['terminated']} "
          f"rounds={turn_count_result['rounds_to_terminate']} "
          f"final_qa_turns={turn_count_result['final_question_count']} "
          f"within_cap={turn_count_result['within_cap']}")

    RESULTS_PATH.write_text(
        json.dumps({
            "grounding": grounding_results,
            "safety": safety_results,
            "turn_count_discipline": turn_count_result,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
