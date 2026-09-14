# Monologue Coach — Multi-Agent MVP Build Spec

## Problem

User talks freeform for ~3 minutes about how they're feeling (like an
unstructured coaching call). The system reflects back an interpretation,
asks up to 3 targeted follow-up questions (one at a time, waiting for the
user's answer each time), then produces a short report with 1-3 concrete
actionables grounded in named psychological frameworks.

Not a therapist, not a diagnostic tool. A structured self-reflection coach.

## Non-goals for this MVP (explicitly deferred)

- Voice input/output (ASR/TTS) — text only for v0
- Long-term memory / cross-session context (Pinecone) — single session only
- Go services (safety-gate isolation, Temporal durability) — pure Python
- Model routing across providers — one LLM provider is fine
- Any clinical language, diagnosis, or personality-typing beyond descriptive
  reflection of what was said

## Tech stack (v0)

- FastAPI (backend, owns all session state and orchestration)
- Pydantic (all agent I/O schemas — no unstructured dict passing between nodes)
- LangGraph (the agent graph below)
- LangChain (LLM client wrappers only, not the orchestration layer)
- SQLite or Postgres (session/turn persistence)
- OpenTelemetry (trace every graph node; span attribute for "critic rejected: y/n")
- Simple frontend: plain HTML/JS or Streamlit, calling the FastAPI backend.
  All orchestration logic lives in the backend — the frontend is disposable.

## Agent graph (LangGraph)

Linear pipeline with one conditional branch and one bounded retry loop:

```
Mood Router
  -> Safety Gate
       -> [if crisis language detected] -> Crisis Response (terminal, no coaching)
       -> [else] -> Interpreter
                       -> Critic
                            -> [if rejected, max 1 retry] -> back to Interpreter
                            -> [if approved] -> Question Generator
                                                  -> (human-in-the-loop, interrupt(),
                                                     up to 3 rounds)
                                                  -> Report Synthesizer -> END
```

### Node specs

**Mood Router**
- Input: user's dropdown selection (stressed / sad / annoyed / better / other) + raw transcript
- Output: `MoodTag`, light triage flag
- Cheap/fast; doesn't need a large model

**Safety Gate** (non-negotiable, do not skip or simplify)
- Input: raw transcript
- Output: `SafetyVerdict { is_crisis: bool, reason: str }`
- If `is_crisis == True`: graph terminates at Crisis Response — no coaching flow, no
  follow-up questions, no report. Response shows calm acknowledgment + crisis resources
  only.
- Bias toward false positives (over-triggering) over false negatives. This is a hard
  requirement, not a tuning preference.

**Interpreter**
- Input: transcript, mood tag
- Output: `Interpretation { summary: str, themes: list[str] }`
- Must be strictly grounded in the transcript. No invented facts, no clinical labels,
  no personality-typing. Reflective, specific, tentative in tone.

**Critic**
- Input: `Interpretation` + original transcript
- Output: `CriticVerdict { approved: bool, reason: str }`
- Checks: is every claim traceable to the transcript? Is the tone non-judgmental and
  appropriately tentative (not generic pop-psychology)?
- On rejection: send back to Interpreter once with the rejection reason. Second
  rejection: approve with the reason logged (don't loop forever).

**Question Generator**
- Input: approved `Interpretation`, prior Q&A turns (if any)
- Output: one follow-up question at a time (max 3 total)
- Uses LangGraph `interrupt()` to pause and wait for the user's answer before
  generating the next question (or deciding no more are needed).
- Questions must respond to what was just said, not be templated/generic.

**Report Synthesizer**
- Input: full transcript + all Q&A turns
- Output: `Report { self_image_reframe: str, good_things: list[str], actionables: list[str] }`
- Two required, structured fields (not free-text "try journaling" filler):
  - **Psycho-cybernetics reframe**: identify one self-belief the user voiced
    (e.g. "I always mess up presentations") and reframe it as a specific,
    rehearsable "success image" — a concrete mental-rehearsal instruction, not
    an affirmation.
  - **Three Good Things**: 1-3 things the user mentioned that went okay,
    however small, pulled directly from the transcript/answers. If nothing
    surfaced naturally, this is a signal the Question Generator should have
    fished for one — don't invent one here.
- 1-3 actionables for today/this week, concrete and specific to what was said.

## Data models (Pydantic — sketch, refine during build)

```python
class Session(BaseModel):
    id: str
    mood_tag: str
    status: Literal["active", "crisis_terminated", "completed"]

class Turn(BaseModel):
    session_id: str
    role: Literal["user", "question"]
    content: str
    turn_index: int

class SafetyVerdict(BaseModel):
    is_crisis: bool
    reason: str

class Interpretation(BaseModel):
    summary: str
    themes: list[str]

class CriticVerdict(BaseModel):
    approved: bool
    reason: str

class Report(BaseModel):
    self_image_reframe: str
    good_things: list[str]
    actionables: list[str]
```

## Evals (build these alongside the graph, not after)

1. **Grounding eval**: LLM-as-judge checks every claim in `Interpretation` and
   `Report` traces back to the transcript/answers. Build a small hand-written
   labeled set (~20 fake transcripts with known "true" content) to test against.
2. **Framework-fidelity check**: binary checklist — does the report contain a
   specific self-image reframe and a specific transcript-grounded "good thing,"
   or generic filler?
3. **Safety recall**: hand-write ~30 test transcripts (clearly-fine,
   ambiguous, clearly-concerning) and measure Safety Gate recall on the
   concerning ones. This is the one number to report honestly, not optimize
   away.
4. **Turn-count discipline**: verify the graph terminates cleanly at 3
   follow-ups and doesn't loop indefinitely on Critic rejections.

## Observability

- OpenTelemetry span per graph node
- Span attributes: node name, latency, `critic_rejected: bool`,
  `safety_triggered: bool`
- This is a real artifact for the portfolio — graph how often the critic
  rejects, how often the safety gate fires, latency per node.

## Build order

1. Pydantic schemas + FastAPI skeleton (session create, submit turn, get report)
2. LangGraph graph with all nodes as stubs returning hardcoded data — prove the
   state machine (including the interrupt/human-in-loop follow-up loop and the
   critic retry-once logic) works before wiring real LLM calls
3. Wire Interpreter + Critic, iterate on grounding until the eval passes
4. Wire Safety Gate, build and run the crisis-recall test set before doing
   anything else
5. Wire Question Generator + Report Synthesizer
6. Add OpenTelemetry tracing
7. Thin frontend on top

Defer everything in "Non-goals" until this is solid and evaluated.