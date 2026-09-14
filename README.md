# Psycho-cybernetics coach

A structured self-reflection coach: the user talks freeform for a few
minutes about how they're feeling, a small multi-agent system reflects back
a grounded interpretation, asks up to three targeted follow-up questions one
at a time, then produces a short report — a psycho-cybernetics-style
"success image" reframe, 1-3 things that went okay, and 1-3 concrete
actionables. Not a therapist, not a diagnostic tool.

See [`requirements.md`](requirements.md) for the full build spec and
[`docs/DESIGN.md`](docs/DESIGN.md) for architecture decisions, the graph
design, and known limitations. This README covers setup and day-to-day use.

## Architecture

```
POST /sessions {mood, transcript}
        │
        ▼
  mood_router -> safety_gate -> [crisis] -> crisis_response -> END
                              -> [else]  -> interpreter -> critic
                                               (reject -> interpreter, max 1 retry)
                                            -> question_generator <-> ask_question
                                               (LLM proposes a question; ask_question
                                                is the only node that calls
                                                interrupt(), up to 3 rounds)
                                            -> report_synthesizer -> END
```

- **Safety Gate**: deterministic keyword/regex crisis-language detector, not
  an LLM call — see `app/safety.py` and [`docs/DESIGN.md`](docs/DESIGN.md)
  for why.
- **Interpreter / Critic**: LangChain (`ChatAnthropic` + `with_structured_output`)
  producing a grounded interpretation, checked by a critic that can send it
  back for one bounded retry.
- **Question Generator**: proposes one follow-up question at a time (or
  signals it has enough), with LangGraph's `interrupt()` pausing the graph
  for the user's answer — resumed via `Command(resume=answer)`. The 3-question
  cap is enforced in plain Python, never trusted to the model.
- **Persistence**: LangGraph's `AsyncSqliteSaver` (thread_id = session_id) is
  the single source of truth for session state — no separate hand-rolled
  session table.
- **OpenTelemetry**: a span per graph node, with `critic_rejected` and
  `safety_triggered` span attributes.

## Setup

```bash
cd "Monologue Coach"
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Set `ANTHROPIC_API_KEY` in the `.env` file at the repo root (one level up
from this directory).

## Running it

```bash
python -m uvicorn app.main:app --reload --port 8000
```

Open http://127.0.0.1:8000/ for the demo UI, or use the API directly (docs
at http://127.0.0.1:8000/docs):

```bash
curl -X POST http://127.0.0.1:8000/sessions -H "Content-Type: application/json" -d '{
  "mood_selection": "stressed",
  "transcript": "I have a big presentation Thursday and keep replaying every time I messed one up before."
}'
# -> {"session_id": "...", "status": "active", "question": "..."}

curl -X POST http://127.0.0.1:8000/sessions/<session_id>/answer -H "Content-Type: application/json" -d '{
  "answer": "Actually the last one I gave went fine, I just don'"'"'t remember that part."
}'
# -> repeat up to 3 times; the final response has status "completed" and a report

curl http://127.0.0.1:8000/sessions/<session_id>   # current state, without advancing
```

## Tests

```bash
python -m pytest -q
```

20 tests, all offline — every LLM call is stubbed via `monkeypatch`
(`tests/test_graph.py` for the orchestration/interrupt-loop logic,
`tests/test_sessions_router.py` for the HTTP API, `tests/test_safety.py` for
the deterministic crisis-language detector).

The frontend's voice-input state machine (`frontend/voice-input.js`) has its
own unit tests, using Node's built-in test runner — no browser, microphone,
or extra dependency required:

```bash
node --test frontend/voice-input.test.js
```

11 tests, using a fake `SpeechRecognition` and fake DOM elements to verify
the record/stop state machine (start/stop/abort handling, no stuck states on
rapid clicks or errors, transcribed text landing in the textarea).

## Evals

```bash
python -m evals.run_evals
```

Runs the real Interpreter/Report Synthesizer (real Anthropic calls) against
`evals/golden_set_grounding.json` (15 synthetic transcripts) and the
deterministic Safety Gate against `evals/golden_set_safety.json` (31
labeled clearly-fine / ambiguous / clearly-concerning transcripts). Reports
four numbers, kept deliberately separate:

- **Grounding** (LLM-as-judge, explicitly advisory — never treated as ground
  truth) — does every factual claim in the interpretation/report trace back
  to the transcript/answers.
- **Framework-fidelity** (deterministic) — does the report contain a
  specific, non-generic reframe and a transcript-grounded good thing where
  one was expected.
- **Safety recall on clearly-concerning cases** (fully deterministic, no LLM
  involved) — the one number to report honestly, not optimize away.
- **Turn-count discipline** — confirms the graph terminates at exactly 3
  questions even when the model always wants to ask another one.

This harness earned its keep during development: it caught a real
`with_structured_output` parsing failure (a list field serialized as
tag-delimited text, and separately the whole payload wrapped in an extra
`"parameters"` key) that Pydantic validation alone didn't catch until a
report's `good_things`/`actionables` silently came back empty. See
`app/llm.py`'s `invoke_structured` and `docs/DESIGN.md`.

## Known limitations / assumptions

- **Safety Gate is keyword/regex-based, not ML/LLM-based** — a deliberate
  choice for auditability and reliability on the one hard safety
  requirement in the spec, at the cost of missing crisis language outside
  the pattern list's phrasing families. See `docs/DESIGN.md`.
- **Single LLM provider** (Anthropic) and **single-process, in-file SQLite**
  checkpointer — fine for the spec's "single session only" v0 scope, not a
  multi-instance production deployment.
- **`with_structured_output` occasionally still fails all retries** on a
  small fraction of calls (observed during eval runs); `invoke_structured`
  retries and attempts a targeted recovery, then raises — the session
  surfaces a 500 rather than silently returning fabricated content.
