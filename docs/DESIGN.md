# Design notes

## Problem

See [`../requirements.md`](../requirements.md) for the full spec. In short:
a user talks freeform for a few minutes, the system reflects back a grounded
interpretation, asks up to 3 targeted follow-up questions one at a time, then
produces a short report with a psycho-cybernetics reframe, transcript-grounded
"good things," and concrete actionables. One hard, non-negotiable
requirement: a Safety Gate that must bias toward false positives on
crisis-language detection.

## Architecture decisions

### Persistence: LangGraph's checkpointer is the only session store

The spec's `Session`/`Turn` Pydantic models are explicitly marked "sketch,
refine during build." Rather than building a second, hand-rolled SQL schema
alongside LangGraph's own state persistence, `app/main.py` wires an
`AsyncSqliteSaver` (thread_id = session_id) as the graph's checkpointer, and
`app/graph.py`'s `session_response_from_snapshot` derives the API's
`SessionResponse` shape directly from `graph.aget_state()`. One source of
truth for session state, not two that can drift apart.

### Safety Gate: keyword/regex, not an LLM call

The spec calls this out as the one hard requirement, biased toward false
positives. `app/safety.py` implements it as a deterministic regex phrase
list — the same philosophy as the sibling `handoff_copilot` project's
`app/security.py`: an LLM should never be the thing deciding whether its own
(or another LLM's) output is safe, and a safety-critical gate benefits from
being auditable, testable, and immune to an off-day in model phrasing or an
API outage. The cost is recall limited to the phrasing families in the
pattern list.

One real bug found and fixed during eval-writing: an early version of the
pattern for "don't want to be here anymore" only matched the contraction,
and silently missed the fully-spelled-out "do not want to be here anymore"
— a very common real phrasing. Fixed by normalizing negation contractions
(`don't`/`do not`, `can't`/`cannot`/`can not`, etc.) to one canonical form
before matching, so every pattern only needs to spell out one form. See
`app/safety.py`'s `_normalize` and the regression test in
`tests/test_safety.py`.

### The question loop is two nodes, not one, because of how `interrupt()` replays

LangGraph re-runs a node's function from the top on resume, up to its
already-resolved `interrupt()` call. If the LLM call that decides the next
question lived in the same node as the `interrupt()` call, that LLM call
would silently re-run on every resume — potentially generating a *different*
question than the one the user actually answered, while the code still
labels the stored turn with the original (now-stale) question text.

`app/graph.py` splits this into `question_generator` (LLM call; decides the
next question or that none is needed; no `interrupt()`) and `ask_question`
(nothing but a state read and `interrupt()`; idempotent to re-run). The LLM
call is checkpointed once and never re-executed on resume.

The 3-question cap is enforced by `_route_after_question_generator` in
plain Python, checking `question_count`, never left to the model to
self-limit — verified in `tests/test_graph.py` and the "turn-count
discipline" eval.

### Structured LLM output needed a retry/recovery layer, not just Pydantic validation

Discovered via the grounding eval, not assumed upfront: `with_structured_output`
occasionally returns malformed tool-call output in ways Pydantic's shape
validation doesn't catch on its own — a list field serialized as
tag-delimited text (passes validation, garbage content), the whole field set
wrapped in an extra `"parameters"` key (fails validation outright), or a
field simply missing. `app/llm.py`'s `invoke_structured` uses
`include_raw=True` to intercept these instead of letting the first one
crash the graph node: it retries (bounded), and for the `"parameters"`-wrapped
shape specifically, re-validates the inner dict directly rather than
discarding a usable response. `report_synthesizer.py`'s `_looks_malformed`
adds a second, content-level check (tag leakage, or a suspiciously long
reframe with empty lists) for the case that passes Pydantic but is still
garbage.

### No SSE/streaming

The spec's frontend is explicitly "disposable" with no streaming
requirement (unlike the sibling `handoff_copilot` project's `/handoff` SSE
endpoint). `frontend/app.js` is a plain request/response poll: `POST
/sessions`, `POST /sessions/{id}/answer`, `GET /sessions/{id}`.

## Eval results (see `evals/run_evals.py`, run against the real Anthropic API)

- Grounding (LLM-judged, advisory): 14/15 on the current golden set. The
  judge is scoped to flag factual misrepresentation, not the inherent
  invented-detail nature of the psycho-cybernetics rehearsal scene itself —
  see the judge prompt's scope note. The one remaining failure is a genuine,
  subtle finding (a causal detail attributed to the user that they didn't
  state) rather than a crash or a judge-calibration issue.
- Framework-fidelity (deterministic): 15/15.
- Safety recall on clearly-concerning cases (deterministic): 100/100%, 0%
  false-positive rate on clearly-fine cases, 0/10 ambiguous cases flagged.
  Reported as measured, not tuned to hit a number.
- Turn-count discipline: terminates at exactly 3 rounds even when the model
  always wants to ask another question.

## Known limitations

- Keyword-only Safety Gate: will miss crisis language outside its pattern
  families (heavily idiomatic or highly indirect phrasing). Biased toward
  over-triggering by design, per the spec.
- Single Anthropic API key, single-process SQLite checkpointer file — v0
  single-session scope only, not multi-instance production.
- `with_structured_output` reliability is mitigated (retry + targeted
  `"parameters"`-unwrap recovery) but not eliminated; after all retries are
  exhausted the session surfaces an error rather than fabricated content.
