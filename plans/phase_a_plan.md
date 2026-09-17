# Phase A — The Harness

**Status:** not started
**Runs on:** your laptop. No GPU in this phase, and none needed.
**Read first:** [`../docs/harness-premise.md`](../docs/harness-premise.md)
**Not in scope here:** [`../docs/integration-plan.md`](../docs/integration-plan.md) — the backend
cutover is deferred. Reading the backend for *inventory* (step 2) is in scope; editing it is not.

---

## Where Phase A sits

The project is four phases. Phase A is steps 1–6.

| Phase | Steps | What it produces | Where |
|---|---|---|---|
| **A — Harness** | 1–6 | an episode that runs end to end against a frontier model, calling real tools, with the format frozen | local |
| B — Signal | 7–9 | eval set, verifier, baseline numbers | local |
| C — Data | 10 | rejection-sampled teacher trajectories | local |
| D — Training | 11–13 | a tuned LoRA and its eval | rented GPU |

**Phase A is done when:** you can run one query end to end, locally, through `run_episode`
against a frontier model; it really calls real tools against real infrastructure; and
`build_context` has a committed golden snapshot that fails loudly if the prompt format moves.

Phase A deliberately does **not** include: the eval set, the verifier, any baseline measurement,
any training, or any backend edits. Those are B onward.

---

## Step 1 — Scaffold the repo

Two packages, both installable, with the dependency direction enforced from day one.

```
packages/harness/      # no torch, no langgraph, no fastapi, no first-party imports
packages/training/     # imports harness; torch lives here
evals/                 # empty for now, filled in Phase B
```

**Done when:** `pip install -e packages/harness` succeeds in a fresh venv that has no torch, and
`python -c "import harness"` works there.

That negative test is the whole point of the step — it's what keeps `pip install harness` from
dragging torch into the backend's Cloud Run image later.

- [ ] `packages/harness` installable standalone
- [ ] `packages/training` installable, imports harness
- [ ] clean-venv import test passes (no torch present)
- [ ] lint/format/test tooling chosen and wired

**Decisions to record:** packaging tool (uv / hatch / poetry / plain setuptools), Python version,
whether the two packages share one lockfile.

---

## Step 2 — Inventory the backend's tools

Read `../den_bot_server` and write down what actually exists today: every tool, its real argument
names and types, what it returns on success, and **what it returns on error**. Error strings are
in the loop, so they count as part of the contract.

This is read-only reconnaissance. No edits to the backend.

**Done when:** the `<!-- FILL -->` blocks under
[open question #1](../docs/open-questions.md) are filled in with real signatures, and each tool is
marked as either *executes for real* or *not usable yet*.

- [ ] every current tool listed with real arg shapes
- [ ] success-string format captured per tool
- [ ] error-string format captured per tool
- [ ] each tool marked executable / not-yet
- [ ] Postgres tools resolved: one tool or several? (open question #1)

**The rule that decides inclusion:** a tool that cannot really execute during trajectory
generation produces invented observations, and the model learns a fiction. Leaving a tool out is
cheap; adding one later is the cheap kind of change. A stub is not an option.

---

## Step 3 — Port the tools, for real

Schema and implementation colocated, one definition per tool, resources reached through the
injected `Deps` object. No first-party imports — if the backend has a shared formatter that shapes
what the model sees, it gets copied *into* the harness, not imported.

**Done when:** every ported tool can be called from a Python REPL with a real `Deps` and returns a
real string from real infrastructure.

- [ ] `@tool` decorator: one definition → JSON schema + callable
- [ ] `Deps` dataclass defined with the real client set
- [ ] each tool executes against real infrastructure
- [ ] error paths produce deliberate, stable strings (not raw tracebacks)
- [ ] a snapshot Qdrant exists for training use, separate from prod

**Decisions to record:** what goes in `Deps`, where the training-side snapshot lives and how it
gets refreshed.

---

## Step 4 — Freeze the action space and the tool descriptions

Two freezes, both cheap now and expensive later.

**The action space** must include the non-tool actions from the start — `answer_directly`,
`clarify`, `out_of_scope`. Adding a behavior *category* later is more disruptive than adding a
tool. This is [open question #2](../docs/open-questions.md), and its type-side twin #2a
(the shape of `Assistant`) gets settled here too.

**The tool descriptions** render into the prompt, so rewriting them later is a format change that
invalidates every trajectory. If a public MCP release is plausible at all, write MCP-quality
descriptions now. This is [open question #7](../docs/open-questions.md) — a one-hour decision that
otherwise costs a full regeneration six weeks in.

- [ ] non-tool actions decided and named (#2)
- [ ] `Assistant` shape decided — representation, multi-call, raw response, `is_final` (#2a)
- [ ] parser conformance test written (#2a)
- [ ] tool descriptions written at MCP quality and marked frozen (#7)
- [ ] reasoning traces: strip or keep? (#8)

---

## Step 5 — Choose the base model

**This must happen before step 6.** The chat template is part of the format contract, so
`build_context` cannot be finished until you know whose template it renders into.

Default recommendation from the premise doc: a Qwen3-class 4B — Apache-2.0, strong tool calling,
small enough that the latency win is real. Start at 4B before reaching for 8B; routing is an easy
task. This is [open question #3](../docs/open-questions.md).

**Done when:** the model is pulled locally and you have personally looked at what its chat
template does to a messages list — including how it renders tool definitions and tool results.
That inspection is the point of the step, not a formality.

- [ ] family and size chosen (#3)
- [ ] license checked for public release (#3)
- [ ] model pulled and running locally
- [ ] chat template inspected by hand: tool defs, tool results, special tokens
- [ ] teacher model and current prod router model identified (#4)

---

## Step 6 — `build_context`, `run_episode`, and the parity snapshot

The fixed middle. Everything except `Deps` and `model_fn` lives here and never varies between
training and production.

```python
def run_episode(query, model_fn: ModelFn, tools, deps: Deps, max_steps=6) -> Trajectory:
```

**Done when:** one real Denver question runs end to end against a frontier model, calls a real
tool, and returns a `Trajectory` — and a golden snapshot of `build_context` is committed.

```python
def test_prompt_parity():
    state = fixture_state()
    assert build_context(state, TOOLS) == golden_snapshot()
```

Treat a diff in that snapshot as a breaking change requiring a version bump. It is described in
the premise doc as the most valuable test in either repo, and it is the only thing that catches
the silent failure mode.

- [ ] `build_context` implemented
- [ ] `run_episode` implemented
- [ ] `ModelFn` implemented for the frontier API
- [ ] trajectory records stored structured, not as rendered strings
- [ ] golden snapshot committed, parity test passing
- [ ] end-to-end episode runs against real tools

**Decision to record:** streaming. Prod eventually wants tokens flowing to the UI, which means
`run_episode` becomes an async generator yielding step events. Build it that way now or defer?

---

## Dependencies between steps

```
1 ──► 2 ──► 3 ──┐
                ├──► 6
      4 ────────┤
      5 ────────┘        (5 must precede 6: chat template is the format contract)
```

Steps 4 and 5 can proceed in parallel with 2 and 3. Step 6 needs all of them.

---

## Decision log

Fill in as decisions land. This is the part that makes the doc worth keeping.

| # | Decision | Chosen | Date | Why |
|---|---|---|---|---|
| 1 | Packaging tool | | | |
| 1 | Python version | | | |
| 2 | Frozen tool list | | | |
| 2 | Postgres: one tool or several | | | |
| 3 | `Deps` field set | | | |
| 3 | Training Qdrant snapshot location | | | |
| 4 | Non-tool actions | | | |
| 4 | `Assistant` shape | | | |
| 4 | Tool descriptions MCP-grade? | | | |
| 4 | Reasoning traces: strip or keep | | | |
| 5 | Base model family + size | | | |
| 5 | Teacher model | | | |
| 6 | Streaming now or later | | | |

## Open questions raised during Phase A

Log anything new here rather than resolving it silently, and promote it to
[`../docs/open-questions.md`](../docs/open-questions.md) if it blocks later phases.

- _(none yet)_
