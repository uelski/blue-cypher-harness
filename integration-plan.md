# Backend ↔ Harness Integration

How the existing Blue Cypher backend changes. Read `harness-premise.md` first.

The backend lives in a sibling checkout (`../<!-- FILL: server dir -->`) and is a separate git
repo. Read it freely from here; prefer making backend *edits* in a session scoped to that repo,
using this doc as the instruction set, so its commit history stays clean.

Note: **production has effectively no users.** Breaking changes are acceptable. There is no
need for a strangler pattern, shadow deploys, or backward compatibility during the cutover.

## The rule

Anything the model's tokens depend on moves to the harness. Everything else stays.

## What moves out of the backend

- Tool JSON schemas
- **Tool execution logic**, including the exact string each tool returns on success *and on error*
- The router/agent system prompt
- Prompt assembly (`build_context`)
- The tool-calling loop itself
- The conditional-edge fan-out that selects between tool nodes (see below)

## What stays in the backend

- FastAPI, routes, auth, request/response models
- SSE streaming to the frontend
- The LangGraph graph, its state schema, and its checkpointer
- Answer synthesis, if it's a separate node
- **All credentials, clients, and connections** — Qdrant, Postgres, API keys
- Deployment: Dockerfile, Cloud Build, Cloud Run config

## The dependency

```
# backend pyproject.toml / requirements.txt
harness @ git+ssh://git@github.com/<!-- FILL: org -->/blue-cypher-harness@v0.1.0#subdirectory=packages/harness
```

Pin an exact tag. Never a branch, never `main`. The pinned version is the mechanism that
guarantees the tuned model and the prod prompt agree.

## The Ctx object

The harness declares what resources tools need; the backend constructs them at startup.

```python
# backend, on startup
from harness import Ctx

ctx = Ctx(
    qdrant=QdrantClient(settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY),
    pg=pg_pool,
    <!-- FILL: nws / rtd / tavily clients -->
)
```

## How the LangGraph graph changes

There are two kinds of conditional edge and they behave differently.

### Edges that route to tools → deleted, not repointed

If the graph currently fans out to a `search_denver_data` node, a `weather` node, a `transit`
node, **that branching logic is the routing decision** — and the harness now owns it along
with the tool-call loop. The whole subgraph collapses into one node:

```python
def agent_node(state):
    traj = harness.run_episode(state["query"], model_fn, TOOLS, ctx)
    return {"traj": traj, "answer": traj.final_answer, "steps": traj.steps}
```

This is not rewiring edges to new tool functions. It is deleting the fan-out because the loop
moved inside.

**Why it must be this way:** if LangGraph decides which tool runs, then LangGraph is assembling
the prompt and interpreting the reply — and training would have to reproduce LangGraph's exact
behavior to match. That's the broken setup. One `run_episode` inside one node is what makes
the two paths provably identical.

### Edges above the loop → untouched

Anything that isn't tool selection stays exactly as it is, reading the harness's output and
branching on it:

- in-scope / out-of-scope gating
- max-steps or error handling
- clarification requests
- checkpointing and interrupts for multi-turn

### Expected shape change

```
before:  route → [search | weather | transit | rtd] → synthesize
after:   agent_node → synthesize
```

...with out-of-scope, max-steps, and checkpointing edges still hanging off wherever they are now.

## ⚠ Audit item before cutting over

Check every current tool node for work **beyond returning an observation to the model** —
writing to Postgres, triggering a side effect, feeding a different graph branch.

Those nodes earn staying in the graph and need handling deliberately. They must not get swept
into `run_episode`, because the harness loop assumes tools are read-only observation producers.
Training rollouts will call these tools thousands of times.

**Findings:**
<!-- FILL after auditing the graph -->
- [ ] audited
- ...

## Cutover steps

1. Audit item above
2. Harness repo exists with tools ported and `run_episode` working, tested standalone
3. `pip install harness@v0.1.0` in the backend
4. Build `Ctx` at startup
5. Delete backend tool modules and the router prompt
6. Replace the tool fan-out subgraph with `agent_node`
7. Verify the surviving conditional edges still read the fields they expect
8. Add trajectory logging in prod — every request becomes real in-distribution training data

Step 8 is worth doing the moment the harness lands, before any training work. Production
traces are the highest-quality data available and the only way to have thousands of them in
six weeks is to start collecting now.

## Parity test

The single most valuable test in either repo. It lives **here**, in `packages/harness`, since
this repo owns `build_context` and both consumers install it. Assert that a fixed state renders
to byte-identical model input:

```python
def test_prompt_parity():
    state = fixture_state()
    assert build_context(state, TOOLS) == golden_snapshot()
```

Snapshot it, commit the snapshot, and treat a diff as a breaking change requiring a harness
version bump. This is the test that catches the silent failure mode.

## Ongoing cost of this arrangement

A tool change now requires a harness version bump plus a backend bump, rather than an edit
in place. That friction is the feature — it's what keeps training and serving in sync.
