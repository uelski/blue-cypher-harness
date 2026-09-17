# The Premise

## The goal, in one line

Make Blue Cypher's tool routing run on a small self-hosted open-weight model that is as good
as or better than prompting a frontier model — cheaper, faster, and ours — while keeping the
ability to swap any model back in.

## What "harness in the loop" means

The model never sees LangGraph. It never sees FastAPI. It sees a token sequence and emits a
token sequence. Everything else is machinery that produces those tokens.

"Harness in the loop" therefore means: **during training, the tokens are produced by the same
code path that produces them in production.**

One training episode:

```
harness builds prompt  →  model emits tool call  →  harness executes it
        ↑                                                    ↓
        └─────────── harness appends observation ────────────┘
                              ...repeat...
                        verifier scores the episode
                          gradient update on weights
```

The broken version of this is training against a mock or a hand-written prompt and then
serving through different code. The difference is invisible — no test catches it — and it
shows up as a model that scored well in eval and underperforms in prod.

### Three things are in the loop

Exactly these, and they are the source of the uplift:

1. **Tool schemas** — the exact JSON the model is trained to emit
2. **Context assembly** — how system prompt, history, and tool results serialize into tokens
3. **Observations** — what tools return, *including error strings*

Point 3 is the non-obvious one. The observation text is what the model learns from, so a
tool's output formatting cannot differ between training and prod. This is why tool
*execution* moves to the harness, not just tool schemas.

### Everything else is outside the loop

Retries, streaming, caching, graph topology, auth, deployment. Change these freely.

### Why a tuned 4B can beat a prompted frontier model here

It has memorized our world: our five tools, our argument shapes, our error messages, our
Denver corpus. That's a narrow advantage and it only exists because the scope is narrow.

## Architecture

The harness is a library. Prod and training both import it. Neither imports the other.

```
packages/harness/        # the contract. no torch, no langgraph, no fastapi.
  tools/                 # schema + implementation, colocated
  context.py             # state → messages   ← the format contract
  loop.py                # run_episode
  trajectory.py          # record schema + writer
  verify.py              # episode → reward
packages/training/       # trajectory gen, SFT, GRPO, vLLM serving. torch lives here.
evals/                   # labeled Denver questions
```

Two packages, not one, so `pip install harness` never pulls torch into the backend's
Cloud Run image.

### The pivot interface

Everything hangs off one signature:

```python
ModelFn = Callable[[list[Message], list[ToolSchema]], Assistant]
```

- Prod passes a function that hits the frontier API
- Trajectory generation passes one that hits a frontier model at temperature
- GRPO rollouts pass one that hits the local vLLM endpoint

Same `run_episode` in every case. This is what makes model swapping a config value and what
makes training/serving parity provable rather than hoped for.

```python
def run_episode(query, model_fn, tools, deps, max_steps=6) -> Trajectory:
    state = State(query=query)
    for _ in range(max_steps):
        messages = build_context(state, tools)      # format contract
        action = model_fn(messages, tools)
        state.record(messages, action)
        if action.is_final:
            break
        obs = execute(action, tools, deps)          # errors become observations
        state.record_observation(obs)
    return state.to_trajectory()
```

Streaming is the one wrinkle: prod wants tokens flowing to the UI. Make `run_episode` an
async generator yielding step events; prod maps them to SSE, training drains to completion.

### Resources are injected, not owned

The harness defines tools; the caller supplies connections. Credentials never leave prod.

```python
# harness
@tool
def search_denver_data(deps: Deps, query: str, neighborhood: str | None = None) -> str:
    hits = deps.qdrant.search("denver_documents", query, filter=neighborhood)
    return format_hits(hits)          # ← this formatting is in the loop

# prod                                      # training
deps = Deps(qdrant=QdrantClient(PROD_URL))  deps = Deps(qdrant=QdrantClient(SNAPSHOT_URL))
```

Training needs a **real** Qdrant. Trajectories generated against fake retrieval teach the
model a fiction.

`Deps` is deliberately *not* named `Ctx`. This repo already uses "context" for two other things —
`build_context` (prompt assembly) and the model's context window — and a third meaning for the
bag of database connections is a collision worth avoiding. It holds dependencies; it is called
`Deps`.

## Decisions already made

| Decision | Rationale |
|---|---|
| Blue Cypher, not Homefile | Small closed tool set, public data, verifiable rewards. Homefile's permission keychain must never be a learned stochastic policy. |
| Train one node (the router), not the agent | Makes a small model viable and the reward programmatic |
| SFT on rejection-sampled trajectories | GRPO only if SFT plateaus. The SFT ceiling is real but you should hit it before paying for RL. |
| Keep LangGraph | Conditional edges + checkpointing earn their place — as orchestration *above* the harness |
| Tools own their schema | One definition → training spec, executor, MCP registration |
| Harness is a shared installable package | Parity requirement; a copy in each repo will drift |
| Store structured trajectory records, not rendered strings | Format change becomes a re-render, not a $300 regeneration |

## What is cheap vs expensive to change later

**Cheap — adding a tool.** The tool list is in the prompt at inference time. A new tool needs
a few hundred trajectories exercising it, mixed with a replay sample of existing data.
Hours, not a restart. Routing will underuse the new tool until you do this.

**Moderate — changing an existing schema.** Now you're unlearning something wrong.
Regenerate the affected trajectories rather than patching.

**Expensive — changing context assembly.** New system prompt structure, different tool-result
serialization, different chat template. This invalidates every trajectory, because the tokens
the model trained on no longer match what it'll see. A real restart.

**So: freeze the *format* and the *action-space shape* early. Stay relaxed about tool count.**

The action space must include the non-tool actions from the start — `answer_directly`,
`clarify`, `out_of_scope`. Adding a whole category of behavior later is more disruptive than
adding a tool.

## Model swapping

Frontier models tolerate any reasonable schema zero-shot. The tuned small model is the picky
consumer — it has memorized exact names, shapes, and prompt text. So the harness is the
single source of truth precisely because only one consumer punishes deviation.

Consequences:
- Matching tool *names* is not sufficient. Same system prompt text, same serialization, same ordering.
- Serve the local model via vLLM's OpenAI-compatible endpoint so both sides look identical from our side of the wire.
- Behavior still differs even when format matches — a frontier model may answer directly where
  the router calls a tool. Eval must run per-model.

## MCP

MCP is a *transport* over the same schemas, not a different contract. The single tool
definition generates both the local in-process executor and the MCP server registration.

**Train against the local in-process path**, never through an MCP client: same schemas, no
network, no stdio, thousands of times faster. Rollouts through stdio would be orders of
magnitude slower.

### If Blue Cypher is released as a public MCP server

**An MCP client brings its own harness.** Claude Desktop assembles its own prompt, runs its own
loop, and selects tools with its own model. So in MCP mode the tuned router is *bypassed
entirely* — what ships is the tools layer.

That's not a problem, but it means choosing which product is being released:

| Shape | What's exposed | Does the tuned router run? |
|---|---|---|
| **Tools** | `search_denver_data`, `get_transit_info`, … | No. The caller's model routes. |
| **Agent** | one tool, `ask_blue_cypher(question)` | Yes. `run_episode` runs behind it. |

Both are shippable, and both can ship together. Only the second makes the training work
matter to MCP users.

### ⚠ Tool descriptions are in the loop

The trap worth avoiding. MCP clients lean heavily on the `description` field — it's how a
foreign model figures out what a tool does with no other context — so the instinct when
releasing publicly is to write rich, verbose descriptions.

But descriptions render into the prompt. They are part of `build_context`. Rewriting them is a
**format change**, which invalidates every generated trajectory.

**Therefore: if a public MCP release is plausible, write MCP-quality tool descriptions *before*
generating any trajectories, and treat them as frozen.** This is a one-hour decision that
otherwise costs a full regeneration six weeks in.

### Schema evolution: two consumers, opposite pressures

- A public MCP server is an API — semver, backward compatibility, deprecation windows.
- The tuned model wants the schema never to change.

These mostly agree (both want stability) but diverge on *additive* change: adding an optional
argument is a free, non-breaking MCP release and a routing regression for the tuned model until
trajectories exercising it exist.

**Rule: the harness schema is the source of truth and changes on the model's cadence, not the
API's.** If a friendlier public surface is ever needed, put a translation shim in the MCP layer
rather than touching what the model sees.

### What MCP doesn't change

- **Deps injection already covers it.** The MCP server is a third consumer building its own `Deps`,
  exactly like prod and training.
- **Tools-are-read-only still holds** — already a harness assumption, and also what makes a
  public server safe to expose.

### What MCP adds

Untrusted callers hitting our Qdrant, Tavily key, and RTD quota. Rate limiting, auth, and
per-caller cost caps are real work — and they live in the MCP server, not the harness.

## What would make this project not worth doing

Stated up front so it's easy to accept:

- ~~Base model scores within ~2 points of the current prompted router zero-shot → ship the base
  model, skip training. This is a win.~~ **Superseded.** Learning to train an open-weight model is
  a goal in its own right (see Goals in `CLAUDE.md`), so a good zero-shot score is a data point,
  not an exit. Measure it, then train anyway.
- No eval set with programmatic scoring exists → there is no training signal and no way to
  know if anything worked. Do not proceed past this.
- Tools can't be run for real during trajectory generation → the data will be fiction.
