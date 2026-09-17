# Open Questions

Not yet decided. **Do not silently resolve these** — surface them instead.

## Blocking, must decide before generating any training data

### 1. The frozen tool list

Only tools that **actually execute** can be included. A schema with no implementation produces
episodes where the observation is invented, and the model learns a fiction. It is fine to leave
a tool out — adding one later is the cheap kind of change.

Current prod tools:
<!-- FILL from the backend repo -->
- [ ] `search_denver_data` (Qdrant) — args?
- [ ] weather (NWS) — args?
- [ ] transit (RTD) — args?
- [ ] denvergov search (Tavily) — args?
- [ ] Postgres demographics / crime / parks — one tool or several?

Wanted but not yet built:
<!-- FILL — and for each, decide: build a crude-but-real version now, or defer? -->

### 2. Non-tool actions

Must be in the action space from the start. Adding a behavior *category* later is more
disruptive than adding a tool.

- [ ] `answer_directly`
- [ ] `clarify`
- [ ] `out_of_scope`
- [ ] anything else?

#### 2a. The shape of `Assistant`

`Assistant` is the return type of every `ModelFn` and the thing `run_episode` branches on, so
its shape is part of the action space and therefore part of the frozen format. It is currently
underspecified. Minimum viable:

```python
@dataclass
class Assistant:
    tool_calls: list[ToolCall]     # zero or more
    text: str | None               # final answer, or reasoning
    is_final: bool                 # loop exit condition
```

Decide before generating trajectories:

- [ ] **How are non-tool actions represented?** As entries in `tool_calls` (uniform, one code
      path, trivially scoreable) or as a separate field (cleaner conceptually, but the verifier
      and the loop each grow a branch)? This is the same decision as #2 viewed from the type side.
- [ ] **Can one step carry more than one tool call?** Parallel calls change what a "step" is, and
      so change both trajectory records and the reward shape.
- [ ] **Is the raw provider response retained?** Useful for debugging and re-rendering, but it is
      provider-shaped, so it must never reach `build_context`.
- [ ] **Where does `is_final` come from** — the model saying so, or the harness inferring it from
      an empty `tool_calls`? Inferring is fewer tokens; explicit is less ambiguous.

⚠ **Parsing is a divergence risk.** Each `model_fn` converts its provider's reply into
`Assistant` — Anthropic content blocks, OpenAI `tool_calls` arrays, and vLLM's output are all
shaped differently. That parsing step is per-caller code, which makes it the one place the three
paths can silently disagree while `build_context` stays identical. Write a shared test that feeds
a recorded response from each provider through its parser and asserts the same `Assistant`.

- [ ] parser conformance test written

### 3. Base model family

The chat template is part of the format contract, so this must be settled before any training
data is rendered. Default recommendation: a Qwen3-class 4B (Apache-2.0, strong tool calling,
small enough that the latency win is real). Start at 4B before trying 8B — routing is an easy task.

- [ ] family chosen
- [ ] license checked for public release

### 4. Prod model for trajectory generation

Which frontier model generates the teacher trajectories, and which one prod currently uses
for routing (they can differ).

- [ ] teacher model
- [ ] current prod router model

## Blocking, must decide before training

### 5. Reward / verifier design

The hardest and most important design work in the project. Exact-match on a free-text `query`
argument is hopeless. Score by **outcome** instead:

- did retrieval return the gold chunk ID?
- did the weather call hit the right neighborhood?
- was an out-of-scope question correctly declined?

- [ ] per-tool outcome check defined
- [ ] partial credit vs binary decided
- [ ] verifier implemented and spot-checked by hand on ~30 cases

### 6. Eval set composition

Target 300–500 labeled Denver questions. Sources: the Reddit threads already mined,
denvergov FAQ pages, synthetic questions generated from the Qdrant corpus, and any real query
logs. Deliberately include hard cases: ambiguous, multi-tool, and out-of-scope.

- [ ] sources decided
- [ ] train / dev / **test** split created — and test not looked at

### 7. Tool description quality — MCP-grade or not?

**Blocking.** Descriptions render into the prompt, so rewriting them later is a format change
that invalidates every trajectory. If a public MCP release is plausible at all, write
MCP-quality descriptions now and freeze them. See the MCP section in `harness-premise.md`.

- [ ] decided
- [ ] descriptions written and frozen

### 8. Reasoning traces: strip or keep?

For a classification-shaped task like routing, training the student to emit `<think>` blocks
buys little accuracy and costs the latency win that motivates the project. Leaning: strip.

- [ ] decided

## Non-blocking, revisit later

### 9. Whether to do GRPO at all

Only if SFT plateaus and there's a specific gap worth closing. Budget real time for reward
hacking — the classic failure is the policy discovering one tool is right 60% of the time and
abandoning the others.

### 10. Serving

vLLM or SGLang, OpenAI-compatible endpoint. Modal is least painful for bursty traffic;
Cloud Run GPU keeps the existing stack. Decide when there's a model worth serving.

### 11. MCP server

Generated from the same tool definitions — a serving-time adapter, not a separate contract.
Deferred, **except** for the description-quality decision above, which is blocking.

When it comes up, decide: expose *tools* (caller's model routes; our tuned router is bypassed)
or expose the *agent* as one `ask_blue_cypher` tool (our router runs). Also budget for rate
limiting, auth, and per-caller cost caps — those live in the MCP server, not the harness.

### 12. Whether LangGraph survives long-term

Currently keeping it — conditional edges and checkpointing are in real use. Once the graph is
thin (one `agent_node` plus gating edges), re-evaluate. Cheap decision to defer.

### 13. Monorepo vs two repos for harness + training

Starting as one repo with two packages. Splitting later is trivial — the package boundary is
already the seam.
