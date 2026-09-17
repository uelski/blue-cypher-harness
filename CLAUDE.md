# blue-cypher-harness

The agent harness and model-training work for Blue Cypher. **This repo is the primary
workspace.** Sessions run from here; sibling repos are read-mostly context.

## What this repo is for

Blue Cypher's tool-routing decision currently runs on a prompted frontier model. We are
training a small open-weight model to do that job instead — cheaper, faster, self-hosted —
while keeping the ability to swap any model back in.

The mechanism is "harness in the loop": the code that turns conversation state into tokens
during training is the *same code* that does it in production. That is why this repo ships
the harness as an installable package that the backend depends on, rather than the backend
keeping its own copy.

**The harness is not trained.** It is ordinary code. The *model* is trained, inside it.

## Layout

```
packages/harness/     # the contract. installed by prod AND training.
  tools/              #   schema + implementation, colocated
  context.py          #   state → messages   ← the format contract
  loop.py             #   run_episode
  trajectory.py       #   record schema + writer
  verify.py           #   episode → reward
packages/training/    # trajectory gen, SFT, GRPO, serving. torch lives here.
evals/                # labeled Denver questions
docs/
```

Two packages, not one, so `pip install harness` never pulls torch into the backend's
Cloud Run image.

`packages/harness` imports nothing we own. `packages/training` imports harness. The backend
imports harness. Harness imports neither. If you want `if training:` inside the harness, that
flag belongs in the injected model function instead.

## Sibling repos

Assumed to be siblings of this repo on the local machine. **Paths are a local convention, not
a guarantee** — check before assuming, and never commit an absolute path referencing them.

| Path | What it is | How to treat it |
|---|---|---|
| `../<!-- FILL: server dir -->` | Blue Cypher backend. FastAPI + LangGraph. | **Read freely, ask before writing.** Will `pip install` this repo's harness package at a pinned tag. |
| `../<!-- FILL: frontend dir -->` | Blue Cypher frontend, Cloudflare. | Read-only. Rarely relevant. |

These are separate git repos. Never commit across them. The only legal coupling is the backend
installing `harness` at a pinned version tag — never a branch, never `main`.

When work is needed in the backend, prefer producing a precise diff or instruction set here and
applying it in a session scoped to that repo, so backend commits stay clean.

## Scope

In scope: tool selection + tool arguments.

Out of scope, deliberately: answer synthesis, retrieval quality, reranking, multi-turn planning.
The narrow scope is what makes a small model viable and the reward verifiable. If a task doesn't
affect which tool gets called with which arguments, it is not this project.

## The two invariants

Everything else is negotiable. These are not.

1. **Tools own their own schema.** One definition per tool yields the JSON schema the model sees,
   the executable function, and (later) the MCP registration. No hand-maintained schema list.
2. **`build_context` is the only place messages get assembled.** If the backend ever builds a
   prompt inline — even one extra system-prompt line for one special case — training/serving
   parity is broken and the tuned model degrades silently with no failing test.

The parity test in `docs/integration-plan.md` is what enforces #2. It is the most valuable test
in either repo.

## Working rules

- **Don't duplicate tool logic.** If you're about to write a tool function in the backend, stop —
  it belongs here.
- **Don't resolve open questions silently.** See `docs/open-questions.md`; surface them instead.
- **Tools must really execute.** A schema with no implementation produces training episodes with
  invented observations, and the model learns a fiction.
- **Version, don't branch.** Cutting a harness release means a tag here and a bump there.

## Docs

- `docs/harness-premise.md` — what "harness in the loop" means, the architecture, decisions made, and what's cheap vs expensive to change
- `docs/integration-plan.md` — what moves out of the backend, what stays, how the LangGraph graph changes, cutover steps
- `docs/open-questions.md` — decisions not yet made
- `plans/` — per-phase implementation plans and decision logs. Living docs: record decisions
  there as they land. Start with `plans/phase_a_plan.md`.

## Current state

<!-- FILL as you go -->
- [ ] Repo scaffolded, two packages installable
- [ ] Backend audited (tool list, graph shape, side-effecting tool nodes)
- [ ] Tool schemas frozen
- [ ] Base model family chosen
- [ ] Eval set written (300–500 labeled questions)
- [ ] Verifier implemented
- [ ] Zero-shot baseline measured (a comparison point, **not** a go/no-go — see Goals below)
- [ ] Trajectories generated
- [ ] First LoRA trained
- [ ] Backend cut over to harness

## Goals

Two, and both count:

1. **Make Blue Cypher's tool routing run on a small self-hosted model.**
2. **Learn what it actually takes to train an open-weight model.** The pipeline and the skill are
   deliverables in their own right.

Goal 2 is why the zero-shot baseline is not a gate. `docs/harness-premise.md` says a base model
scoring within ~2 points zero-shot means ship it and skip training; that no longer holds. Measure
the baseline as data, then train regardless.

Goal 2 also sets the pace: prefer small, individually verifiable steps with the mechanics
explained, and transparent tooling over config-driven abstractions that hide what is happening.

## Note on production

Production has effectively no users. Breaking changes are acceptable. No strangler pattern,
shadow deploys, or backward compatibility are needed during cutover.
