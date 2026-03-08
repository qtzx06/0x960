# Codex Swarm Plan

This is the current highest-leverage path for building a stronger Chess960 eval engine quickly.

## Goal

Use multiple Codex workers as an outer-loop engine lab:

- propose changes to `eval.py` and small search heuristics
- benchmark every candidate against the current champion
- keep only Elo-positive patches
- periodically distill the best traces back into a smaller open student

The point is not to replace the OpenEnv environment. The point is to use strong coding agents to search the engine-design space faster than raw RL can.

## Why This Path

The project has already shown two things:

- base small models do not reliably discover the edit loop on their own
- a distilled student can learn `write_file -> run_match -> finish`

That solves workflow compliance, but the actual submission claim needs to be engine strength. The next bottleneck is finding better eval/search ideas, not teaching the loop again from scratch.

## Worker Architecture

Run one coordinator plus several parallel Codex workers locally by default. Use the H100 only for heavy benchmark or training jobs.

- Coordinator:
  - assigns experiment ideas
  - tracks the current champion engine
  - merges only benchmark-positive patches
- Worker:
  - runs in its own git worktree when possible, with a lightweight local clone fallback when the environment cannot write `.git/worktrees`
  - researches the current champion, accepted history, and benchmark code before editing
  - edits a narrow surface area
  - returns one bounded patch plus a short rationale
  - lets the coordinator run the held-out benchmark and promotion gate

The default fast wave should use 3 workers. The coordinator should re-rank hook lanes each round so empty hooks are targeted first, then simple passthrough hooks, and already-customized hooks last. Workers remain specialist researcher-implementers with read-only access to:

- `AGENTS.md`, `README.md`, and this plan
- `train/benchmark_eval.py`, `train/benchmark_league.py`, and `train/benchmark_uci.py`
- the current champion at `outputs/codex_swarm/champion_eval.py`
- the promotion ledger at `outputs/codex_swarm/ledger.jsonl`
- all accepted snapshots under `outputs/codex_swarm/accepted/`

The available specialist roles are:

- worker 1: Structure Researcher
  king safety and castling structure in Chess960 starts
- worker 2: Tactical Safety Researcher
  loose-piece pressure, attacked-undefended pieces, and practical safety terms
- worker 3: Activity Researcher
  piece activity, development, space, and centralization at shallow search depth
- worker 4: Pawn-Endgame Researcher
  pawn structure, passed pawns, rook files, and simple endgame conversion terms
- worker 5: Initiative Tuner
  tempo, mobility pressure, queen safety, and initiative terms that convert shallow-search advantages faster

After each promotion, the coordinator should automatically deprioritize the lane that just gained custom logic and spend the next short wave on the emptier hooks.

There are now two practical swarm surfaces:

- `eval` surface:
  workers edit only `src/zero960/workspace_template/eval.py` and benchmark with `train/benchmark_eval.py`
- `search` surface:
  workers edit only `src/zero960/engine/search.py` and benchmark with `train/benchmark_engine.py` so each side gets its own eval plus its own searcher

## Evaluation Loop

Every candidate should go through the same loop:

1. read current engine code and latest benchmark results
2. make one bounded patch
3. stop quickly and hand the patch back to the coordinator
4. let the coordinator run a cheap screen benchmark first
5. run a heavier final benchmark only on the best screen winner
6. keep only patches that improve held-out score or estimated Elo delta

Preferred rule:

- no patch is promoted unless it beats the current champion on a fixed held-out benchmark set
- each worker should make one bounded patch and stop; the coordinator owns held-out benchmarking
- benchmark in stages: cheap screen on all eligible candidates, heavier final check only for the best screen winner
- workers that run too long should be timed out rather than left to wander
- workers should inspect accepted history first so lanes diverge instead of repeating the same patch four times

## Safety Constraints

Keep the search legible and hard to game.

- edit only engine files, benchmark code, or clearly scoped support code
- no dependency churn unless explicitly needed
- no broad repo rewrites
- benchmark on held-out Chess960 starts, not only the training positions
- record candidate, benchmark settings, and result for every accepted patch

## Local Setup

The default shape is:

- install Codex CLI locally
- log in locally
- create multiple git worktrees next to the main repo
- run several Codex workers in parallel from those worktrees
- keep the main repo as the coordinator / champion branch

Useful local pattern:

- main repo: champion branch and benchmark history
- `/tmp/0x960-codex-swarm/worker-1`
- `/tmp/0x960-codex-swarm/worker-2`
- `/tmp/0x960-codex-swarm/worker-3`

If device auth is used, `codex login --device-auth` will print a one-time URL and code. If API-key auth is easier, `codex login --with-api-key` is also fine.

## Optional H100 Use

The H100 is still useful, but not as the primary Codex host.

- run large held-out benchmarks there
- run student SFT there
- run RL refinement there if needed later

This keeps Codex orchestration simple while still using the GPU box where it actually matters.

## Relationship To OpenEnv

OpenEnv is still the core environment and submission artifact.

This swarm loop is an outer optimization layer:

- OpenEnv remains the bounded agent environment
- teacher traces can still be collected from Codex in the bounded action schema
- the best engine patches found by the swarm can become:
  - stronger workspace templates
  - better baselines
  - better teacher data
  - better student targets

## Immediate Next Steps

1. Finish local Codex CLI auth.
2. Run `uv run python -m train.codex_swarm setup --workers 3`.
3. Start with `uv run python -m train.codex_swarm run --workers 3 --rounds 1 --model gpt-5.3-codex --screen-positions 8 --positions 16 --worker-timeout-sec 180 --max-diff-lines 80`.
4. Start on the `eval` surface only until the hook lanes are no longer giving clean wins, then open the `search` surface.
5. Promote only patches that improve held-out benchmark score.
6. Use the H100 only for heavier benchmark or training passes.
7. Distill the accepted traces back into the student model after enough wins accumulate.

For a longer autonomous loop, use:

```sh
uv run python -m train.codex_swarm run \
  --workers 5 \
  --continuous \
  --max-stall-rounds 3 \
  --model gpt-5.3-codex \
  --screen-positions 8 \
  --positions 16 \
  --max-diff-lines 80 \
  --worker-timeout-sec 180
```

The coordinator should stay opinionated about patch size. Recent Codex waves tended to rewrite nearly the whole file even when the actual improvement was a one-line or one-function tweak, so the current default rejects candidates that exceed the `--max-diff-lines` budget.

This keeps running until interrupted or until several consecutive rounds fail to promote a new champion.
