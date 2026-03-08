---
title: "0x960"
emoji: "♟️"
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
---

# 0x960

0x960 is an OpenEnv environment where a model improves a minimal Chess960 engine by editing a bounded `eval.py` file and getting rewarded by match outcomes.

The core task is not "play chess." The task is "act like a bounded engine engineer": inspect the current evaluation logic, edit it, test it, and decide when to finish.

## Current Direction

The repo currently supports two training paths:

- teacher distillation first: collect high-quality bounded-action trajectories from Codex or another strong coding agent, then fine-tune a smaller open model on those traces
- RL refinement second: use the OpenEnv reward loop to sharpen a student model that already knows the `write_file -> run_match -> finish` workflow

This ordering is deliberate. The main failure mode so far has not been raw model size; it has been weak action priors. Base models tend to spam `run_static_eval` or `finish` instead of discovering code edits. Distillation fixes that faster than asking GRPO to invent the workflow from scratch.

There is also a complementary outer-loop path: use multiple local Codex workers to iterate directly on the engine, benchmark every patch, keep only Elo-positive changes, and then distill the best traces back into an open student. See [Codex Swarm Plan](docs/codex-swarm-plan.md).

## GRPO + OpenEnv Training Notes

We ran a Hugging Face TRL GRPO setup over a bounded OpenEnv coding environment rather than a plain text-only reward model. The policy was not optimizing free-form completions in the abstract; it was optimizing structured multi-step tool use over a Chess960 engine-editing task.

Concretely, the agent observed the current `eval.py`, recent action history, remaining budget, and match feedback, then emitted bounded actions like `write_file`, `run_match`, and `finish`. Reward was downstream and environment-grounded: real engine outcomes plus shaping around valid edits, meaningful test cycles, and clean episode completion. The RL objective was therefore closer to long-horizon workflow optimization than standard single-turn preference tuning.

The key lesson from the GRPO runs was that raw RL was not the bottleneck solver by itself because base policies had weak action priors. Small and mid-size base models often failed to discover the core edit loop and instead collapsed into degenerate actions like repeated evaluation calls or premature `finish`.

So the training strategy evolved into a teacher-student stack:

1. collect successful bounded-action trajectories from a stronger coding teacher,
2. SFT/distill a smaller open student onto that workflow,
3. apply GRPO as a refinement stage.

In NLP terms, GRPO became the policy-improvement phase on top of a behaviorally competent initialization, not the mechanism for discovering the task ontology from scratch.

Implementation-wise, the stack uses Hugging Face `transformers` + `trl` GRPO with Qwen 3.5-family models on the student side, including `Qwen/Qwen3.5-0.8B` for the distilled student path and earlier larger-Qwen GRPO experiments as scaling probes. The training entrypoint is one TRL/OpenEnv script with handcrafted, infer, and train modes, so the same environment contract is reused for debugging, rollout inspection, and RL.

We also added heavy observability: rollout logs, parsed action traces, code previews, match scores, and per-step summaries. That made it possible to diagnose the real failure mode: not “reward too low” in the abstract, but “policy never meaningfully edits code, so GRPO is optimizing noise around a bad exploration frontier.”

## Repo Layout

- `src/zero960/`: engine, workspace, and episode runtime
- `src/zero960_env/`: OpenEnv server, models, and client
- `train/minimal_trl_openenv.py`: handcrafted demo, inference loop, and GRPO training entrypoint
- `train/codex_distill.py`: Codex teacher rollout collector and SFT sample exporter
- `docs/`: concise project docs and process log

## Local Smoke Test

Start the OpenEnv server:

```sh
uv run python -m uvicorn zero960_env.server.app:app --host 127.0.0.1 --port 8000
```

Run the bounded-action demo:

```sh
uv run python -m train.minimal_trl_openenv --mode handcrafted --base-url http://127.0.0.1:8000
```

## Codex Teacher Distillation

Prerequisites:

- Codex CLI installed and logged in
- local OpenEnv server running

Collect teacher rollouts and export SFT-ready samples:

```sh
uv run python -m train.codex_distill \
  --base-url http://127.0.0.1:8000 \
  --model gpt-5.4 \
  --episodes 20
```

Outputs go to `outputs/codex_distill/`:

- `teacher_rollouts_*.jsonl`: raw per-episode teacher traces
- `sft_samples_*.jsonl`: filtered turn-level chat samples for student fine-tuning

## Student SFT

Train a small student on the collected teacher traces:

```sh
uv run python -m train.sft_student \
  --model Qwen/Qwen3.5-0.8B \
  --output-dir outputs/sft_qwen_0p8b
```

Dry-run the dataset loader first if you want to verify counts and filtering:

```sh
uv run python -m train.sft_student --dry-run
```

The loader validates the assistant action JSON and drops malformed older rows automatically, so the early pre-cleanup SFT dump does not need manual editing.

## Benchmarking Engine Strength

Compare two eval files on held-out Chess960 start positions:

```sh
uv run python -m train.benchmark_eval \
  --candidate-file src/zero960/workspace_template/eval.py \
  --baseline-file src/zero960/engine/default_eval.py \
  --positions 64 \
  --depth 2
```

This is the metric that matters for "better chess" in this repo. Training reward can teach the workflow, but real strength should be checked with held-out match score.

Benchmark a local eval file against an external UCI engine such as Stockfish:

```sh
uv run python -m train.benchmark_uci \
  --candidate-file src/zero960/workspace_template/eval.py \
  --engine-command stockfish \
  --engine-option UCI_LimitStrength=true \
  --engine-option UCI_Elo=1320 \
  --positions 32 \
  --candidate-depth 2 \
  --engine-depth 1
```

This is the cleanest anchor for demo purposes: keep the repo baseline as `0 Elo`, then report how the current champion scores against fixed Stockfish settings under the same Chess960 benchmark.

Benchmark two full engine roots so each side uses its own `search.py` plus its own eval file:

```sh
uv run python -m train.benchmark_engine \
  --candidate-root /tmp/0x960-codex-swarm/worker-1 \
  --baseline-root /Users/qtzx/Desktop/codebase/0x960 \
  --positions 32 \
  --depth 2
```

Use this when you want to open search heuristics safely. The older eval-only benchmark is still the right promotion gate while workers only edit `eval.py`, but once search changes are allowed, head-to-head must load each side's own `search.py` instead of sharing the live repo implementation.

To benchmark a candidate against the original baseline plus accepted swarm champions:

```sh
uv run python -m train.benchmark_league \
  --candidate-file outputs/codex_swarm/champion_eval.py \
  --positions 16
```

By default this league includes the original baseline and the most recent accepted swarm snapshots, while skipping any snapshot that is byte-identical to the candidate. This is the simplest self-play style check for “did the engine improve against its own history, not just one baseline?”

To generate a static dashboard with swarm progression, league results, and optional Stockfish anchors:

```sh
uv run python -m train.build_dashboard --include-stockfish
```

This writes [index.html](outputs/dashboard/index.html) plus the backing [dashboard_data.json](outputs/dashboard/dashboard_data.json). Open the HTML file locally to inspect accepted champions, internal Elo deltas, league self-play, and anchor bars in one place.

To generate submission-ready PNGs for media uploads (score progression + anchor bars), run:

```sh
python3 scripts/generate_submission_media.py
```

This writes tracked files under `media/submission/`.

To also surface the current search gain against the saved pre-upgrade engine baseline:

```sh
uv run python -m train.build_dashboard \
  --include-engine-progress \
  --engine-baseline-root /tmp/0x960-search-baseline \
  --include-stockfish
```

## Local Codex Swarm

Initialize the local champion plus worker sandboxes:

```sh
uv run python -m train.codex_swarm setup --workers 3
```

Run one champion/challenger round with Codex workers:

```sh
uv run python -m train.codex_swarm run \
  --workers 5 \
  --rounds 1 \
  --model gpt-5.3-codex \
  --screen-positions 8 \
  --positions 16 \
  --worker-timeout-sec 180 \
  --max-diff-lines 80
```

Run a search-focused round that edits only `src/zero960/engine/search.py` and benchmarks full engine roots:

```sh
uv run python -m train.codex_swarm run \
  --workers 3 \
  --rounds 1 \
  --surface search \
  --model gpt-5.3-codex \
  --screen-positions 4 \
  --positions 8 \
  --worker-timeout-sec 180 \
  --max-diff-lines 100
```

Dry-run the coordinator without invoking Codex:

```sh
uv run python -m train.codex_swarm run --workers 3 --rounds 1 --dry-run --serial
```

Run the swarm in a continuous champion/challenger loop:

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

The coordinator now rejects overgrown whole-file rewrites by default. Workers are expected to make surgical `eval.py` edits that stay within the `--max-diff-lines` budget; increasing that flag should be a deliberate choice, not the default. Codex workers no longer run the held-out match benchmark themselves. They patch, optionally do one tiny local sanity check, and stop; the coordinator runs an `8`-position screen on every eligible patch and only runs the heavier final benchmark on the best screen winner.

For `--surface search`, the coordinator freezes a baseline engine snapshot for the round and uses [benchmark_engine.py](train/benchmark_engine.py) so each side gets its own `search.py` plus its own eval. That is the safe path once workers are allowed to touch search heuristics.

The coordinator tries real git worktrees first and falls back to lightweight local clones under `/tmp/0x960-codex-swarm/` when worktree metadata is not writable. Swarm state and accepted challengers are recorded under `outputs/codex_swarm/`. The fast default is now a 3-worker wave, and the coordinator reorders hook lanes each round so empty hooks are targeted first, then simple passthrough hooks, and already-customized hooks last.

Each worker now gets a small local research pack before it edits:

- `AGENTS.md`, `README.md`, and [Codex Swarm Plan](docs/codex-swarm-plan.md)
- benchmark scripts in `train/`
- the current champion snapshot
- the swarm ledger
- accepted historical winners under `outputs/codex_swarm/accepted/`

The default roles are:

- `worker-1`: Structure Researcher
- `worker-2`: Tactical Safety Researcher
- `worker-3`: Activity Researcher
- `worker-4`: Pawn-Endgame Researcher
- `worker-5`: Initiative Tuner

Workers still edit only one file per round. On the default `eval` surface they patch `src/zero960/workspace_template/eval.py`; on the `search` surface they patch `src/zero960/engine/search.py`. In both modes they can inspect the full local research pack to avoid repeating prior winners and to justify their patch against actual benchmark history.

To copy the current swarm champion back into the source tree:

```sh
uv run python -m train.codex_swarm promote
```

## Notes

- The environment already includes the current `eval.py` contents in each observation.
- Reward shaping now favors valid edits, explicit `run_match`, and clean `finish`.
- Invalid writes are rolled back immediately so bad code does not poison the rest of the episode.

## Docs

- [Architecture](docs/architecture.md)
- [Codex Swarm Plan](docs/codex-swarm-plan.md)
- [Why Chess960](docs/why_chess960.md)
- [Demo Script](docs/demo-script.md)
- [Process Log](docs/process.md)
