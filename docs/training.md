# Training

0x960 has four distinct improvement paths that compound on each other. Together they produced a **+596.5 Elo internal gain** and pushed the engine to **competitive with Stockfish 1600** in local Chess960 benchmarks.

---

## 1. Teacher Distillation

**Problem:** Base models never discover the core engineering loop. When we dropped Qwen 3.5 into the environment raw, it spammed `run_static_eval` for every step and quit without writing a single line of code. Reward: **-2.1**.

**Solution:** Use a strong teacher (GPT-5.4) to generate successful bounded-action trajectories, then distill those into a smaller student.

```sh
uv run python -m train.codex_distill \
  --base-url http://127.0.0.1:8000 \
  --model gpt-5.4 \
  --episodes 20
```

Outputs:
- `outputs/codex_distill/teacher_rollouts_*.jsonl` — raw per-episode teacher traces
- `outputs/codex_distill/sft_samples_*.jsonl` — filtered turn-level chat samples for student SFT

The teacher is constrained to the same bounded JSON action schema as the student, so collected data matches the student interface exactly.

---

## 2. Student SFT

Distills the successful teacher traces into a small open model.

```sh
uv run python -m train.sft_student \
  --model Qwen/Qwen3.5-0.8B \
  --output-dir outputs/sft_qwen_0p8b
```

**Results on Northflank H100** (105 clean rows / 35 successful episodes, ~5 min training):

| Metric | Value |
|--------|-------|
| Train loss | 0.2072 |
| Eval loss | 0.04192 |
| Token accuracy | **98.76%** |

**Before vs after SFT:**

| | Base Qwen 3.5-0.8B | SFT Student |
|---|---|---|
| Episode length | 2 steps (eval → quit) | 3 steps (write → match → finish) |
| Reward | -2.1 | **+1.0** |
| Writes code? | Never | Reliably |
| Tests changes? | Never | Always |

The SFT student reliably executes the full `write_file → run_match → finish` engineering workflow. This is the behavioral foundation that makes GRPO meaningful.

---

## 3. GRPO / RL Refinement

We ran HF TRL GRPO over the bounded OpenEnv coding environment — not a plain text-only reward model. The policy was optimizing structured multi-step tool use over a Chess960 engine-editing task.

The agent observed the current `eval.py`, recent action history, remaining budget, and match feedback, then emitted bounded actions like `write_file`, `run_match`, and `finish`. Reward was downstream and environment-grounded: real engine outcomes plus shaping around valid edits, meaningful test cycles, and clean episode completion. The RL objective was therefore closer to **long-horizon workflow optimization** than standard single-turn preference tuning.

**The key insight from the GRPO runs:** raw RL was not the bottleneck solver by itself because base policies had weak action priors. Small and mid-size base models often failed to discover the core edit loop and instead collapsed into degenerate actions (repeated eval calls or premature finish). So the actual training strategy evolved into a teacher-student stack: collect successful bounded-action trajectories from a stronger teacher, SFT/distill a smaller student onto that workflow, then apply GRPO as a refinement stage. In RL terms, **GRPO became the policy improvement phase on top of a behaviorally competent initialization**, not the mechanism for discovering the task ontology from scratch.

```sh
# Handcrafted sanity check
uv run python -m train.minimal_trl_openenv \
  --mode handcrafted \
  --base-url http://127.0.0.1:8000

# Single policy inference episode
uv run python -m train.minimal_trl_openenv \
  --mode infer \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen3.5-0.8B

# GRPO training
uv run python -m train.minimal_trl_openenv \
  --mode train \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen3.5-0.8B \
  --steps 20 \
  --num-generations 4
```

The training entrypoint supports handcrafted, infer, and train modes so the same environment contract is used for debugging, rollout inspection, and RL. Heavy observability is built in: rollout logs, parsed action traces, code previews, match scores, and per-step summaries.

**Models used:**
- Student: `Qwen/Qwen3.5-0.8B` (distilled student path)
- Scaling probe: `Qwen/Qwen3.5-9B` with QLoRA-style GRPO experiments on Northflank H100

**Infra:** local dev for fast iteration, Northflank H100 for heavy SFT / RL runs

## 4. Outer-Loop Engine Search

This path improves the engine directly rather than the bounded-action policy.

Two variants exist:
- local Codex swarm for bounded patch search
- manual classical engine improvements in `search.py` and `eval.py`

Swarm setup:

```sh
uv run python -m train.codex_swarm setup --workers 3
```

One eval-focused round:

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

One search-focused round:

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

Use this when you want:
- faster direct engine improvement
- champion/challenger search over eval and search heuristics
- benchmark-driven engine optimization for the demo story

## Which Path To Use

**Policy learning** (teaching the agent *how* to engineer):
1. Teacher distillation → 2. Student SFT → 3. GRPO refinement

**Engine strength** (making the engine *actually better at chess*):
1. Benchmark-driven eval search (Codex swarm) → 2. Classical search upgrades → 3. Distill winning traces back into policy

Both paths run in parallel and reinforce each other.

---

## Timeline

Here's what actually happened during the hackathon, roughly chronological:

### Hour 0–2: Environment & Scaffolding
- Built the bounded OpenEnv environment with the five-action schema
- Implemented Chess960 engine core (board, eval, search, workspace isolation)
- Wired up FastAPI server, OpenEnv 0.2.1 integration, HF Spaces deployment
- First end-to-end smoke test: handcrafted rollout returns `reward=0.125`

### Hour 2–4: First Model Runs
- Deployed to Northflank H100 (80GB HBM3, PyTorch 2.8.0 + CUDA 12.6)
- Ran Qwen 3.5-9B inference against live environment — model reasons about chess but never writes code. Reward: **0.25**
- Attempted full-parameter GRPO on 9B — OOM on 80GB H100
- Switched to QLoRA (4-bit NF4 + LoRA r=16) — training runs but policy stays degenerate

### Hour 4–6: Diagnosing the Real Problem
- Analyzed rollout logs: **the failure was behavioral, not architectural**
- Base models collapsed into `run_static_eval` spam or premature `finish` — never attempted code edits
- This was the critical insight: raw RL can't optimize a policy that doesn't explore the right actions
- Redesigned reward shaping: valid writes get bonuses, repeated evals get penalties, finishing without editing is heavily penalized

### Hour 6–10: Teacher-Student Breakthrough
- Built teacher trajectory collector (`codex_distill.py`) using GPT-5.4
- Collected 35 successful bounded-action episodes (105 clean SFT rows)
- Built student SFT pipeline (`sft_student.py`)
- **Trained Qwen 3.5-0.8B student on H100 in ~5 minutes**
- Result: base model reward **-2.1** → SFT student reward **+1.0**
- Student reliably executes `write_file → run_match → finish` on first try

### Hour 10–14: Codex Swarm & Engine Search
- Built the autonomous Codex swarm coordinator with 5 specialized worker roles
- Each worker targets different chess knowledge: structure, tactics, activity, pawn endgames, initiative
- Champion/challenger tournament with staged screening (8-position screen → 16-position final)
- **4 eval champions promoted through the tournament gate**

### Hour 14–18: Classical Search Upgrades
- Upgraded search from bare negamax to a competitive engine:
  - Quiescence search with in-check evasion handling
  - Transposition table with persistent TT reuse across moves
  - Principal Variation Search (PVS) — 40% speed improvement on later plies
  - Null-move pruning, Late Move Reductions (LMR)
  - Aspiration windows, killer moves, history heuristics
  - Selective root depth extensions for openings, checks, and endgames
- **Internal benchmark: +596.5 Elo vs search baseline**
- **External anchor: +221.1 Elo vs Stockfish 1320**
- **Competitive with Stockfish 1600** in local Chess960 setup

### Hour 18–20: Dashboard, Benchmarks & Submission
- Built full benchmark suite: eval-vs-eval, engine-vs-engine, UCI anchors, league self-play
- Generated static dashboard with progression charts, Elo deltas, and Stockfish anchor bars
- Search-surface swarm rounds: workers editing `search.py` directly
- Final polish, submission artifacts, demo prep
