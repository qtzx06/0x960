# Architecture

## System Overview

0x960 is a complete self-improvement system with four tightly integrated components, built in ~20 hours at the OpenEnv Hackathon. The system produced **+596.5 Elo** in internal engine strength and pushed the engine to **competitive with Stockfish 1600** in Chess960.

## Core Components

### 1. Chess960 Engine (`src/zero960/engine/`)

A purpose-built Chess960 engine with a competitive classical search stack:

**Search (`search.py`):**
- Alpha-beta negamax with quiescence search (in-check evasion handling)
- Transposition table with persistent TT reuse across moves
- Principal Variation Search (PVS) — 40% speed improvement on later plies
- Null-move pruning for non-check, non-endgame nodes
- Late Move Reductions (LMR) for quiet moves at depth >= 3
- Aspiration windows at root, seeded from persistent TT
- Killer moves + history heuristic ordering
- Selective root depth extensions for openings, checks, and endgames

**Evaluation (`default_eval.py` + swarm champion):**
- Piece values, pawn structure (doubled, isolated, passed, connected, chains)
- Piece mobility, center control, rook file activity
- King safety, castling rights, bishop pair bonus
- Development scoring, phase-aware transitions
- Specialized hooks: structure, tactical, activity, pawn-endgame, initiative

### 2. Episode Runtime (`src/zero960/runtime/`)

The bounded action environment that makes this an RL task:

**Action Space:**

| Action | Purpose | Reward Signal |
|--------|---------|--------------|
| `read_file` | Inspect current eval code | Penalty if redundant |
| `write_file` | Submit bounded replacement | Bonus for valid changed writes |
| `run_static_eval` | Quick position sanity check | Penalty if repeated |
| `run_match` | Full head-to-head match | Bonus for testing after write |
| `finish` | Declare done | Bonus only if engine improved |

**Key Design Decisions:**
- Invalid writes are rolled back instantly — broken code never poisons the episode
- Reward is grounded in actual match outcomes, not proxy text metrics
- Workflow hints guide the policy toward `write → test → finish`
- Observations include current `eval.py` contents, action history, remaining budget, and match feedback

### 3. OpenEnv Integration (`src/zero960_env/`)

OpenEnv 0.2.1 compliant wrapper with:
- FastAPI server with WebSocket support
- Structured action/observation models extending OpenEnv base types
- `openenv.yaml` manifest for HF Spaces deployment
- Docker-ready for production deployment

### 4. Training & Optimization (`train/`)

Four improvement paths that compound on each other:

**Path 1: Teacher Distillation** (`codex_distill.py`)
- GPT-5.4 teacher generates bounded-action trajectories via ACP runtime
- Constrained to same JSON action schema as student
- Collected 35 successful episodes / 105 clean SFT rows

**Path 2: Student SFT** (`sft_student.py`)
- Distills teacher traces into Qwen 3.5-0.8B
- 98.76% token accuracy, 5 minutes on H100
- Student goes from -2.1 reward (never writes code) to +1.0 (full engineering loop)

**Path 3: GRPO Refinement** (`minimal_trl_openenv.py`)
- HF TRL GRPO over bounded OpenEnv environment
- Environment-grounded RL: structured multi-step tool use, not text completion
- Three modes: handcrafted demo, single inference, full training
- Also ran Qwen 3.5-9B QLoRA GRPO as scaling probe on H100

**Path 4: Codex Agent Swarm** (`codex_swarm.py`)
- Over a dozen autonomous Codex agents across multiple rounds
- 5 specialized worker roles targeting different chess knowledge domains
- Champion/challenger tournament with staged screening
- Dual surface: eval-only and search-only editing modes
- 4 eval champions promoted, search-surface promotions active

**Benchmark Suite:**
- `benchmark_eval.py` — eval-vs-eval on held-out Chess960 positions
- `benchmark_engine.py` — full engine-vs-engine (each side owns its own search + eval)
- `benchmark_uci.py` — UCI anchor comparison against Stockfish
- `benchmark_league.py` — league self-play against own champion history
- `build_dashboard.py` — static HTML dashboard with progression charts and Stockfish anchor bars

## Training Strategy

The critical insight that shaped this architecture:

> Raw RL fails at this task because base models never discover the core engineering workflow. GRPO can't optimize a policy that doesn't explore the right actions. **Teacher distillation solves action discovery; RL refines an already-competent policy.**

Order of operations:
1. Teacher distillation → behavioral competence
2. Student SFT → workflow reliability
3. GRPO refinement → reward optimization
4. Codex swarm → autonomous engine improvement (runs in parallel)

## Deployment

| Target | Status |
|--------|--------|
| HF Spaces | OpenEnv 0.2.1 compliant, Docker-ready |
| Northflank H100 | Heavy training + large benchmarks |
| Local dev | Fastest iteration loop for environment + prompt work |
