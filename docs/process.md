# Process Log

Chronological build log for 0x960. Every entry is a meaningful work block with decisions, results, and next steps.

---

## 2026-03-07 20:10 EST — Project Kickoff

- Scoped the hackathon MVP: a model that improves a Chess960 engine through bounded code edits in an OpenEnv environment.
- Cut everything that didn't serve the core claim: no OAuth, no broad repo editing, no frontier-model dependence at runtime.
- Locked the stack: OpenEnv 0.2.1, HF Spaces, TRL/Unsloth-compatible training.

## 2026-03-07 20:20 EST — Docs & Architecture

- Rewrote all project docs to match the narrowed scope.
- Set the editable surface to `eval.py` with structured bounded actions.
- Added `docs/architecture.md`, `docs/why_chess960.md`, `docs/demo-script.md`.

## 2026-03-07 20:35 EST — Scaffolding & Engine Core

- Built the full source layout under `src/`.
- Implemented Chess960 engine core: board representation, default evaluation, search routine, workspace template for bounded edits.
- Episode runtime: resets fresh workspace, accepts structured actions, computes reward from match outcomes.

## 2026-03-07 20:42 EST — OpenEnv Integration

- Wired up OpenEnv-facing models, FastAPI server, and minimal rollout script.
- Kept this layer intentionally thin — core environment logic stays in the local runtime package.
- Verified syntax compilation across `src/` and `train/`.

## 2026-03-07 20:52 EST — Git & GitHub

- Initialized git, created batched local history.
- Created and pushed GitHub repo at `qtzx06/0x960`.

## 2026-03-07 21:15 EST — OpenEnv 0.2.1 API Integration

- Fixed all API mismatches against actual `openenv-core` 0.2.1 signatures.
- Models now extend proper `Action` and `Observation` base classes.
- Replaced `HTTPEnvClient` with WebSocket-based `EnvClient`.
- Added `openenv.yaml` manifest and `Dockerfile` for HF Spaces.
- Rewrote training script with `--mode handcrafted` and `--mode train`.
- **First end-to-end success: server starts, `/health` OK, handcrafted rollout completes with reward=0.125.**

## 2026-03-07 21:30 EST — H100 Deployment & First Model Run

- Made repo public.
- Set up Northflank H100 (80GB HBM3) with PyTorch 2.8.0 + CUDA 12.6.
- Ran Qwen 3.5-9B inference against live environment on H100.
- Model downloaded 19.3GB weights at ~3.3GB/s, loaded in ~7s.
- **Result: model ran 2-step episode (eval → finish), reward=0.25. Reasons about chess but never writes code.**

## 2026-03-07 21:45 EST — TRL GRPO Architecture

- Evaluated agent harness options: ACP, smolagents, SkyRL+OpenEnv, TRL+OpenEnv.
- Chose TRL's native OpenEnv integration (`rollout_func` + `generate_rollout_completions`).
- Rewrote training script to use TRL's multi-turn episode pattern.

## 2026-03-07 22:00 EST — QLoRA on H100

- Full-parameter GRPO OOMed on 80GB H100 (9B model).
- Switched to QLoRA: 4-bit NF4 + LoRA r=16 on attention + MLP projections.
- Discovered Qwen3.5 hybrid architecture: `self_attn` + `linear_attn` + `partial_rotary_factor=0.25`.
- Fixed shape mismatch in gradient checkpointing, dropped vLLM (version conflict).
- **Training running on H100 with QLoRA + gradient checkpointing.**

## 2026-03-08 00:30 EST — The Critical Diagnosis

- Analyzed rollout logs and identified the real failure mode: **behavioral, not architectural.**
- Agent collapsed into `run_static_eval` spam or premature `finish` — never attempted code edits.
- This was the key insight: raw RL can't optimize a policy that doesn't explore the right actions.
- Redesigned reward shaping: valid writes get bonuses, repeated evals get penalties, finishing without editing is heavily penalized.
- Added workflow hints and suggested next actions to observations.
- Replaced brittle regex JSON parser with brace-balanced extractor.

## 2026-03-08 02:10 EST — Teacher Distillation Pipeline

- Built `train/codex_distill.py` — teacher trajectory collector using GPT-5.4.
- Teacher constrained to same bounded JSON action schema as student.
- Collected successful trajectories and exported SFT-ready chat samples.
- Added generated training artifacts to `.gitignore`.

## 2026-03-08 04:20 EST — Student SFT Pipeline

- Built `train/sft_student.py` — minimal student fine-tuning entrypoint.
- Conversational dataset with `assistant_only_loss` — student learns the teacher's bounded actions, not the prompt text.
- Validates assistant action payloads, drops malformed rows, deduplicates.
- Dry-run mode for inspecting corpus before real training.

## 2026-03-08 05:05 EST — Engine Evaluation Upgrade

- Replaced toy default eval with production-quality Chess960-safe heuristic.
- Added: pawn structure, mobility, center control, rook-file activity, king safety, castling rights, bishop pair, development terms.
- Added move ordering to search: captures, checks, promotions, castling prioritized.
- Built `train/benchmark_eval.py` — held-out Chess960 benchmark with Elo delta estimation.

## 2026-03-08 05:40 EST — Student SFT Training (H100)

- Ran first remote SFT job on Northflank H100.
- Corpus: 105 clean rows / 35 successful episodes. Training time: ~5 minutes.
- **Final SFT metrics: train loss 0.2072, eval loss 0.04192, token accuracy 98.76%.**
- **Before vs after:**
  - Base Qwen 3.5-0.8B: spammed `run_static_eval` for all 6 steps, reward **-2.1**
  - SFT student: executed `write_file → run_match → finish` in 3 steps, reward **+1.0**
- The teacher-student path solved action discovery completely.

## 2026-03-08 06:05 EST — Codex Swarm Plan

- Designed champion/challenger engine-iteration loop using multiple Codex workers.
- OpenEnv remains the core submission artifact; Codex swarm is the outer optimization layer.
- Local workers as default orchestration path for fast debugging.

## 2026-03-08 06:40 EST — Codex Swarm Implementation

- Built `train/codex_swarm.py` — runnable local coordinator.
- Champion eval snapshot, worker sandboxes under `/tmp/0x960-codex-swarm/`.
- Git worktree first, lightweight clone fallback.
- Promotion ledger, per-worker JSON results, round logging.
- Refactored `train/benchmark_eval.py` into reusable library surface.

## 2026-03-08 06:58 EST — UCI Anchor Benchmark

- Built `train/benchmark_uci.py` — benchmark against external engines like Stockfish.
- Calibrated anchors with `UCI_LimitStrength=true` and `UCI_Elo` settings.
- **First external ladder: worker-1 patch scored 4.5/8 vs Stockfish 1320, 2.0/8 vs Stockfish 1600.**

## 2026-03-08 07:20 EST — Swarm Tightening

- Narrowed from undirected agents to explicit heuristic lanes: king safety, tactical pressure, piece activity, pawn/rook structure.
- Added per-worker timeout enforcement.
- Workers make one bounded patch and stop; coordinator handles all benchmarking.
- Added `--max-diff-lines` enforcement (default 80) to reject noisy whole-file rewrites.

## 2026-03-08 07:34 EST — League Self-Play Benchmark

- Built `train/benchmark_league.py` — league-style self-play against original baseline plus all accepted champion history.
- Skips mirror matches, skips byte-identical snapshots.
- First league run showed champion splitting: strong vs original baseline, competitive vs accepted history.

## 2026-03-08 07:45 EST — Results Dashboard

- Built `train/build_dashboard.py` — static HTML dashboard generator.
- Visualizes: champion progression, Elo deltas, swarm results, league self-play, Stockfish anchor bars.
- Self-contained HTML + JSON, no framework needed.

## 2026-03-08 08:00 EST — Swarm Hook Architecture

- Refactored champion eval into explicit hook lanes: `_structure_hook`, `_tactical_hook`, `_activity_hook`, `_pawn_endgame_hook`, `_initiative_hook`.
- Each worker role bound to one named hook.
- Fixed diff gate to measure against frozen champion snapshot (not repo HEAD).
- Added staged benchmark funnel: cheap 8-position screen → final 16-position benchmark only for screen winner.

## 2026-03-08 08:28 EST — First Swarm Promotion

- **worker-2 patched `_tactical_hook`, screened at 0.656 over 16 games, held 0.578 over 32 games: +54.7 Elo.**
- Promoted to champion. Saved accepted snapshot.
- Added round scheduler that prioritizes underdeveloped hook lanes automatically.

## 2026-03-08 08:54 EST — Classical Search Overhaul

- Eval stacking hit diminishing returns. Pivoted to classical search quality.
- Added quiescence search (captures + promotions at depth-0 leaves).
- Added transposition table with probe/store.
- Added killer-move and history-heuristic move ordering.
- **Internal engine-vs-engine: 15.5/16 games, score=0.969, +596.5 Elo vs search baseline.**
- **External anchor: 12.5/16 vs Stockfish 1320, score=0.781, +221.1 Elo.**

## 2026-03-08 10:08 EST — Search Surface Swarm

- Extended Codex swarm with `--surface search` mode.
- Search workers edit only `search.py`, targeting specific functions.
- Added `champion_search.py`, per-round baseline engine snapshots.
- Full engine-vs-engine benchmark for search-surface promotion.

## 2026-03-08 11:14 EST — Selective Root Deepening

- When root is in check, move count <= 12, or low-material endgame: search one extra ply.
- Opening roots stayed fast (~0.07s to 0.11s on sampled starts).
- Synced best eval into workspace template.

## 2026-03-08 11:28 EST — Quiescence Fix + Pawn Structure

- Fixed search bug: quiescence no longer uses stand-pat when in check; now searches all legal evasions.
- Filled `_structure_hook` with pawn coordination terms: connected pawns, chains, central duos, phase-weighted.

## 2026-03-08 11:36 EST — Persistent Transposition Table

- Module-level TT persists across `select_move` calls within the same game.
- TT best move used at root for move ordering.
- **Mid-game plies improved from ~0.62-1.03s to ~0.32-0.63s (roughly 40% faster).**

## 2026-03-08 11:44 EST — Principal Variation Search (PVS)

- First move at each node: full window. Later moves: zero-window first, full re-search only on fail-high.
- **Later plies improved from ~0.32-0.63s to ~0.25-0.46s (another 30% speed gain).**

## 2026-03-08 11:51 EST — Opening Depth Extension

- Trade PVS speed wins for deeper opening search.
- If `fullmove_number <= 2` and `<= 24` legal moves, search one extra ply at root.
- Move choices changed from PVS-only run — intended effect.

## 2026-03-08 12:00 EST — Null-Move Pruning + Persistent History

- Null-move pruning for non-check, non-endgame nodes at depth >= 3.
- Persistent history ordering across moves within same game.
- Operating envelope stable: opening plies ~0.86-1.21s, later plies ~0.10-0.72s.

## 2026-03-08 12:08 EST — LMR + Aspiration Windows

- Late Move Reductions: later quiet moves at depth >= 3 searched at reduced ply first.
- Aspiration windows at root, seeded from persistent TT score.
- Tried and reverted quiescence delta pruning (made early plies worse).
- **Final search stack: alpha-beta negamax, quiescence with in-check evasions, TT probe/store, persistent TT reuse, TT root move ordering, persistent history, killer ordering, PVS, null-move pruning, LMR, aspiration windows, selective root extensions.**

## 2026-03-08 12:20 EST — Swarm Promotions Continue

- 4 total eval champions promoted through the tournament gate.
- Search-surface acceptance on latest round.
- Current engine: **competitive with Stockfish 1600 in local Chess960 setup.**

## 2026-03-08 12:45 EST — Submission Artifacts

- Generated submission media: score progression chart, Stockfish anchor bars.
- Built final dashboard with all results.
- Finalized swarm tooling and submission artifacts.

## 2026-03-08 13:00 EST — Final Polish

- Rewrote README: results-first layout with Elo gains, architecture diagram, concrete metrics.
- Updated training docs with GRPO deep-dive and full hackathon timeline.
- Sharpened demo script with before/after numbers.
- All artifacts committed and pushed.
