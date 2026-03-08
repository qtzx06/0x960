# process log

This file is the running build log for 0x960.

Purpose:

- keep a chronological record of meaningful implementation steps
- capture decisions and blockers while they are fresh
- preserve evidence that can be reused in the README, demo, and judging

Logging rules:

- append one entry per meaningful work block
- keep entries short and factual
- include decisions, blockers, and concrete next steps
- summarize command/test results instead of pasting long raw output

## 2026-03-08 01:05 PST

- Upgraded the local Codex swarm prompt in [train/codex_swarm.py](../train/codex_swarm.py) from generic lanes to explicit specialist researcher-implementer roles.
- Workers now receive a local research pack before patching: `AGENTS.md`, `README.md`, the swarm plan, benchmark scripts, the current champion snapshot, the swarm ledger, and accepted historical winners copied into each worker sandbox.
- Kept the editable surface narrow at `src/zero960/workspace_template/eval.py` so promotion still measures one variable cleanly, while making accepted history visible so workers can differentiate instead of repeating the same rewrite.
- Updated [README.md](../README.md) and [docs/codex-swarm-plan.md](./codex-swarm-plan.md) to match the new role-based swarm shape.

## 2026-03-08 01:20 PST

- Expanded the default local swarm from 4 to 5 workers by adding an `Initiative Tuner` role in [train/codex_swarm.py](../train/codex_swarm.py).
- Added continuous swarm mode with `--continuous`, `--max-stall-rounds`, and `--sleep-sec` so the coordinator can keep running promotion rounds until interrupted or until it stalls.
- Kept promotion eval-focused because the current benchmark path only measures `eval.py` cleanly; search edits still need a separate promotion harness before they should be opened up.
- Updated [README.md](../README.md) and [docs/codex-swarm-plan.md](./codex-swarm-plan.md) with the 5-worker defaults and the long-running loop command.

## 2026-03-08 01:35 PST

- Found and fixed a coordinator bug in [train/codex_swarm.py](../train/codex_swarm.py): worker sandboxes were copying the repo `workspace_template/eval.py` instead of overwriting it with the frozen swarm champion before Codex started.
- Stopped the invalid live five-worker loop, patched `_sync_worker_snapshot()` to copy `outputs/codex_swarm/champion_eval.py` into each worker's editable `eval.py`, and prepared to restart the loop from a valid champion snapshot.
- Confirmed this also explains why all five workers initially converged on the same hash despite the new specialist-role prompts.

## 2026-03-08 01:50 PST

- Added a separate search-safe benchmark harness in [train/benchmark_engine.py](../train/benchmark_engine.py).
- This harness benchmarks two full engine roots against each other, loading each side's own `select_move()` from its own `src/zero960/engine/search.py` plus its own eval file, instead of sharing the live repo search module.
- Kept the main swarm promotion gate unchanged for now; this new harness is the prerequisite for later opening `search.py` edits without corrupting head-to-head comparisons.

## 2026-03-07 17:10 PST

- Read the initial project docs and collapsed the scope from a broad research wishlist into a narrow hackathon MVP.
- Decided the core claim should be: a model improves a Chess960 engine through bounded code edits in an OpenEnv environment.
- Explicitly cut frontier-model dependence, OAuth, swarm ideas, and broad repo editing from the MVP.
- Next: rewrite the docs so concept, architecture, and open questions all describe the same project.

## 2026-03-07 17:20 PST

- Rewrote the main docs to match hackathon constraints and the narrowed project scope.
- Locked the required stack to OpenEnv `0.2.1`, HF Spaces, and TRL/Unsloth-compatible training.
- Set the default editable surface to `eval.py` with structured actions instead of unrestricted shell access.
- Next: scaffold the actual repo so the docs describe code that exists.

## 2026-03-07 17:35 PST

- Added project scaffolding, packaging metadata, and the initial source layout under `src/`.
- Added a minimal Chess960 engine core with a default eval, a simple search routine, and a workspace template for bounded edits.
- Added an episode runtime that resets a fresh workspace, accepts structured actions, and computes reward from match performance against a fixed baseline.
- Blocker: local runtime smoke test could not run because `python-chess` is not installed in the current sandbox.
- Next: add the OpenEnv-facing wrapper and a minimal training stub.

## 2026-03-07 17:42 PST

- Added the OpenEnv-facing models, server app scaffold, and a minimal rollout script under `train/`.
- Kept this layer intentionally thin so the core environment logic remains inside the local runtime package.
- Verified syntax compilation across `src/` and `train/`; runtime execution still depends on installing `python-chess`.
- Next: initialize git, clean up commit history, and push the first usable skeleton.

## 2026-03-07 17:52 PST

- Initialized git and created a batched local history for docs, scaffolding, core runtime, and OpenEnv/training glue.
- Rewrote early commit messages to use normal prefixes after the initial naming pattern was corrected.
- Created and pushed the private GitHub repo at `qtzx06/0x960`.
- Next: strengthen the storytelling layer so the repo has a concise research motivation and demo script.

## 2026-03-07 17:56 PST

- Added Chess960 background and project framing to the README.
- Added `docs/why_chess960.md` for judge-facing motivation and `docs/demo-script.md` for the one-minute video.
- Tightened the claim to focus on robustness and self-improvement under distribution shift, not on overclaiming "understanding."
- Next: keep this process log updated as implementation, HF Spaces wiring, and training work continue.

## 2026-03-07 18:15 PST

- Fixed OpenEnv API integration to match actual `openenv-core` 0.2.1 signatures.
- Changed PyPI dependency from `openenv>=0.2.1` to `openenv-core[core]>=0.2.1` (correct package name).
- Models now extend `openenv.core.env_server.types.Action` and `Observation` base classes instead of plain `BaseModel`.
- Fixed `create_app()` kwargs: `env_class` → `env`, `action_type` → `action_cls`, `observation_type` → `observation_cls`.
- Updated `Environment` to use 3 type params `[Act, Obs, State]` and corrected `reset`/`step` signatures with `seed`, `episode_id`, `timeout_s` params.
- Replaced `HTTPEnvClient` with WebSocket-based `EnvClient`; implemented `_step_payload`, `_parse_result`, `_parse_state` abstract methods.
- Added `openenv.yaml` manifest and `Dockerfile` for HF Spaces deployment.
- Rewrote `train/minimal_trl_openenv.py` to include both `--mode handcrafted` (quick demo) and `--mode train` (TRL GRPO with Qwen2.5-Coder-0.5B).
- Verified end-to-end: server starts, `/health` returns OK, handcrafted rollout completes with reward=0.125.
- Next: make repo public, deploy to HF Spaces, test training script in Colab.

## 2026-03-07 18:30 PST

- Made repo public via `gh repo edit --visibility public`.
- Set up Northflank H100 (80GB HBM3) instance with PyTorch 2.8.0 + CUDA 12.6.
- Installed project and all deps on the H100 pod.
- Ran Qwen 3.5 9B (`Qwen/Qwen3.5-9B`) inference test against live Zero960 env on H100.
- Model downloaded 19.3GB weights at ~3.3GB/s, loaded in ~7s on H100.
- Result: model ran 2-step episode (run_static_eval → finish), got reward=0.25.
- Model reasons coherently about the Chess960 task but doesn't attempt code edits yet (expected pre-training baseline).
- Next: run GRPO training loop to teach the model to edit eval.py and improve match scores.

## 2026-03-07 18:45 PST

- Investigated agent harness options: ACP (Agent Client Protocol), smolagents, SkyRL+OpenEnv, TRL+OpenEnv.
- ACP is for IDE↔agent communication — doesn't fit RL training loops.
- Chose TRL's native OpenEnv integration (`rollout_func` + `generate_rollout_completions`) per official docs.
- Rewrote training script to use TRL's `rollout_func` pattern with multi-turn episodes (like TRL's Wordle example).
- Key changes: `rollout_func` runs full multi-turn episodes per prompt, collects token-level prompt_ids/completion_ids/logprobs, passes env_reward through kwargs to reward functions.
- Uses `vllm_mode="colocate"` for single-GPU H100 training.
- Kept --mode infer for quick Qwen testing, --mode handcrafted for scripted demo.
- Next: test GRPO training on H100, push to HF Spaces.

## 2026-03-07 19:00 PST

- Full-parameter GRPO OOMed on 80GB H100 (9B model = ~18GB weights + ~36GB optimizer + gradients).
- Switched to QLoRA: 4-bit quantization (NF4 via BitsAndBytes) + LoRA r=16 on attention + MLP projections.
- 4-bit model loads at ~8GB, leaving plenty of room for LoRA adapters and optimizer states.
- Discovered Qwen3.5 is a hybrid architecture (ConditionalGeneration, not CausalLM): has both `self_attn` (full attention, 1/4 layers) and `linear_attn` (gated delta, 3/4 layers) + `partial_rotary_factor=0.25`.
- Initial LoRA attempt hit a shape mismatch in `apply_rotary_pos_emb` during gradient checkpointing — fixed by using `use_reentrant=False` and QLoRA.
- Also hit vLLM 0.17 vs transformers 5.3 incompatibility (vLLM wants <5.0). Dropped vLLM for now, using native HF generation.
- Training running on Northflank H100 with QLoRA + gradient checkpointing.
- Next: confirm training completes, check reward progression, update docs.

## 2026-03-08 09:30 PST

- Inspected rollout logs and confirmed the failure mode was policy-level, not pure model size: the agent kept choosing `run_static_eval` or early `finish` and rarely attempted a code edit.
- Tightened the runtime reward shaping around the intended workflow in `src/zero960/runtime/episode.py`: valid changed writes now get an immediate bonus, explicit `run_match` after a write is rewarded, repeated `run_static_eval` and wasted `read_file` calls are penalized, and finishing without an edit or explicit match is penalized.
- Changed write handling so `write_file` validates `eval.py` immediately by loading `evaluate(board)` and rolls back invalid edits instead of leaving the episode in a broken workspace.
- Extended observations with workflow hints and suggested next actions so the policy sees explicit guidance like "write first" and "run_match next" after each step.
- Reworked `train/minimal_trl_openenv.py` prompt instructions to state that `eval.py` is already visible, show the preferred `write_file -> run_match -> finish` sequence, and reduced completion length from 512 to 256 to bias toward compact JSON outputs.
- Replaced the brittle regex JSON parser with a brace-balanced extractor so `write_file` actions containing nested braces in Python code are more likely to parse correctly.
- Next: run fresh `infer` and short GRPO checks to see whether the action distribution shifts from `run_static_eval`/`finish` toward `write_file`.

## 2026-03-08 11:10 PST

- Added `train/codex_distill.py`, a teacher-data collection path that runs Codex through the same bounded Zero960 action schema and writes both raw rollout traces and SFT-ready chat samples.
- Kept the teacher constrained to one JSON action per turn with a strict output schema, so the collected data matches the student policy interface instead of leaking shell/editor tool use.
- Simplified the top-level docs around the current strategy: distillation first, RL refinement second.
- Deleted redundant planning/research docs that mostly restated old RL-first assumptions and rewrote the README to point at the active entrypoints and docs that still matter.
- Added generated training artifacts to `.gitignore` so local and remote runs stop cluttering the worktree.
- Next: run a first short Codex teacher collection against the live env and inspect how many traces survive the reward filter.

## 2026-03-08 21:20 PST

- Added `train/sft_student.py`, a minimal student fine-tuning entrypoint that reads the exported `sft_samples_*.jsonl` files, validates the assistant action payloads, drops malformed legacy rows, deduplicates identical chats, and trains with TRL `SFTTrainer`.
- Kept the dataset conversational and enabled `assistant_only_loss` so the student is optimized on the teacher’s bounded JSON action turn, not on reproducing the prompt text.
- Added a dry-run mode and basic dataset stats so the repo can inspect the current teacher corpus before launching a real training job.
- Updated the README with the new student-SFT command, keeping the distill-first flow explicit.
- Next: run a dry-load smoke test locally, then train a first 0.8B student checkpoint and compare `infer` behavior against the base model.

## 2026-03-08 22:05 PST

- Re-read the project docs and aligned the next work with the intended claim: reward should reflect downstream Chess960 engine strength, not only loop compliance.
- Replaced the toy default eval with a stronger Chess960-safe heuristic in both `src/zero960/engine/default_eval.py` and `src/zero960/workspace_template/eval.py`, adding pawn-structure, mobility, center control, rook-file, king-safety, castling-rights, bishop-pair, and development terms.
- Added simple move ordering to `src/zero960/engine/search.py` so shallow alpha-beta spends more time on captures, checks, promotions, and castling moves.
- Updated the deterministic training write path in `train/minimal_trl_openenv.py` to make small valid edits against the new eval constants instead of the old toy eval body.
- Added `train/benchmark_eval.py` plus a README command so candidate eval files can be compared against a baseline on held-out Chess960 start positions with an estimated Elo delta.
- Next: run the benchmark on the H100 against saved candidate evals and use that metric, not shaped reward alone, to judge whether future training actually improves play.

## 2026-03-08 22:40 PST

- Ran the first remote student SFT job on the Northflank H100 against the merged teacher corpus (`105` clean rows / `35` successful episodes); the job finished cleanly in about `5m 11s`.
- Final SFT metrics on the remote run were strong for this narrow dataset: train loss `0.2072`, eval loss `0.04192`, eval token accuracy `0.9876`.
- Compared `infer` behavior on the H100: base `Qwen/Qwen3.5-0.8B` still spammed `run_static_eval` for all six steps and ended at reward `-2.1`, while the SFT checkpoint executed the intended `write_file -> run_match -> finish` loop in three steps for reward `1.0`.
- Updated `train/minimal_trl_openenv.py` so infer mode can accept a separate tokenizer path when evaluating checkpoints that do not bundle tokenizer files at the model root.
- Next: run a small batched eval of base vs SFT student across multiple episodes and then decide whether to add GRPO refinement or collect more teacher data first.

## 2026-03-08 23:05 PST

- Wrote down the new higher-level direction in `docs/codex-swarm-plan.md`: use multiple Codex workers on the H100 as a champion/challenger engine-iteration loop, benchmark every candidate, and only keep Elo-positive patches.
- Kept the repo story explicit: OpenEnv remains the core environment and submission artifact, while the Codex swarm acts as an outer optimization layer for discovering stronger engine code and better teacher traces.
- Updated the README to link this plan so the project direction is visible without digging through chat history.
- Next: finish Codex CLI auth on the H100, create isolated worker worktrees, and start with a small `eval.py`-only worker swarm before broadening the editable surface.

## 2026-03-08 23:20 PST

- Simplified the Codex swarm plan: local Codex workers are now the default orchestration path, while the H100 is treated as an optional heavy-compute box for larger benchmarks and training.
- Updated `docs/codex-swarm-plan.md` to reflect the practical setup that avoids remote Node/npm bootstrap friction and keeps the worker loop easier to debug.
- Updated the README wording so the Codex outer loop is clearly described as a local worker swarm rather than an H100-hosted agent farm.
- Next: finish local Codex auth, create 3 local worker worktrees, and start with an `eval.py`-only champion/challenger loop.

## 2026-03-08 23:40 PST

- Added `train/codex_swarm.py`, a runnable local coordinator for the new champion/challenger loop instead of leaving the swarm idea only in docs.
- The coordinator initializes a champion eval snapshot under `outputs/codex_swarm/champion_eval.py`, spins up worker sandboxes under `/tmp/0x960-codex-swarm/`, and runs one Codex worker per sandbox against the same frozen champion each round.
- Worker setup now tries `git worktree add` first and falls back to a lightweight local `git clone --shared` when `.git/worktrees` cannot be written, which makes the swarm usable in stricter local sandboxes too.
- Round execution writes prompts, Codex stdout/stderr, final summaries, and per-worker JSON results under `outputs/codex_swarm/runs/`, then promotes only the best challenger whose held-out score beats the configured threshold.
- Refactored `train/benchmark_eval.py` into a reusable library surface with `benchmark_eval_files(...)` plus a structured `BenchmarkResult`, so the CLI benchmark and swarm coordinator share the same evaluation logic.
- Smoke-tested the new entrypoints locally with `py_compile`, `uv run python -m train.codex_swarm setup --workers 2`, `uv run python -m train.codex_swarm run --workers 2 --rounds 1 --dry-run --serial`, and `uv run python -m train.codex_swarm status`.
- Next: run a live local Codex round, inspect the first real challenger diffs and benchmark scores, then decide whether to broaden the editable surface beyond `eval.py`.

## 2026-03-08 23:58 PST

- Added `train/benchmark_uci.py`, a separate UCI benchmark entrypoint for anchoring the local eval/search engine against external engines like Stockfish under fixed Chess960 start positions.
- The new harness loads a local `eval.py`, plays both colors against a UCI engine, and reports wins, draws, losses, score, and an Elo-style delta estimate so the demo can show both relative improvement and an external anchor.
- Documented the new Stockfish-style benchmark command in `README.md` alongside the existing baseline-vs-candidate benchmark flow.
- Smoke-tested the new entrypoint locally with `python3 -m py_compile train/benchmark_uci.py` and `uv run python -m train.benchmark_uci --help`.

## 2026-03-09 00:09 PST

- Extended `train/benchmark_uci.py` with repeated `--engine-option NAME=VALUE` support so the repo can run calibrated UCI anchors such as `UCI_LimitStrength=true` and `UCI_Elo=1320` instead of only raw depth-based Stockfish tests.
- Installed `stockfish` locally and ran the first rough external ladder against the best current challenger from the Codex swarm.
- On a small `4`-position / `8`-game sample, the worker-1 patch scored `4.5/8` against `stockfish` with `UCI_Elo=1320`, then `2.0/8` against both `UCI_Elo=1600` and `UCI_Elo=1800`; this is noisy but enough to bracket the current engine above the weakest anchor and below the stronger ones.
- Updated the README example to show the calibrated Stockfish option flow rather than only raw `engine-depth`.

## 2026-03-09 00:20 PST

- Tightened `train/codex_swarm.py` for faster live rounds instead of just adding more undirected agents: the default worker count is now `4`, each worker gets an explicit heuristic lane, and the coordinator enforces a per-worker timeout.
- The default lanes now spread the first wave across king safety, loose-piece/tactical pressure, piece activity, and pawn/rook structure so four Codex workers do not all rediscover the same generic positional patch.
- Updated the worker prompt so each agent is explicitly told to make one bounded patch, run one final benchmark, and stop. This should keep rounds short enough to iterate like a real champion/challenger loop.
- Updated `README.md` and `docs/codex-swarm-plan.md` to use the 4-worker setup and document the new `--worker-timeout-sec` flow.
- Smoke-tested the coordinator changes with `python3 -m py_compile train/codex_swarm.py train/benchmark_uci.py train/benchmark_eval.py`.

## 2026-03-09 00:34 PST

- Added `train/benchmark_league.py`, a new league-style self-play benchmark that evaluates one candidate against the original baseline plus the accepted swarm champion history instead of only one current baseline.
- The default league builder pulls from `outputs/codex_swarm/accepted/`, includes the original baseline, skips the candidate itself, and also skips any accepted snapshot whose contents are byte-identical to the candidate so the league does not accidentally include a mirror match.
- Smoke-tested the new script with `python3 -m py_compile train/benchmark_league.py`, `uv run python -m train.benchmark_league --help`, and a tiny real run at `--positions 4`.
- That sample run showed the current champion splitting the small league overall: strong against the original baseline, weaker against the older accepted worker-1 snapshot, and neutral overall on the combined pool.
- Updated the README with the new league benchmark command so the self-play path is visible next to the existing head-to-head and Stockfish anchor commands.

## 2026-03-09 00:45 PST

- Added `train/build_dashboard.py`, a static dashboard generator that reads the swarm ledger, current champion, accepted history, league benchmark, and optional Stockfish anchors, then writes a self-contained `outputs/dashboard/index.html` plus `outputs/dashboard/dashboard_data.json`.
- The generated page visualizes accepted-champion progression, internal Elo deltas, recent swarm results, league self-play rows, and Stockfish anchor bars without needing a frontend framework or a running web server.
- Fixed the default dashboard pool so it skips accepted snapshots that are byte-identical to the current champion instead of showing a misleading mirror match.
- Smoke-tested the generator with `python3 -m py_compile train/build_dashboard.py`, `uv run python -m train.build_dashboard`, and a full `uv run python -m train.build_dashboard --include-stockfish`.
- Updated the README with the new dashboard command and output paths so the visualization can be regenerated after each swarm round.

## 2026-03-09 00:09 PST

- Tightened `train/codex_swarm.py` so worker prompts now explicitly require surgical edits via `apply_patch` and call out a hard diff-size budget instead of loosely asking for “small” changes.
- Added `--max-diff-lines` to the swarm CLI, defaulting to `80`, and recorded added/deleted diff counts in each worker result so whole-file rewrites are visible in the ledger.
- Promotion/acceptance now requires both `benchmark.score > min_score` and `diff_lines_added + diff_lines_deleted <= max_diff_lines`, which stops noisy 250-line rewrites from winning by a tiny margin.
- Updated `README.md` and `docs/codex-swarm-plan.md` to use the new `--max-diff-lines 80` flag in the standard swarm commands.
- Smoke-tested the coordinator change with `python3 -m py_compile train/codex_swarm.py` and `uv run python -m train.codex_swarm run --workers 2 --rounds 1 --dry-run --serial --max-diff-lines 40`.

## 2026-03-09 00:09 PST

- Checked the official Codex docs and aligned `train/codex_swarm.py` to the best practices that are actually supported by the installed CLI on this box.
- Added a per-worker root `AGENTS.override.md` so the “surgical patch only / no whole-file rewrite / one probe max / one final benchmark” constraints live in Codex’s native instruction channel instead of only in the prompt body.
- Kept workers on sandboxed automatic execution, but disabled web search with `-c 'web_search="disabled"'` so the swarm stays local and reproducible.
- Switched worker prompts from giant argv strings to stdin (`codex exec ... -`), which keeps process listings readable and avoids shoving long prompts into the command line.
- Enabled `--ephemeral` and `--json` for worker execs so automation runs stay stateless and stdout captures machine-readable Codex events for debugging.
- Verified that `npm install -g @openai/codex@latest` still resolves to `@openai/codex@0.111.0`; this box is already on the newest npm-published CLI, and that version does not support the newer `--ask-for-approval` flag from the docs.
- Smoke-tested the updated coordinator with `python3 -m py_compile train/codex_swarm.py` and `uv run python -m train.codex_swarm run --workers 1 --rounds 1 --dry-run --serial --max-diff-lines 40`.

## 2026-03-09 00:22 PST

- Tightened the swarm loop again after seeing five workers converge on near-identical 250-300 line rewrites without finishing promptly.
- Changed `train/codex_swarm.py` so Codex workers no longer run `train.benchmark_eval` themselves. They now research, patch `eval.py`, optionally do one tiny local sanity check, and stop.
- Moved the expensive held-out benchmark fully into the coordinator path and made the coordinator skip benchmarking entirely when a worker exceeds the `--max-diff-lines` budget.
- Updated `README.md` and `docs/codex-swarm-plan.md` to reflect the new control flow: Codex proposes, coordinator benchmarks, promotion stays centralized.
- Smoke-tested the refactor with `python3 -m py_compile train/codex_swarm.py` and `uv run python -m train.codex_swarm run --workers 1 --rounds 1 --dry-run --serial --max-diff-lines 40`.

## 2026-03-09 00:31 PST

- Switched the swarm default model from `gpt-5.4` to `gpt-5.3-codex` after inspecting live worker diffs and seeing GPT-5.4 repeatedly collapse into near-identical 300-line eval rewrites.
- Updated the standard swarm commands in `README.md` and `docs/codex-swarm-plan.md` to use `gpt-5.3-codex` as the preferred local worker model.
- Next: run a single bounded `gpt-5.3-codex` wave, inspect the raw diffs directly, and only restore continuous mode if the patches become smaller and more diverse.

## 2026-03-09 00:42 PST

- Refactored `outputs/codex_swarm/champion_eval.py` into explicit swarm hook lanes: `_structure_hook`, `_tactical_hook`, `_activity_hook`, `_pawn_endgame_hook`, and `_initiative_hook`.
- Preserved the prior champion behavior by wrapping the existing extra heuristics inside the new hook functions instead of changing the score formula itself.
- Updated `train/codex_swarm.py` so each worker role is now bound to one named hook rather than a vague lane description. Prompts and `AGENTS.override.md` now tell workers to edit only their assigned hook body.
- Verified the refactor with `python3 -m py_compile train/codex_swarm.py outputs/codex_swarm/champion_eval.py`, a dry-run coordinator pass, and a quick old-vs-new champion benchmark: `score=0.500` over `8` games.

## 2026-03-09 01:06 PST

- Found a bug in the swarm diff gate: it was measuring candidate changes against repo `HEAD` instead of the frozen champion snapshot copied into each worker sandbox, which falsely made every worker look like a 300-line rewrite.
- Fixed `train/codex_swarm.py` to compute diff counts against the pre-run snapshot, not the git checkout below it.
- Fixed the worker-timeout path so `subprocess.TimeoutExpired` stdout/stderr bytes are decoded cleanly and timed-out workers still return structured results instead of crashing the coordinator.
- Ran a short hook-targeted `gpt-5.3-codex` probe and confirmed the new structure works: the worker produced a localized patch only inside `_structure_hook` rather than rewriting the evaluator.
- The first localized patch added king-shield, pawn-storm, and Chess960 castled-structure terms inside `_structure_hook`; benchmark measurement is slower than the interactive loop, but the swarm behavior is finally aligned with the intended patch surface.

## 2026-03-09 01:15 PST

- Added a staged benchmark funnel to `train/codex_swarm.py` so workers no longer all pay for the full held-out benchmark.
- New flow: every eligible patch gets a cheap screen benchmark (`--screen-positions`, default `8`), then only the best screen winner gets the heavier final benchmark (`--positions`, now the final-stage sample count).
- Added `--screen-positions` and `--screen-min-score` CLI flags; the default fast path is now `8` positions for screening and `16` for the final promotion check.
- Reduced the recommended worker timeout in the docs from `600s` to `180s` because workers now only patch and return, not benchmark locally.
- Smoke-tested the updated coordinator with `python3 -m py_compile train/codex_swarm.py`, `uv run python -m train.codex_swarm run --workers 1 --rounds 1 --dry-run --serial --screen-positions 4 --positions 8 --max-diff-lines 40`, and `uv run python -m train.codex_swarm run --help`.

## 2026-03-09 01:28 PST

- The first hook-targeted screened round produced a real promotion: `worker-2` patched `_tactical_hook`, screened at `0.656` over `16` games, and held `0.578` over `32` games for an estimated `+54.7 Elo` versus the previous champion.
- Promoted that tactical hook patch into `outputs/codex_swarm/champion_eval.py` and saved the accepted snapshot as `outputs/codex_swarm/accepted/20260308T092035Z_worker-2_eval.py`.
- Tightened the round scheduler so it now reads the current champion and prioritizes underdeveloped hook lanes automatically: empty hooks (`return 0`) first, then simple passthrough hooks (`return _base_*`), then already-customized hooks last.
- Reordered the default worker specializations so the fast three-worker wave now naturally targets `structure`, `pawn_endgame`, and `initiative` once the tactical hook is already carrying custom logic.
- Smoke-tested the prioritizer with `python3 -m py_compile train/codex_swarm.py`, `uv run python -m train.codex_swarm run --workers 3 --rounds 1 --dry-run --serial --screen-positions 4 --positions 8 --worker-timeout-sec 60`, and a direct hook-state probe under `uv run python -`.

## 2026-03-09 01:54 PST

- Multiple follow-up eval-only hook waves regressed on held-out screens: recent `_structure_hook`, `_pawn_endgame_hook`, and `_initiative_hook` candidates all scored below the current tactical-hook champion on fresh `train.benchmark_eval` probes.
- Concluded that the fastest path to a larger jump was no longer eval stacking but classical search quality, since `src/zero960/engine/search.py` was still a bare fixed-depth negamax with no quiescence or transposition memory.
- Upgraded `src/zero960/engine/search.py` with:
  - quiescence search at depth-0 leaves (captures and promotions only),
  - transposition-table probe/store using `board._transposition_key()`,
  - killer-move and history-heuristic move ordering on quiet moves.
- Snapshotted the pre-change search/eval pair into `/tmp/0x960-search-baseline/` and benchmarked the new search against it with `train.benchmark_engine`.
- Internal engine-vs-engine results were dramatic:
  - `positions=2`: `4.0/4`, `score=1.000`
  - `positions=4`: `8.0/8`, `score=1.000`
  - `positions=8`: `15.5/16`, `score=0.969`, estimated `+596.5 Elo`
- External anchor also improved sharply under the upgraded search:
  - `uv run python -m train.benchmark_uci --candidate-file outputs/codex_swarm/champion_eval.py --engine-command stockfish --engine-option UCI_LimitStrength=true --engine-option UCI_Elo=1320 --positions 8 --candidate-depth 2 --engine-depth 1 --max-plies 120 --seed 42`
  - result: `12.5/16`, `score=0.781`, estimated `+221.1 Elo` versus the `1320` anchor in this local setup.

## 2026-03-09 03:08 PST

- Extended `train/codex_swarm.py` with a second swarm surface: `--surface search`.
- Search-mode workers now edit only `src/zero960/engine/search.py`, targeting one named search function per worker:
  - `_move_order_score`
  - `_quiescence`
  - `negamax`
  - `select_move`
  - `_tactical_moves`
- Added `outputs/codex_swarm/champion_search.py` as the frozen swarm search baseline, parallel to `champion_eval.py`.
- The coordinator now snapshots a per-round baseline engine root and uses `train.benchmark_engine` for search-surface promotion, so each side gets its own eval plus its own searcher during held-out matches.
- Added benchmark timeout support to `train/codex_swarm.py` via `--benchmark-timeout-sec` so pathological search patches can be rejected instead of stalling the whole swarm.
- Updated `train/build_dashboard.py` to support `--include-engine-progress`, which benchmarks the current champion eval plus current repo search against `/tmp/0x960-search-baseline` and exposes that result in `dashboard_data.json` / `index.html`.
- Updated `README.md` and `docs/codex-swarm-plan.md` to document:
  - the new search-surface swarm command
  - the engine-progress dashboard command
  - the difference between eval-surface and search-surface promotion
- Smoke-tested the new coordinator and dashboard code with:
  - `python3 -m py_compile train/codex_swarm.py train/build_dashboard.py`
  - `uv run python -m train.codex_swarm run --workers 2 --rounds 1 --dry-run --serial --surface search --screen-positions 2 --positions 4 --worker-timeout-sec 60 --benchmark-timeout-sec 30`

## 2026-03-09 03:23 PST

- The first real search-surface Codex round produced clean small patches in `_move_order_score` and `_quiescence`, but both candidates timed out in the original search screen benchmark configuration.
- Tightened the search-surface coordinator path so search screening is now intentionally cheaper than eval screening:
  - added `--search-screen-positions`
  - added `--search-screen-depth`
  - added `--search-screen-max-plies`
  - added a separate `--final-benchmark-timeout-sec`
- Current intended search fast path is:
  - cheap screen: `positions=1`, `depth=1`, `max_plies=20`
  - final check: a slightly heavier engine-vs-engine match with its own timeout budget
- Fixed a worker snapshot refresh race in `_copy_tree()` by switching the pre-copy cleanup to `shutil.rmtree(..., ignore_errors=True)`, which avoids spurious `FileNotFoundError` failures when reusing local worker sandboxes under `/tmp/0x960-codex-swarm/`.
- Smoke-tested the cheaper search-screen path with:
  - `python3 -m py_compile train/codex_swarm.py`
  - `uv run python -m train.codex_swarm run --workers 2 --rounds 1 --dry-run --serial --surface search --search-screen-positions 1 --search-screen-depth 1 --search-screen-max-plies 20 --positions 4 --depth 2 --max-plies 120 --worker-timeout-sec 60 --benchmark-timeout-sec 20 --final-benchmark-timeout-sec 60`
- Direct fast-screen probes against the earlier search candidates finally returned promptly at `max_plies=20`:
  - move-ordering patch: `score=0.500` over `2` games
  - quiescence patch: `score=0.500` over `2` games
- That is not enough to claim improvement, but it proves the search-surface screen is now operational instead of timing out by default.
- The current engine also held up better than the earlier rough anchor read against a bigger `Stockfish UCI_Elo=1600` sample:
  - `uv run python -m train.benchmark_uci --candidate-file outputs/codex_swarm/champion_eval.py --engine-command stockfish --engine-option UCI_LimitStrength=true --engine-option UCI_Elo=1600 --positions 4 --candidate-depth 2 --engine-depth 1 --max-plies 120 --seed 42`
  - result: `4.5/8`, `score=0.5625`, estimated `+43.7 Elo` versus that local `1600` anchor setting
- Loosened the search-surface screen gate in `train/codex_swarm.py` so neutral search screens (`score == threshold`) can still advance to one heavier final benchmark. The ultra-fast `2`-game search screen is too coarse to treat `0.500` as automatic rejection.

## 2026-03-09 04:14 PST

- Stopped waiting on slow full-match probes and moved to faster direct checks on the already-strong search baseline.
- Added selective root deepening to `src/zero960/engine/search.py` and synced the same change into `outputs/codex_swarm/champion_search.py`:
  - when the root is in check,
  - or the root move count is small (`<= 12`),
  - or the game is in a low-material endgame with moderate branching,
  - `select_move(..., depth=2)` now searches one extra ply at the root instead of paying for full-time `depth=3`.
- Timing sanity checks on the current champion eval with the new searcher:
  - opening roots at nominal `depth=2` stayed fast (`~0.07s` to `0.11s` on three sampled Chess960 starts),
  - a short 10-ply sample game mostly stayed under `~1.0s` per move, with a few heavier later plies around `1.0s`,
  - full `depth=3` remained much slower (`~1.3s` to `1.5s` opening roots, growing to multi-second later plies), so selective root deepening is the better trade for now.
- Quick engine checks on the selective-depth searcher:
  - internal engine-vs-engine smoke test against `/tmp/0x960-search-baseline` with `positions=1`, `depth=2`, `max_plies=80` still swept `2/2` games.
  - local anchor smoke test against `Stockfish UCI_Elo=1600` with `positions=1`, `candidate_depth=2`, `engine_depth=1`, `max_plies=80` also scored `2/2` games.
- Synced the measured best eval surface into `src/zero960/workspace_template/eval.py` so the actual environment workspace now matches `outputs/codex_swarm/champion_eval.py` instead of lagging behind the swarm champion.

## 2026-03-09 04:28 PST

- Fixed a real search-quality bug in both `src/zero960/engine/search.py` and `outputs/codex_swarm/champion_search.py`: quiescence no longer uses stand-pat when the side to move is in check, and it now searches all legal evasions in that case instead of only tactical captures.
- Timing sanity after the in-check quiescence fix stayed healthy on sampled Chess960 openings at nominal `depth=2`:
  - `0.099s`, `0.058s`, and `0.069s` on three sampled starts.
- Filled the previously empty `_structure_hook` in both `src/zero960/workspace_template/eval.py` and `outputs/codex_swarm/champion_eval.py` with conservative pawn-coordination terms:
  - connected pawns,
  - pawn chains,
  - central pawn duos,
  - modest bonuses for advanced central pawns,
  - all phase-weighted so they matter in the middlegame without distorting late endgames.
- Avoided further king-safety duplication in that hook; the new structure terms are intended to complement the existing tactical/activity hooks rather than re-score the same shelter signals.

## 2026-03-09 04:36 PST

- Added a persistent module-level transposition table in both `src/zero960/engine/search.py` and `outputs/codex_swarm/champion_search.py` instead of rebuilding the TT from scratch on every `select_move` call.
- Also started using the stored TT best move at the root for move ordering.
- This is a classical engine improvement rather than a prompt/surface change: later moves in the same game can now reuse earlier search work.
- Short same-game timing probe on Chess960 start `123` at nominal `depth=2` improved substantially versus the earlier selective-depth-only version:
  - early plies dropped to roughly `0.05s` to `0.10s`,
  - later mid-opening plies stayed around `0.32s` to `0.63s`,
  - compared to the prior selective-depth run where similar later plies were around `0.62s` to `1.03s`.
- Kept the selective root deepening path in place, so the current searcher now combines:
  - quiescence,
  - TT probe/store,
  - persistent TT reuse across moves,
  - killer/history ordering,
  - selective one-ply root extensions in tactical / low-branching roots.

## 2026-03-09 04:44 PST

- Added principal variation search (PVS) to both `src/zero960/engine/search.py` and `outputs/codex_swarm/champion_search.py`:
  - first move at a node is searched on the full window,
  - later moves use a zero-window search first,
  - only fail-high candidates get the full re-search.
- This is another classical-engine speed optimization on top of the earlier alpha-beta + TT stack.
- Same 10-ply timing probe on Chess960 start `123` at nominal `depth=2` improved again versus the TT-persistent version:
  - later plies that had been around `0.32s` to `0.63s` came down to roughly `0.25s` to `0.46s`,
  - opening plies stayed in the same healthy range (`~0.05s` to `0.11s`).
- Current search stack is now:
  - alpha-beta negamax,
  - quiescence with in-check evasions,
  - TT probe/store,
  - persistent TT reuse across moves,
  - TT root move ordering,
  - killer/history ordering,
  - PVS,
  - selective one-ply root extensions.

## 2026-03-09 04:51 PST

- Spent part of the newly-won search speed on opening strength by widening selective root deepening for the very early game:
  - new rule in both `src/zero960/engine/search.py` and `outputs/codex_swarm/champion_search.py`:
    - if `fullmove_number <= 2` and the root has `<= 24` legal moves, search one extra ply.
- Short timing probe on Chess960 start `123` at nominal `depth=2` after this change:
  - first two plies were about `0.72s` to `0.79s`,
  - later plies mostly stayed below `~1.0s`,
  - move choices changed from the previous PVS-only run, which is exactly the intended effect.
- This is a deliberate trade:
  - use the earlier TT/PVS speed wins to buy more opening search depth,
  - keep the rest of the game closer to the cheaper depth-2 profile.

## 2026-03-09 05:00 PST

- Added null-move pruning in both `src/zero960/engine/search.py` and `outputs/codex_swarm/champion_search.py` for non-check, non-endgame nodes at depth `>= 3`.
- Null-move did not produce a clean universal speed win on the sampled 10-ply probe, but it did alter the searched lines and reduced some later plies while making the earliest opening ply somewhat heavier. Kept it in place as a standard classical pruning rule pending larger-match validation.
- Added persistent history ordering across moves in both search files so quiet-move ordering can reuse what the engine has already learned earlier in the same game.
- Timing on the same Chess960 start after the last two changes stayed in the same general operating envelope:
  - opening plies roughly `0.86s` to `1.21s` under the widened early-opening extension,
  - later plies mostly around `0.10s` to `0.72s`,
  - still materially better than the older pre-TT / pre-PVS search stack on later same-game plies.
- Tiny one-position `Stockfish UCI_Elo=1600` anchor probes are still too slow / flaky to treat as decision-grade, so the most reliable signal from this phase remains the measured same-game search speed improvements plus the earlier larger baseline/anchor results already recorded above.

## 2026-03-09 05:08 PST

- Added late move reductions (LMR) in both `src/zero960/engine/search.py` and `outputs/codex_swarm/champion_search.py`:
  - later quiet moves at depth `>= 3` are searched at one reduced ply first,
  - only moves that improve the window get the full re-search.
- Added aspiration windows at the root, seeded from the persistent TT score with automatic fallback to a full window on fail-low / fail-high.
- On the standard 10-ply Chess960 timing probe, these two changes kept the engine on a better branch:
  - first ply dropped to about `0.86s`,
  - second ply to about `0.61s`,
  - later plies mostly in the `0.11s` to `0.46s` range.
- Also tried quiescence delta pruning as another leaf-speed optimization, but reverted it after it made several early plies materially worse on the same probe.
- Current kept search stack is therefore:
  - alpha-beta negamax
  - quiescence with in-check evasions
  - TT probe/store
  - persistent TT reuse across moves
  - TT root move ordering
  - persistent history ordering
  - killer ordering
  - PVS
  - null-move pruning
  - LMR
  - aspiration windows at the root
  - selective opening / tactical / endgame root extensions

## 2026-03-09 05:16 PST

- Tried widening the opening-depth policy from:
  - `fullmove_number <= 2` / `<= 24` legal moves
  to
  - `fullmove_number <= 3` / `<= 22` legal moves.
- On the standard 10-ply timing probe, that pushed the first opening plies too high (`~1.40s` and `~1.05s`) without enough evidence of compensating benefit, so the change was reverted.
- Keeping the more conservative opening-depth rule that was already in place before that experiment.
