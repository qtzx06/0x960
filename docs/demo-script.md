# Demo Script

## 30-Second Version

0x960 is an OpenEnv self-improvement environment where an AI learns to *engineer* chess engines, not play chess. It gets a bounded coding workspace, edits engine evaluation code, tests changes with real matches, and is rewarded only when the engine actually gets stronger. We went from a base model that never wrote a single line of code (reward: -2.1) to a distilled student that reliably executes the full engineering loop (reward: +1.0), while our autonomous Codex agent swarm pushed engine strength by **+596.5 Elo** internally and **beat Stockfish 1320 by +221.1 Elo** — reaching competitive strength with Stockfish 1600 in Chess960.

## One-Minute Outline

### 1. Opening (10s)

0x960 is a bounded self-improvement environment for Chess960 engine engineering, built on OpenEnv 0.2.1.

We turned engine engineering into a trainable RL task: inspect code, edit it, test it against a baseline, and get rewarded only when the engine is measurably stronger.

### 2. Why Chess960 (10s)

Chess960 randomizes the starting position across 960 setups. No opening books, no memorized lines. The engine has to *actually understand chess positions* to improve — you can't game the reward by memorizing patterns.

### 3. The Problem We Solved (15s)

When we dropped Qwen 3.5 into this environment, it scored **-2.1 reward** — it never once attempted to write code. It just read files and quit. Raw GRPO RL couldn't fix this because the policy never explored the right actions.

Our breakthrough: **teacher-student distillation first, RL second.**

- GPT-5.4 teacher generates successful bounded-action trajectories via ACP runtime
- Qwen 3.5-0.8B student learns the workflow through SFT (98.76% token accuracy in 5 minutes on H100)
- TRL GRPO refines the student on real match reward
- We also ran Qwen 3.5-9B QLoRA GRPO as a scaling probe on the Northflank H100

After distillation: reward **+1.0**, reliable `write_file → run_match → finish` execution.

### 4. The Codex Agent Swarm (15s)

In parallel, we built an autonomous Codex agent swarm — over a dozen agents across multiple rounds, each specializing in different chess knowledge (king safety, tactics, pawn structure, piece activity, initiative).

Champion/challenger tournament format: every patch gets benchmarked on held-out Chess960 positions. Only verified winners get promoted. 4 eval champions promoted through the gate. The swarm also edits search heuristics directly.

### 5. Results (10s)

- **+596.5 Elo** internal gain (vs search baseline)
- **+221.1 Elo** vs Stockfish 1320 anchor
- **Competitive with Stockfish 1600** in local Chess960 benchmarks
- Engine went from bare negamax to PVS + TT + null-move pruning + LMR + aspiration windows
- Full benchmark suite: eval-vs-eval, engine-vs-engine, UCI anchors, league self-play, static dashboard

All built in ~20 hours at the hackathon. Two parallel self-improvement loops — policy learning and autonomous engine search — feeding the same engine, with every claim backed by held-out match results.
