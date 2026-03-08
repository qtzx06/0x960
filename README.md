# 0x960

0x960 is an OpenEnv-oriented environment where a model improves a minimal Chess960 engine by editing a bounded evaluation file and getting rewarded by match outcomes.

## background

Chess960 is a strong benchmark for generalization because the rules of chess stay the same while the starting position changes across 960 legal configurations. That removes much of the opening-book structure that standard chess systems can exploit and puts more pressure on transferable positional reasoning and search.

Recent engine and research results make this useful for our setting. Classical search engines such as Stockfish remain extremely strong in Chess960, while several neural and RL-heavy systems lose more relative strength than they do in standard chess. Recent work also shows that transformer chess models trained on standard chess suffer noticeable drops on Chess960 positions, which suggests that high in-distribution performance can still rely on brittle configuration-specific pattern matching.

0x960 turns that observation into an OpenEnv task. Instead of asking a model to output chess moves directly, we ask it to improve a minimal Chess960 engine through bounded code edits. The model reads files, edits the evaluation logic, runs checks, and gets rewarded by whether the edited engine performs better against a baseline.

## why chess960

- Chess960 is a controlled distribution shift: the rules are unchanged, but the initial conditions vary.
- That makes it a cleaner test of robustness than standard chess alone.
- The agent is not rewarded for imitation or move prediction; it is rewarded for improving a real system.
- This makes the environment a better fit for OpenEnv than a direct gameplay benchmark because it requires multi-step tool use, debugging, and iterative refinement.

## repo layout

- `docs/`: concept, architecture, and scope docs
- `src/zero960/`: shared engine and episode runtime logic
- `src/zero960_env/`: OpenEnv-facing models, server, and client
- `train/`: minimal TRL/Colab-oriented training entrypoints

## current status

This repo currently contains a thin but functional skeleton:

- minimal Chess960 engine core
- workspace-based bounded file editing runtime
- OpenEnv wrapper scaffold
- minimal TRL rollout stub

## next steps

1. tighten the engine and reward harness
2. validate the OpenEnv app structure against `0.2.1`
3. add a small Colab notebook around the training stub
4. deploy the server to HF Spaces

## supporting docs

- `docs/why_chess960.md`: short research framing for judges and README reuse
- `docs/demo-script.md`: one-minute demo outline
- `docs/process.md`: chronological build log for demo storytelling and judging
- `docs/agent-log-instruction.md`: reusable instruction snippet for coding agents
