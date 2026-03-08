# Architecture

## Core Shape

0x960 has four moving parts:

1. `src/zero960/engine/`
   A minimal Chess960 engine with fixed search and one narrow editable surface: `eval.py`.
2. `src/zero960/runtime/`
   The episode runtime that owns workspace resets, bounded actions, reward shaping, and match scoring.
3. `src/zero960_env/`
   The OpenEnv wrapper and WebSocket client/server layer.
4. `train/`
   Distillation and RL entrypoints that operate on the same bounded action schema.

## Action Space

The policy only gets structured actions:

- `read_file`
- `write_file`
- `run_static_eval`
- `run_match`
- `finish`

The full repo is not editable. The policy can only modify `eval.py` inside a fresh workspace.

## Observation Shape

Each observation includes:

- the task instruction
- the current `eval.py` contents
- recent action history
- remaining steps
- last match score
- workflow hints and suggested next actions

The current file contents are already visible in the observation, so the intended high-reward loop is:

`write_file -> run_match -> finish`

## Reward Design

Reward is match-score-based with explicit shaping around the edit loop:

- positive signal for valid changed writes
- positive signal for explicit `run_match` after a write
- penalties for repeated `run_static_eval`, redundant `read_file`, and finishing without a meaningful edit/test cycle
- invalid writes are rolled back immediately

This keeps the environment learnable while still grounding the main score in downstream engine strength.

## Training Strategy

Current order of operations:

1. teacher distillation
   Use a strong coding model such as Codex/GPT-5.4 to generate successful bounded-action trajectories.
2. student fine-tuning
   Fine-tune a smaller open model on those trajectories.
3. RL refinement
   Use GRPO or a similar method only after the student already knows the workflow.

This is the main shift from the earlier RL-first plan. The hard part has been action discovery, not just optimization.

## Deployment

- HF Spaces: public OpenEnv artifact
- Northflank H100: practical heavy training and debugging box
- local dev: fastest loop for environment and prompt iteration
