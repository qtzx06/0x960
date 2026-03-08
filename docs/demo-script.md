# Demo Script

## 30-Second Version

0x960 is an OpenEnv environment where a model learns to act like a Chess960 engine engineer, not a chess player. The model gets a bounded coding workspace, edits engine code, tests the change with fast matches, and is rewarded by whether the engine actually improves. Raw RL alone struggled because base models often failed to discover the edit loop, so we used a teacher-student path: a stronger coding model generated successful trajectories, a smaller open student learned that workflow, and then we improved the engine further through benchmark-driven search and self-play.

## One-Minute Outline

### 1. Opening

0x960 is a bounded self-improvement environment for a minimal Chess960 engine.

We used OpenEnv to turn engine engineering into a trainable task: inspect code, edit it, test it, and finish only when the engine is actually better.

### 2. Why Chess960

Chess960 keeps the rules of chess fixed while changing the starting position, so it is a cleaner robustness test than standard chess alone.

### 3. What the Agent Does

The policy sees the current `eval.py`, writes a bounded replacement, runs a match, and decides when to finish.

### 4. Why Teacher Distillation

Base models were not discovering `write_file` reliably, so we added a teacher path: collect successful bounded-action trajectories from a stronger coding agent, fine-tune a smaller open model on those traces, then use RL to refine it.

Concretely, we used:
- `gpt-5.4` as the teacher
- `Qwen/Qwen3.5-0.8B` as the distilled student
- GRPO as the RL refinement path once the student knew the workflow

We also ran larger Qwen 3.5 RL experiments on a Northflank H100, but the teacher-student path was the cleaner way to solve action discovery.

### 5. Why OpenEnv

This is a real multi-step tool-use task with code edits, failures, and downstream evaluation. The reward comes from engine strength, not proxy text metrics.

In parallel with policy learning, we also built a benchmark-driven outer loop that directly searches for stronger eval and search heuristics, so the final story is both:
- better agent behavior
- better engine strength

### 6. Close

The result is a self-improvement environment where the model learns a real engineering workflow instead of just outputting moves or text.

Current local strength is roughly in the low-to-mid `1600` range against our calibrated Stockfish anchors, with the usual caveat that this is local Chess960 benchmark strength, not universal Elo.
