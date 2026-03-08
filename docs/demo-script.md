# Demo Script

## 30-Second Version

0x960 is an OpenEnv environment where a model learns to act like a Chess960 engine engineer, not a chess player. The model gets a bounded coding workspace, edits `eval.py`, tests the change with fast matches, and is rewarded by whether the engine actually improves. We found that raw RL alone struggled because base models did not discover the edit loop, so the current path is teacher distillation first and RL refinement second.

## One-Minute Outline

### 1. Opening

0x960 is a bounded self-improvement environment for a minimal Chess960 engine.

### 2. Why Chess960

Chess960 keeps the rules of chess fixed while changing the starting position, so it is a cleaner robustness test than standard chess alone.

### 3. What the Agent Does

The policy sees the current `eval.py`, writes a bounded replacement, runs a match, and decides when to finish.

### 4. Why Teacher Distillation

Base models were not discovering `write_file` reliably, so we added a teacher path: collect successful bounded-action trajectories from a stronger coding agent, fine-tune a smaller open model on those traces, then use RL to refine it.

### 5. Why OpenEnv

This is a real multi-step tool-use task with code edits, failures, and downstream evaluation. The reward comes from engine strength, not proxy text metrics.

### 6. Close

The result is a self-improvement environment where the model learns a real engineering workflow instead of just outputting moves or text.
