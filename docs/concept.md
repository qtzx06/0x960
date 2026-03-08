# 0x960

## what this is

0x960 is an OpenEnv environment where a model improves a minimal Chess960 engine by making bounded code edits to its evaluation logic.

The agent does not play chess directly. It reads engine files, edits the eval function, runs checks, and is rewarded by match performance against a fixed baseline.

## hackathon fit

This project is designed to satisfy the OpenEnv hackathon constraints:

- use OpenEnv `0.2.1`
- deploy the environment on HF Spaces
- provide a minimal training script in Colab using HF TRL or Unsloth

## core claim

The interesting task is not "can an LLM output a good chess move?"

The interesting task is:

- can a model operate inside a real coding environment
- make multi-step edits to a live system
- and improve that system under an objective downstream metric

Chess960 is useful because opening memorization is much less valuable than in standard chess. That makes engine improvement a better fit for an agentic environment than pure next-move prediction.

## novelty claim

We should not claim that tool use is new or that Chess960 benchmarking is new.

The stronger and more defensible claim is:

- Chess960 engine evaluation is an existing benchmark domain
- coding agents with tool use are an existing capability pattern
- 0x960 combines them into a self-improvement RL environment where the model modifies engine code and is rewarded by actual engine strength

## why this is a good OpenEnv task

- it is multi-step, not single-step classification
- reward comes from a real external process: engine matches
- the agent interacts with files and commands, not just text
- failure modes are meaningful: bad edits, crashes, invalid code, weak evals

This aligns best with:

- Statement 3.1: Professional Tasks
- Statement 4: Self-Improvement

## MVP

The MVP should be intentionally narrow.

- one minimal Chess960 engine scaffold
- one fixed search implementation
- one narrow editable surface: `eval.py` or `weights.json`
- one fixed baseline opponent
- one held-out evaluation suite
- one training path using OpenEnv + TRL/Unsloth

## non-goals for MVP

- no frontier model dependency in the training loop
- no OAuth or hosted coding-agent integration
- no multi-agent swarm
- no broad repo-wide code editing
- no polished Elo dashboard unless the core loop already works

## practical pitch

"We built an OpenEnv environment where a model learns to be a Chess960 engine engineer, not a chess player. The model uses bounded coding actions to improve an engine's eval function, and reward comes from whether the edited engine actually performs better."
