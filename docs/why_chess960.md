# Why Chess960

## The Problem with Standard Chess

Standard chess engines can be improved by memorizing opening books — thousands of well-known opening sequences that have been optimized over centuries. An RL agent that "improves" a standard chess engine might just be learning to parrot known opening theory rather than developing genuine evaluation ability.

This is exactly the kind of reward hacking we wanted to avoid.

## Why Chess960 Is the Right Benchmark

Chess960 (Fischer Random Chess) keeps the rules of chess identical but randomizes the back-rank piece placement across **960 possible starting positions**. This eliminates:

- Opening book memorization
- Known opening theory exploitation
- Position-specific pattern matching that doesn't generalize

The engine must evaluate positions it has never seen before based on fundamental chess principles — piece activity, king safety, pawn structure, tactical threats. **If the evaluation code is better, it wins more games. Period.**

## What This Means for Self-Improvement

Chess960 is a cleaner robustness test than standard chess for exactly the reason that matters in RL:

- **You can't game the reward.** There's no shortcut where memorizing patterns gets you a higher score without actually understanding chess positions.
- **Generalization is mandatory.** The engine must perform across 960 different starting setups, not just one canonical opening tree.
- **The signal is real.** Win/loss/draw outcomes on held-out Chess960 positions are ground truth — not proxy metrics, not text quality scores, not human preferences.

## How 0x960 Uses This

The agent doesn't play chess. It writes evaluation code that a chess engine uses to play. The reward comes from whether that code makes the engine win more games on held-out Chess960 positions.

This is the bridge from the research motivation to the OpenEnv environment design:
- Chess960 provides a clean, non-gameable benchmark
- Bounded code editing provides the action space
- Real match outcomes provide the reward signal
- The agent has to write code that *actually understands chess positions* to improve

## Results

The system works. Starting from a basic eval function, the combination of teacher-student policy learning and autonomous Codex swarm search pushed the engine to:

- **+596.5 Elo** vs the search baseline (internal)
- **+221.1 Elo** vs Stockfish 1320 (external anchor)
- **Competitive with Stockfish 1600** in local Chess960 benchmarks

All verified on held-out Chess960 positions that the agent never trained on.
