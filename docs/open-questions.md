# open questions

## blockers to resolve first

1. **engine skeleton**

What is the smallest Chess960 engine we can ship quickly while still making eval edits meaningful?

Default assumption:

- python move generation
- fixed search
- pluggable eval module

2. **OpenEnv integration**

What is the thinnest wrapper needed to expose the environment through OpenEnv `0.2.1` and still support a multi-step episode?

We should prefer the simplest compliant implementation over clever abstractions.

3. **training loop shape**

What is the smallest public Colab example that proves the reward loop works with TRL or Unsloth?

The goal is not large-scale training in Colab. The goal is to show a valid training script and some observable reward signal.

4. **baseline and held-out suite**

We need one fixed training baseline and one fixed held-out evaluation suite.

If the baseline is too weak, reward saturates. If it is too strong, the policy gets no signal.

5. **episode speed**

How many games can we afford per episode while keeping iteration tight enough to show learning during the hackathon?

## defaults unless they fail

These are no longer open-ended research questions. They are the default implementation choices until proven insufficient.

1. **model**

Start with a small open model in the `7B-14B` range, with `Qwen3.5-9B` as the default first candidate.

2. **action space**

Use structured actions, not unrestricted shell access.

3. **editable surface**

Restrict writes to `eval.py` and optionally `weights.json`.

4. **reward**

Use fixed-baseline match score with a crash penalty.

5. **comparison models**

Do not use frontier closed models in the core training loop.

## possible upgrades if time remains

1. **parent-checkpoint reward**

Add score against the previous checkpoint or a small checkpoint pool as an auxiliary curriculum signal, not as the only reward.

2. **frontier comparison**

Run a closed frontier coding agent in the same environment for demo purposes only.

3. **visualization**

Plot reward curves, checkpoint strength, and action traces.

4. **league evaluation**

Run small tournaments among checkpoints to show progression over time.

## explicitly deferred

- multi-agent or swarm architectures
- OAuth integration
- unrestricted ACP-style terminal access
- large-model training beyond what a single H100 can support comfortably
- polished benchmark packaging beyond the hackathon submission
