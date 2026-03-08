# architecture

## stack decisions

These are fixed by the hackathon or by scope discipline:

- environment interface: OpenEnv `0.2.1`
- deployment target: HF Spaces
- training demo: HF TRL or Unsloth in Colab
- core model class: open-weight OSS model only
- optional infra for real training: Northflank H100

Closed frontier models are not part of the core training path. If we use them at all, they are comparison-only in the demo layer.

## system shape

The system should have four layers.

### 1. engine workspace

A minimal Chess960 engine scaffold with:

- fixed move generation
- fixed search implementation
- fixed tournament runner
- one narrow editable surface

Recommended editable surface:

- `engine/eval.py`

Optional later extension:

- `engine/weights.json`

The whole repo should not be editable by the policy. Narrow edit scope keeps training stable and makes the story legible.

### 2. environment runtime

The environment owns the full episode lifecycle:

1. clone a fresh engine workspace
2. sample one Chess960 start or a small suite of starts
3. expose bounded actions to the model
4. execute actions and return observations
5. after the step budget is exhausted, run matches
6. compute reward and terminate

The environment should be written as a normal Python runtime first, then wrapped cleanly for OpenEnv.

### 3. reward and evaluation harness

This layer runs fast matches between the edited engine and baselines.

It should provide:

- training reward matches
- held-out evaluation matches
- crash handling
- reproducible position sampling

### 4. training loop

The training loop should use:

- GRPO or equivalent in TRL/Unsloth
- a rollout function that runs a full episode
- checkpoint logging
- reward curves and crash-rate metrics

The training loop is minimal by design. The goal is to show the environment can produce a learnable signal, not to max out Elo during the hackathon.

## episode contract

### observation

Each step should return a compact structured observation containing:

- the task instruction
- current editable file contents
- recent action history
- recent command outputs or error messages
- remaining step budget
- start-position metadata

### actions

Start with structured actions, not open shell access.

- `read_file(path)`
- `write_file(path, content)`
- `run_static_eval()`
- `run_match()`
- `finish()`

If needed, a restricted shell tool can be added later, but it should not be required for MVP.

### termination

An episode ends when:

- the agent calls `finish()`
- the step budget is exhausted
- the workspace becomes invalid in a fatal way

## reward design

Default reward for MVP:

- primary reward: match score against a fixed baseline engine
- penalty: invalid edit, crash, or timeout

Recommended first-pass formula:

`reward = score_vs_fixed_baseline - crash_penalty`

Do not make parent-checkpoint self-play the only reward. If we use it, it should be a secondary signal only.

## evaluation protocol

Training and evaluation must be separated.

### training

- sample Chess960 starts from a training pool
- play a small number of fast games against the fixed baseline
- compute reward

### held-out eval

- separate fixed start-position suite
- fixed baseline configuration
- fixed game count and time control
- run periodically on saved checkpoints

This is how we avoid fooling ourselves with a rising training reward that does not correspond to stronger engines.

## model strategy

We should optimize for stable tool behavior, not for the largest model possible.

Recommended order:

1. `Qwen3.5-9B`
2. one backup model with good coding/tool-use behavior
3. only try larger models if the smaller path is already stable

Single-H100-safe priority:

- dense 7B to 14B class models first
- larger MoE models only if integration is already working

## speed target

A good MVP episode should be cheap enough to run many times.

Target envelope:

- step budget: `4-8` actions
- match count: very small during training
- episode runtime: ideally under `30s`

If episodes are too slow, we reduce game count before we add complexity elsewhere.

## deployment

### HF Spaces

HF Spaces hosts the OpenEnv environment and provides the submission artifact judges can inspect.

### Colab

Colab provides the minimal public training notebook using TRL or Unsloth.

### Northflank

Northflank is the practical training box if we want a real H100-backed run, but it is not required for the minimal architecture itself.

## deferred work

These are explicitly outside the MVP:

- frontier model integrations
- OAuth-based coding agent sessions
- multi-agent swarm variants
- Elo dashboards
- tournament leagues across many checkpoints
- full ACP-like unrestricted workspace tooling
