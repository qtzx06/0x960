# 0x960

0x960 is an OpenEnv-oriented environment where a model improves a minimal Chess960 engine by editing a bounded evaluation file and getting rewarded by match outcomes.

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

