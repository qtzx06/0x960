# one-minute demo script

## 30-second version

We built 0x960, an OpenEnv environment where a model learns to be a Chess960 engine engineer, not a chess player. Chess960 is useful because it removes much of the opening memorization that standard chess systems can rely on, making it a stronger test of generalization. In our environment, the model gets a bounded coding workspace, edits the engine's eval function, runs checks, and is rewarded by whether the edited engine actually performs better against a fixed baseline. The training signal comes from real downstream engine strength, not just text imitation or next-move prediction.

## full one-minute outline

### 1. opening

0x960 is an OpenEnv environment for training models to improve a Chess960 engine through bounded code edits.

### 2. why this task

Chess960 keeps the rules of chess the same but randomizes the starting position, so it is a cleaner test of robustness than standard chess alone.

### 3. what the model does

The model does not play chess directly. It reads engine files, edits `eval.py`, runs checks, and decides when to finish.

### 4. reward

After the edit budget is used, the engine plays fast matches against a fixed baseline. Reward is based on match score, with penalties for invalid edits or crashes.

### 5. why OpenEnv

This is a real multi-step tool-use task with files, commands, failures, and downstream evaluation. That makes it a strong fit for Statement 3.1 and Statement 4.

### 6. close

The result is a self-improvement environment where the model learns to engineer a stronger Chess960 system, not just imitate chess moves.
