# 0x960 story deck

## slide 1 — 0x960: self-improving chess960 engine with openenv
- 0x960 is an ai system that engineers stronger chess engines, not just better text outputs.
- the agent operates inside a bounded OpenEnv episode with verifiable tool actions.
- reward is grounded in downstream match performance on held-out chess960 starts.
- stack: OpenEnv 0.2.1 + TRL/GRPO + qwen students + gpt teacher + codex swarm.

> notes:
> set the tone: this is an engineering system with measurable outcomes, not a prompt toy.

---

## slide 2 — who i am + why this problem
- i’m joshua lin (`qtzx06`), data science + math-cs at UC san diego.
- i build systems at the intersection of llm agents, infra, and verifiable evaluation.
- thesis: self-improvement only matters if the loop is constrained, testable, and hard to game.
- 0x960 is my attempt to operationalize that thesis end-to-end.

> notes:
> keep intro short and technical; establish builder credibility and project intent.

---

## slide 3 — problem framing: rl for tool-use usually fails at action discovery
- base models in this environment often never discover `write_file -> run_match -> finish`.
- naive rl optimizes degenerate behavior (eval spam, early finish) instead of useful edits.
- text proxy rewards miss the real objective: stronger engine behavior under match play.
- we need a loop where exploration, reward, and validation all live in the same environment.

> notes:
> emphasize that failure mode is behavioral cold-start, not just model size.

---

## slide 4 — why chess960 (not classical chess)
- chess960 removes opening-book memorization and forces broader positional generalization.
- randomized starts reduce reward hacking through rote move sequences.
- benchmark outcomes better reflect true evaluation + search quality under distribution shift.
- this makes it a stronger substrate for self-improving agents.

- reference: `docs/why_chess960.md`

---

## slide 5 — bounded openenv setup (the contract)
- per episode, the policy gets a constrained workspace and 5 actions: `read_file`, `write_file`, `run_static_eval`, `run_match`, `finish`.
- invalid writes are rolled back; reward is tied to executable engine outcomes.
- observation includes code state, action history, and remaining budget.
- this creates a tight, auditable optimization loop for agent behavior.

- references: `docs/architecture.md`, `openenv.yaml`, `src/zero960_env/server/app.py`

---

## slide 6 — training pipeline: teacher -> sft -> grpo
- teacher (gpt-5.4) generates successful bounded-action trajectories.
- student sft (`Qwen/Qwen3.5-0.8B`) distills workflow competence from those traces.
- grpo then refines policy quality once the agent has a viable action prior.
- key insight: sft solved the bottleneck; rl becomes useful after behavior bootstrap.

- references: `docs/training.md`, `train/codex_distill.py`, `train/sft_student.py`, `train/minimal_trl_openenv.py`

---

## slide 7 — codex swarm: autonomous outer-loop engine search
- multiple codex workers propose bounded patches to eval/search surfaces.
- coordinator enforces staged screening and promotion gates on held-out positions.
- only benchmark-positive candidates become new champions; regressions are discarded.
- this gives continuous engine-level self-improvement in parallel with policy learning.

- references: `docs/codex-swarm-plan.md`, `train/codex_swarm.py`, `outputs/codex_swarm/`

---

## slide 8 — key results metrics (behavior + strength)
- training reward: **-2.1 -> +1.0** (base qwen 0.8b to sft student).
- sft metrics: train loss **0.2072**, eval loss **0.04192**, token acc **98.76%**.
- engine strength: **+596.5 elo** internal gain vs baseline search stack.
- anchor gain: **+221.1 elo** vs stockfish 1320; local strength near ~1600.
- swarm throughput: 4 promoted champions across 9+ rounds / 50+ evaluated patches.

- references: `README.md`, `docs/training.md`, `media/submission/submission_summary.txt`

---

## slide 9 — visuals to include (proof over claims)
- compound uplift: `figures/fig_compound_uplift.png`
- reward regime shift: `figures/fig_reward_regime_shift.png`
- anchor positioning: `figures/fig_anchor_positioning.png`
- behavior quality + velocity: `figures/fig_behavior_quality_shift.png`, `figures/fig_execution_velocity.png`

> notes:
> ideally animate this as a left-to-right story: behavior unlock -> engine uplift -> external anchoring.

---

## slide 10 — demo flow (live and deterministic)
- start server + show bounded action schema in one episode trace.
- run live match demo clip to prove engine behavior under chess960 starts.
- show progression artifacts and anchor comparison outputs.
- close with swarm promotion ledger to demonstrate autonomous improvement cycle.

- references: `docs/demo-script.md`, `media/submission/0x960-demo-62s.mp4`, `media/submission/live_demo/0x960_live_match_stockfish.mp4`, `media/submission/live_demo_v2/0x960_live_match_doublecut.mp4`

---

## slide 11 — architecture + loop diagrams
- policy/environment loop diagram: `docs/diagrams/mermaid.md` (system architecture).
- training pipeline diagram: teacher rollouts -> sft dataset -> student -> grpo.
- champion/challenger loop: candidate patch -> screen -> final benchmark -> promote/reject.
- show that all loops are measurable, bounded, and composable.

- references: `docs/diagrams/mermaid.md`, `docs/architecture.md`

---

## slide 12 — roadmap (next 30/60/90)
- 30 days: scale teacher trace coverage + more diverse chess960 position sets.
- 60 days: stabilize grpo refinement on top of sft students with stronger eval harness.
- 90 days: distill best swarm champions back into policy training corpus for closed-loop compounding.
- productization: harden reproducible dashboards and hosted demo endpoints.

> notes:
> stay bullish but concrete: each milestone tied to measurable benchmark deltas.

---

## slide 13 — links + call to action
- github repo: **TBD add canonical repo url** (expected format: `https://github.com/<org-or-user>/0x960`)
- hugging face space: **TBD add deployed space url**
- colab notebook: **TBD add reproducible run notebook url**
- paper: **placeholder — no paper published yet (add arxiv link when ready)**
- call to action: collaborate on benchmark packs, stronger teachers, and fully autonomous self-improvement rounds.

> notes:
> if links are known at presentation time, replace placeholders with live urls and add QR codes.
