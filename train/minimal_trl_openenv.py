from __future__ import annotations

import argparse
from dataclasses import dataclass

from zero960_env.client import Zero960Client
from zero960_env.models import Zero960Action


@dataclass(slots=True)
class RolloutSummary:
    reward: float
    steps: int
    final_status: str


def run_handcrafted_rollout(base_url: str) -> RolloutSummary:
    client = Zero960Client(base_url=base_url)
    observation = client.reset()

    observation = client.step(Zero960Action(action_type="read_file", path="eval.py"))
    if observation.remaining_steps > 1:
        observation = client.step(Zero960Action(action_type="run_static_eval"))
    if observation.remaining_steps > 1:
        observation = client.step(Zero960Action(action_type="run_match"))
    observation = client.step(Zero960Action(action_type="finish"))

    return RolloutSummary(
        reward=observation.reward or 0.0,
        steps=len(observation.history),
        final_status=observation.status_message,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal OpenEnv rollout stub for 0x960.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    summary = run_handcrafted_rollout(base_url=args.base_url)
    print(
        {
            "reward": summary.reward,
            "steps": summary.steps,
            "final_status": summary.final_status,
        }
    )


if __name__ == "__main__":
    main()
