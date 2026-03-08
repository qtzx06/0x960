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
    """Quick demo: connect, read eval, run match, finish."""
    with Zero960Client(base_url=base_url) as client:
        result = client.reset()
        obs = result.observation

        result = client.step(Zero960Action(action_type="read_file", path="eval.py"))
        obs = result.observation

        if obs.remaining_steps > 1:
            result = client.step(Zero960Action(action_type="run_static_eval"))
            obs = result.observation

        if obs.remaining_steps > 1:
            result = client.step(Zero960Action(action_type="run_match"))
            obs = result.observation

        result = client.step(Zero960Action(action_type="finish"))
        obs = result.observation

    return RolloutSummary(
        reward=result.reward or 0.0,
        steps=len(obs.history),
        final_status=obs.status_message,
    )


# ---------------------------------------------------------------------------
# TRL GRPO training (Colab-ready)
# ---------------------------------------------------------------------------

def _format_observation_as_messages(obs) -> list[dict[str, str]]:
    """Format a Zero960Observation as chat messages for the LLM."""
    eval_code = obs.file_contents.get("eval.py", "<not read yet>")
    system_msg = (
        "You are a Chess960 evaluation engineer. You can take ONE action per turn.\n"
        "Actions (respond with valid JSON):\n"
        '  {"action_type":"read_file","path":"eval.py"}\n'
        '  {"action_type":"write_file","path":"eval.py","content":"..."}\n'
        '  {"action_type":"run_static_eval"}\n'
        '  {"action_type":"run_match"}\n'
        '  {"action_type":"finish"}\n'
    )
    user_msg = (
        f"Position index: {obs.start_position}\n"
        f"Steps remaining: {obs.remaining_steps}\n"
        f"Last match score: {obs.last_match_score}\n"
        f"History: {obs.history}\n\n"
        f"Current eval.py:\n```python\n{eval_code}\n```\n\n"
        "Choose your next action."
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def _parse_llm_output(text: str) -> Zero960Action:
    """Best-effort parse of LLM output into a Zero960Action."""
    import json
    import re

    # Try to extract JSON from the response
    json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return Zero960Action(**data)
        except (json.JSONDecodeError, ValueError):
            pass
    # Fallback: finish
    return Zero960Action(action_type="finish")


def run_grpo_training(
    base_url: str,
    model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    num_train_steps: int = 20,
    episodes_per_step: int = 4,
) -> None:
    """Run TRL GRPO training against a live Zero960 environment."""
    from trl import GRPOConfig, GRPOTrainer

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build rollout dataset: list of prompt dicts
    # Each item is a conversation that the model will complete
    def generate_prompts(n: int) -> list[dict]:
        """Generate n initial prompts by resetting the environment."""
        prompts = []
        with Zero960Client(base_url=base_url) as client:
            for _ in range(n):
                result = client.reset()
                msgs = _format_observation_as_messages(result.observation)
                prompts.append({"messages": msgs})
        return prompts

    # Reward function: run a full episode using the model's proposed action
    def env_reward_func(completions: list[str], **kwargs) -> list[float]:
        """Score each completion by executing it in the environment."""
        rewards = []
        with Zero960Client(base_url=base_url) as client:
            for completion in completions:
                try:
                    result = client.reset()
                    action = _parse_llm_output(completion)
                    result = client.step(action)

                    # If not done, finish to get reward
                    if not result.done:
                        result = client.step(Zero960Action(action_type="finish"))

                    rewards.append(float(result.reward or 0.0))
                except Exception:
                    rewards.append(0.0)
        return rewards

    print(f"Starting GRPO training with {model_name}")
    print(f"  steps={num_train_steps}, episodes_per_step={episodes_per_step}")
    print(f"  env={base_url}")

    config = GRPOConfig(
        output_dir="./zero960_grpo_output",
        num_train_epochs=1,
        max_steps=num_train_steps,
        per_device_train_batch_size=episodes_per_step,
        learning_rate=5e-6,
        logging_steps=1,
        num_generations=episodes_per_step,
        max_completion_length=512,
        report_to="none",
    )

    dataset = generate_prompts(episodes_per_step * num_train_steps)

    trainer = GRPOTrainer(
        model=model_name,
        config=config,
        reward_funcs=[env_reward_func],
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Training...")
    trainer.train()
    print("Training complete.")
    trainer.save_model("./zero960_grpo_final")
    print("Model saved to ./zero960_grpo_final")


def main() -> None:
    parser = argparse.ArgumentParser(description="0x960 training / rollout script.")
    parser.add_argument(
        "--mode",
        choices=["handcrafted", "train"],
        default="handcrafted",
        help="handcrafted = quick demo rollout, train = TRL GRPO training",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--episodes-per-step", type=int, default=4)
    args = parser.parse_args()

    if args.mode == "handcrafted":
        summary = run_handcrafted_rollout(base_url=args.base_url)
        print(
            {
                "reward": summary.reward,
                "steps": summary.steps,
                "final_status": summary.final_status,
            }
        )
    elif args.mode == "train":
        run_grpo_training(
            base_url=args.base_url,
            model_name=args.model,
            num_train_steps=args.steps,
            episodes_per_step=args.episodes_per_step,
        )


if __name__ == "__main__":
    main()
