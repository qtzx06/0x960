"""0x960 training script — TRL GRPO + OpenEnv multi-turn rollouts.

Modes:
  --mode handcrafted   Quick scripted rollout (no LLM)
  --mode infer         Single episode with Qwen inference (no training)
  --mode train         Full GRPO training with multi-turn env rollouts
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass

from zero960_env.client import Zero960Client
from zero960_env.models import Zero960Action

SYSTEM_PROMPT = (
    "You are a Chess960 evaluation engineer. You can take ONE action per turn.\n"
    "Actions (respond with valid JSON only, no other text):\n"
    '  {"action_type":"read_file","path":"eval.py"}\n'
    '  {"action_type":"write_file","path":"eval.py","content":"<new code>"}\n'
    '  {"action_type":"run_static_eval"}\n'
    '  {"action_type":"run_match"}\n'
    '  {"action_type":"finish"}\n'
    "\n"
    "Goal: improve eval.py so the Chess960 engine beats the baseline.\n"
    "Strategy: read eval.py → edit it → run_match to test → finish when satisfied."
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def format_observation_as_prompt(obs, system_prompt: str = SYSTEM_PROMPT) -> str:
    """Build a user-turn string from an observation."""
    eval_code = obs.file_contents.get("eval.py", "<not read yet>")
    user_msg = (
        f"Position index: {obs.start_position}\n"
        f"Steps remaining: {obs.remaining_steps}\n"
        f"Last match score: {obs.last_match_score}\n"
        f"History: {obs.history}\n\n"
        f"Current eval.py:\n```python\n{eval_code}\n```\n\n"
        "Choose your next action (JSON only)."
    )
    return user_msg


def format_messages(obs) -> list[dict[str, str]]:
    """Format observation as chat messages."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_observation_as_prompt(obs)},
    ]


def parse_llm_output(text: str) -> Zero960Action:
    """Best-effort parse of LLM output into a Zero960Action."""
    # Try to find JSON with nested braces (for write_file with content)
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return Zero960Action(**data)
        except (json.JSONDecodeError, ValueError):
            pass
    # Simpler JSON match
    json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return Zero960Action(**data)
        except (json.JSONDecodeError, ValueError):
            pass
    return Zero960Action(action_type="finish")


# ---------------------------------------------------------------------------
# Mode: handcrafted
# ---------------------------------------------------------------------------

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
# Mode: infer (single episode with a loaded model, no training)
# ---------------------------------------------------------------------------

def run_inference_test(
    base_url: str,
    model_name: str = "Qwen/Qwen3.5-9B",
    max_episode_steps: int = 6,
) -> RolloutSummary:
    """Run a single episode with Qwen generating actions against the live env."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto",
    )

    with Zero960Client(base_url=base_url) as client:
        result = client.reset()
        obs = result.observation

        for step_i in range(max_episode_steps):
            if result.done:
                break

            msgs = format_messages(obs)
            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs, max_new_tokens=1024,
                temperature=0.7, top_p=0.9, do_sample=True,
            )
            generated = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            print(f"\n--- Step {step_i + 1} ---")
            print(f"LLM output: {generated[:300]}...")

            action = parse_llm_output(generated)
            print(f"Parsed action: {action.action_type}", end="")
            if action.path:
                print(f" path={action.path}", end="")
            print()

            result = client.step(action)
            obs = result.observation
            print(f"Status: {obs.status_message}")

        if not result.done:
            result = client.step(Zero960Action(action_type="finish"))
            obs = result.observation

    summary = RolloutSummary(
        reward=result.reward or 0.0,
        steps=len(obs.history),
        final_status=obs.status_message,
    )
    print(f"\nFinal: reward={summary.reward}, steps={summary.steps}")
    return summary


# ---------------------------------------------------------------------------
# Mode: train — TRL GRPO with OpenEnv rollout_func
# ---------------------------------------------------------------------------

def _rollout_one_episode(
    trainer,
    env: Zero960Client,
    tokenizer,
    dataset_prompt: str,
    max_turns: int = 6,
) -> dict[str, list]:
    """Run one multi-turn episode, collecting token-level data for GRPO."""
    from trl.experimental.openenv import generate_rollout_completions

    result = env.reset()
    obs = result.observation

    prompt_ids: list = []
    completion_ids: list = []
    logprobs: list = []
    step_rewards: list[float] = []

    for _turn in range(max_turns):
        if result.done:
            break

        # Build prompt from current observation
        msgs = format_messages(obs)
        prompt_text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )

        # Generate via TRL's vLLM-aware helper
        rollout_outputs = generate_rollout_completions(trainer, [prompt_text])[0]
        prompt_ids.extend(rollout_outputs["prompt_ids"])
        completion_ids.extend(rollout_outputs["completion_ids"])
        logprobs.extend(rollout_outputs["logprobs"])

        completion_text = rollout_outputs.get("text") or tokenizer.decode(
            rollout_outputs["completion_ids"], skip_special_tokens=True,
        )

        # Parse action and step the environment
        action = parse_llm_output(completion_text)
        result = env.step(action)
        obs = result.observation
        step_rewards.append(float(result.reward or 0.0))

    # If the model didn't finish, force finish to get terminal reward
    if not result.done:
        result = env.step(Zero960Action(action_type="finish"))
        step_rewards.append(float(result.reward or 0.0))

    final_reward = step_rewards[-1] if step_rewards else 0.0

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "env_reward": final_reward,
        "num_turns": len(step_rewards),
    }


def run_grpo_training(
    base_url: str,
    model_name: str = "Qwen/Qwen3.5-9B",
    num_train_steps: int = 20,
    num_generations: int = 4,
    max_turns: int = 6,
) -> None:
    """Run TRL GRPO training with multi-turn OpenEnv rollouts."""
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    env = Zero960Client(base_url=base_url)
    env.connect()

    # Build dataset of initial prompts
    def make_dataset(n: int) -> Dataset:
        prompts = []
        for _ in range(n):
            result = env.reset()
            msgs = format_messages(result.observation)
            prompt_text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
            prompts.append(prompt_text)
        return Dataset.from_dict({"prompt": prompts})

    dataset = make_dataset(num_generations * num_train_steps)

    # Rollout function: runs multi-turn episodes for a batch of prompts
    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, list]:
        all_prompt_ids = []
        all_completion_ids = []
        all_logprobs = []
        all_env_rewards = []

        for prompt in prompts:
            episode = _rollout_one_episode(
                trainer=trainer,
                env=env,
                tokenizer=tokenizer,
                dataset_prompt=prompt,
                max_turns=max_turns,
            )
            all_prompt_ids.append(episode["prompt_ids"])
            all_completion_ids.append(episode["completion_ids"])
            all_logprobs.append(episode["logprobs"])
            all_env_rewards.append(episode["env_reward"])

        return {
            "prompt_ids": all_prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
            "env_reward": all_env_rewards,
        }

    # Reward function: extract env_reward passed through rollout_func
    def reward_from_env(completions, **kwargs):
        rewards = kwargs.get("env_reward", [])
        if rewards:
            return [float(r) for r in rewards]
        return [0.0] * len(completions)

    config = GRPOConfig(
        output_dir="./zero960_grpo_output",
        num_train_epochs=1,
        max_steps=num_train_steps,
        per_device_train_batch_size=num_generations,
        learning_rate=5e-6,
        logging_steps=1,
        num_generations=num_generations,
        max_completion_length=1024,
        use_vllm=True,
        vllm_mode="colocate",
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model_name,
        processing_class=tokenizer,
        reward_funcs=[reward_from_env],
        train_dataset=dataset,
        args=config,
        rollout_func=rollout_func,
    )

    print(f"Starting GRPO training: {model_name}")
    print(f"  steps={num_train_steps}, generations={num_generations}, max_turns={max_turns}")
    print(f"  env={base_url}")

    trainer.train()

    trainer.save_model("./zero960_grpo_final")
    print("Training complete. Model saved to ./zero960_grpo_final")

    env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="0x960 training / rollout script.")
    parser.add_argument(
        "--mode",
        choices=["handcrafted", "infer", "train"],
        default="handcrafted",
        help="handcrafted = scripted demo, infer = Qwen inference test, train = TRL GRPO",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=6)
    args = parser.parse_args()

    if args.mode == "handcrafted":
        summary = run_handcrafted_rollout(base_url=args.base_url)
        print({
            "reward": summary.reward,
            "steps": summary.steps,
            "final_status": summary.final_status,
        })
    elif args.mode == "infer":
        summary = run_inference_test(
            base_url=args.base_url,
            model_name=args.model,
        )
        print({
            "reward": summary.reward,
            "steps": summary.steps,
            "final_status": summary.final_status,
        })
    elif args.mode == "train":
        run_grpo_training(
            base_url=args.base_url,
            model_name=args.model,
            num_train_steps=args.steps,
            num_generations=args.num_generations,
            max_turns=args.max_turns,
        )


if __name__ == "__main__":
    main()
