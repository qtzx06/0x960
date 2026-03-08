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

def run_grpo_training(
    base_url: str,
    model_name: str = "Qwen/Qwen3.5-9B",
    num_train_steps: int = 20,
    num_generations: int = 4,
    max_turns: int = 6,
) -> None:
    """Run TRL GRPO training with environment-based rewards.

    Each GRPO generation produces a completion (the model's proposed action
    sequence). The reward function runs a full multi-turn episode against
    the live environment to score it.
    """
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    env = Zero960Client(base_url=base_url)
    env.connect()

    # Build dataset: each row is a prompt from a fresh env reset
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

    # Reward function: run multi-turn episode from the model's completion
    def env_reward_func(completions, **kwargs):
        """Score each completion by running a full episode in the env."""
        rewards = []
        for completion in completions:
            try:
                result = env.reset()
                obs = result.observation

                # First action from the model's completion
                action = parse_llm_output(completion)
                result = env.step(action)
                obs = result.observation

                # If the model wrote code, run a match to get a real score
                if not result.done and action.action_type == "write_file":
                    result = env.step(Zero960Action(action_type="run_match"))
                    obs = result.observation

                # Finish to get terminal reward
                if not result.done:
                    result = env.step(Zero960Action(action_type="finish"))

                rewards.append(float(result.reward or 0.0))
            except Exception:
                rewards.append(0.0)
        return rewards

    from peft import LoraConfig

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # full attention
            "in_proj_qkv", "out_proj",                # linear attention
            "gate_proj", "up_proj", "down_proj",      # MLP
        ],
        task_type="CAUSAL_LM",
    )

    model_kwargs = {
        "quantization_config": __import__("transformers").BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=__import__("torch").bfloat16,
            bnb_4bit_quant_type="nf4",
        ),
        "device_map": "auto",
    }

    config = GRPOConfig(
        output_dir="./zero960_grpo_output",
        num_train_epochs=1,
        max_steps=num_train_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=num_generations,
        learning_rate=5e-6,
        logging_steps=1,
        num_generations=num_generations,
        max_completion_length=1024,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model_name,
        processing_class=tokenizer,
        reward_funcs=[env_reward_func],
        train_dataset=dataset,
        args=config,
        peft_config=lora_config,
        model_init_kwargs=model_kwargs,
    )

    print(f"Starting GRPO training: {model_name}")
    print(f"  steps={num_train_steps}, generations={num_generations}")
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
