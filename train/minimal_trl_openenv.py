"""0x960 training script — TRL GRPO + OpenEnv multi-turn rollouts.

Modes:
  --mode handcrafted   Quick scripted rollout (no LLM)
  --mode infer         Single episode with Qwen inference (no training)
  --mode train         Full GRPO training with multi-turn env rollouts
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from zero960_env.client import Zero960Client
from zero960_env.models import Zero960Action

EXAMPLE_WRITE_ACTION = json.dumps(
    {
        "action_type": "write_file",
        "path": "eval.py",
        "content": (
            "from __future__ import annotations\n\n"
            "import chess\n\n"
            "PIECE_VALUES = {\n"
            "    chess.PAWN: 100,\n"
            "    chess.KNIGHT: 320,\n"
            "    chess.BISHOP: 330,\n"
            "    chess.ROOK: 500,\n"
            "    chess.QUEEN: 900,\n"
            "    chess.KING: 0,\n"
            "}\n\n"
            "def evaluate(board: chess.Board) -> int:\n"
            "    score = 0\n"
            "    for piece_type, piece_value in PIECE_VALUES.items():\n"
            "        score += piece_value * len(board.pieces(piece_type, chess.WHITE))\n"
            "        score -= piece_value * len(board.pieces(piece_type, chess.BLACK))\n"
            "    return score\n"
        ),
    }
)

ACTION_SCHEMA_TEXT = (
    "Return exactly one JSON object matching one of these shapes:\n"
    '1. {"action_type":"write_file","path":"eval.py","content":"<full eval.py source>"}\n'
    '2. {"action_type":"run_match"}\n'
    '3. {"action_type":"finish"}\n'
    '4. {"action_type":"run_static_eval"}\n'
    '5. {"action_type":"read_file","path":"eval.py"}'
)

ACTION_CHOICE_MAP = {
    "1": "write_file",
    "2": "run_match",
    "3": "finish",
    "4": "run_static_eval",
    "5": "read_file",
}

TRAIN_ACTION_REWARD_BIAS = {
    "write_file": 0.35,
    "run_match": -0.15,
    "finish": -0.30,
    "run_static_eval": -0.25,
    "read_file": -0.30,
}

SYSTEM_PROMPT = (
    "You are a Chess960 evaluation engineer. You can take ONE action per turn.\n"
    "Respond with exactly one JSON object and no extra text.\n"
    f"{ACTION_SCHEMA_TEXT}\n"
    "Actions:\n"
    '  {"action_type":"read_file","path":"eval.py"}\n'
    '  {"action_type":"write_file","path":"eval.py","content":"<full replacement eval.py>"}\n'
    '  {"action_type":"run_static_eval"}\n'
    '  {"action_type":"run_match"}\n'
    '  {"action_type":"finish"}\n'
    "\n"
    "Important rules:\n"
    "- The full current eval.py is already included in the observation, so read_file is usually unnecessary.\n"
    "- High-reward loop: write_file a valid full replacement, run_match, then finish.\n"
    "- Repeating run_static_eval, finishing before a write, or finishing before an explicit match is penalized.\n"
    "- If you write code, keep it short and valid Python that defines evaluate(board).\n"
    "- Do not output analysis, markdown, XML tags, or prose. Do not emit <think> blocks.\n"
    "\n"
    "Examples:\n"
    f"Fresh episode best first move:\n{EXAMPLE_WRITE_ACTION}\n"
    'After a valid write, best next move:\n{"action_type":"run_match"}\n'
    'After a match score is available, best next move:\n{"action_type":"finish"}'
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
        f"Has valid edit: {obs.has_valid_edit}\n"
        f"Has explicit match: {obs.has_run_match}\n"
        f"Suggested actions: {', '.join(obs.suggested_actions)}\n"
        f"Workflow hint: {obs.workflow_hint}\n"
        f"History: {obs.history}\n\n"
        f"Current eval.py:\n```python\n{eval_code}\n```\n\n"
        f"{ACTION_SCHEMA_TEXT}\n"
        "Choose your next action. Output JSON only."
    )
    return user_msg


def format_messages(obs) -> list[dict[str, str]]:
    """Format observation as chat messages."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_observation_as_prompt(obs)},
    ]


def format_action_selection_messages(obs) -> list[dict[str, str]]:
    """Ask the model to choose only the next action type ID."""
    eval_code = obs.file_contents.get("eval.py", "<not read yet>")
    return [
        {
            "role": "system",
            "content": (
                "Choose the next action for a Chess960 eval-editing task.\n"
                "Return exactly one digit and nothing else.\n"
                "1 = write_file\n"
                "2 = run_match\n"
                "3 = finish\n"
                "4 = run_static_eval\n"
                "5 = read_file"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Steps remaining: {obs.remaining_steps}\n"
                f"Last match score: {obs.last_match_score}\n"
                f"Has valid edit: {obs.has_valid_edit}\n"
                f"Has explicit match: {obs.has_run_match}\n"
                f"Suggested actions: {', '.join(obs.suggested_actions)}\n"
                f"Workflow hint: {obs.workflow_hint}\n"
                f"History: {obs.history}\n\n"
                f"Current eval.py:\n```python\n{eval_code}\n```\n\n"
                "Choose the best next action ID. Return exactly one digit."
            ),
        },
    ]


def format_write_messages(obs) -> list[dict[str, str]]:
    """Ask the model to output a full replacement eval.py file only."""
    eval_code = obs.file_contents.get("eval.py", "<not read yet>")
    write_prefix = build_write_prefix(eval_code)
    return [
        {
            "role": "system",
            "content": (
                "Continue a Python file for a Chess960 engine.\n"
                "The assistant response is appended directly after a provided prefix.\n"
                "Output only the remaining Python lines after the prefix.\n"
                "Do not repeat the prefix. No markdown, no prose, no JSON, no <think>."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Steps remaining: {obs.remaining_steps}\n"
                f"Last match score: {obs.last_match_score}\n"
                f"Workflow hint: {obs.workflow_hint}\n"
                "Improve the evaluation function while keeping valid Python that defines evaluate(board).\n"
                "You are completing the file after this exact prefix:\n\n"
                f"```python\n{write_prefix}```"
            ),
        },
    ]


def apply_action_chat_template(tokenizer, messages: list[dict[str, str]]) -> str:
    """Apply Qwen chat template while disabling thinking when the template supports it."""
    template_attempts = [
        {"chat_template_kwargs": {"enable_thinking": False}},
        {"enable_thinking": False},
        {},
    ]
    for extra_kwargs in template_attempts:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **extra_kwargs,
            )
        except TypeError:
            continue
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def strip_reasoning(text: str) -> str:
    """Remove common reasoning wrappers before JSON parsing."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<\|im_start\|>assistant\s*", "", cleaned)
    cleaned = re.sub(r"<\|im_end\|>", "", cleaned)
    return cleaned.strip()


def extract_python_source(text: str) -> str:
    """Extract raw Python source from model output."""
    cleaned = strip_reasoning(text)
    fenced = re.findall(r"```(?:python)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fenced:
        return fenced[0].strip()
    return cleaned.strip()


def build_write_prefix(current_code: str) -> str:
    """Build a stable file prefix that the model must continue."""
    marker = "def evaluate(board: chess.Board) -> int:\n"
    match = re.search(re.escape(marker), current_code)
    if match:
        return current_code[:match.end()] + "    score = 0\n"
    return (
        "from __future__ import annotations\n\n"
        "import chess\n\n"
        "PIECE_VALUES = {\n"
        "    chess.PAWN: 100,\n"
        "    chess.KNIGHT: 320,\n"
        "    chess.BISHOP: 330,\n"
        "    chess.ROOK: 500,\n"
        "    chess.QUEEN: 900,\n"
        "    chess.KING: 0,\n"
        "}\n\n"
        "def evaluate(board: chess.Board) -> int:\n"
        "    score = 0\n"
    )


def extract_python_continuation(text: str) -> str:
    """Extract only indented Python lines for the evaluate() body continuation."""
    cleaned = extract_python_source(text)
    lines = cleaned.splitlines()
    kept: list[str] = []
    started = False

    for line in lines:
        if not started:
            if not line.strip():
                continue
            if line.startswith("    ") or line.startswith("\t"):
                started = True
                kept.append(line)
                continue
            if re.match(r"(for|if|elif|else|while|return|score|white_|black_|center_|mobility_|piece_|pawn_|king_|board)", line.strip()):
                started = True
                kept.append(f"    {line.strip()}")
                continue
            continue

        if line.strip() and not (line.startswith("    ") or line.startswith("\t")):
            break
        kept.append(line)

    return "\n".join(kept).rstrip() + "\n" if kept else ""


def fallback_eval_tail(current_code: str) -> str:
    """Reuse the existing evaluate() body tail as a safe syntax fallback."""
    marker = "    score = 0\n"
    index = current_code.find(marker)
    if index == -1:
        return "    return score\n"
    return current_code[index + len(marker):].rstrip() + "\n"


def choose_action_id(model, tokenizer, prompt: str) -> tuple[str, dict[str, float]]:
    """Score a fixed set of action IDs and return the most likely one."""
    import torch

    prompt_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_input_ids = prompt_inputs["input_ids"]
    prompt_attention_mask = prompt_inputs["attention_mask"]
    option_scores: dict[str, float] = {}

    with torch.no_grad():
        for option_id in ACTION_CHOICE_MAP:
            option_tokens = tokenizer(option_id, add_special_tokens=False, return_tensors="pt")
            option_input_ids = option_tokens["input_ids"].to(model.device)
            option_attention_mask = option_tokens["attention_mask"].to(model.device)

            full_input_ids = torch.cat([prompt_input_ids, option_input_ids], dim=1)
            full_attention_mask = torch.cat([prompt_attention_mask, option_attention_mask], dim=1)
            outputs = model(input_ids=full_input_ids, attention_mask=full_attention_mask)
            log_probs = outputs.logits[:, prompt_input_ids.shape[1] - 1:-1, :].log_softmax(dim=-1)

            token_log_prob = 0.0
            for index in range(option_input_ids.shape[1]):
                token_id = option_input_ids[0, index]
                token_log_prob += float(log_probs[0, index, token_id])
            option_scores[option_id] = token_log_prob / max(option_input_ids.shape[1], 1)

    best_option = max(option_scores, key=option_scores.get)
    return best_option, option_scores


def parse_action_choice(text: str) -> str:
    """Parse a one-token action ID from a completion."""
    cleaned = strip_reasoning(text)
    digit_match = re.search(r"\b([1-5])\b", cleaned)
    if digit_match:
        return digit_match.group(1)

    lowered = cleaned.lower()
    for action_id, action_type in ACTION_CHOICE_MAP.items():
        if action_type in lowered:
            return action_id
    return "3"


def build_training_write_code(current_code: str, variant_index: int = 0) -> str:
    """Apply a deterministic valid edit so GRPO can learn the task loop first."""
    candidates = [
        (
            "CENTER_ATTACK_BONUS = 3",
            "CENTER_ATTACK_BONUS = 4",
        ),
        (
            "BISHOP_PAIR_BONUS = 35",
            "BISHOP_PAIR_BONUS = 45",
        ),
        (
            "ROOK_OPEN_FILE_BONUS = 20",
            "ROOK_OPEN_FILE_BONUS = 24",
        ),
        (
            "PASSED_PAWN_BONUS_BY_RANK = [0, 5, 10, 18, 28, 42, 60, 0]",
            "PASSED_PAWN_BONUS_BY_RANK = [0, 6, 12, 20, 32, 48, 68, 0]",
        ),
    ]

    for offset in range(len(candidates)):
        source, target = candidates[(variant_index + offset) % len(candidates)]
        if source in current_code:
            candidate_code = current_code.replace(source, target, 1)
            try:
                ast.parse(candidate_code, filename="eval.py")
            except SyntaxError:
                continue
            if candidate_code != current_code:
                return candidate_code
    return current_code


def build_training_action(choice_id: str, obs, variant_index: int = 0) -> Zero960Action:
    """Convert an action-choice completion into a concrete env action."""
    action_type = ACTION_CHOICE_MAP.get(choice_id, "finish")
    if action_type == "write_file":
        current_code = obs.file_contents.get("eval.py", "")
        content = build_training_write_code(current_code, variant_index=variant_index)
        return Zero960Action(action_type="write_file", path="eval.py", content=content)
    if action_type == "read_file":
        return Zero960Action(action_type="read_file", path="eval.py")
    return Zero960Action(action_type=action_type)


def generate_write_action(model, tokenizer, obs) -> tuple[Zero960Action, str]:
    """Generate the full eval.py replacement after action type selection."""
    current_code = obs.file_contents.get("eval.py", "")
    write_prefix = build_write_prefix(current_code)
    write_prompt = apply_action_chat_template(tokenizer, format_write_messages(obs)) + write_prefix
    inputs = tokenizer(write_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
    )
    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    continuation = extract_python_continuation(generated)
    if not continuation:
        continuation = fallback_eval_tail(current_code)
    code = write_prefix + continuation
    try:
        ast.parse(code, filename="eval.py")
    except SyntaxError:
        code = write_prefix + fallback_eval_tail(current_code)
    return Zero960Action(action_type="write_file", path="eval.py", content=code), generated


def choose_structured_action(
    model,
    tokenizer,
    obs,
    deterministic_write: bool = False,
) -> tuple[Zero960Action, dict[str, float], str | None]:
    """Choose action type via fixed-option scoring, then generate code only if needed."""
    action_prompt = apply_action_chat_template(tokenizer, format_action_selection_messages(obs))
    action_id, scores = choose_action_id(model, tokenizer, action_prompt)
    adjusted_scores = dict(scores)

    # Make the policy respect the environment workflow instead of repeatedly editing.
    if obs.has_valid_edit and not obs.has_run_match:
        adjusted_scores["2"] += 3.0
        adjusted_scores["1"] -= 2.0
        adjusted_scores["5"] -= 1.5
        adjusted_scores["4"] -= 1.5
    elif obs.has_run_match:
        if obs.last_match_score is not None and (obs.last_match_score >= 0.25 or obs.remaining_steps <= 2):
            adjusted_scores["3"] += 2.5
            adjusted_scores["1"] -= 1.0
            adjusted_scores["5"] -= 1.0
            adjusted_scores["4"] -= 1.0
        else:
            adjusted_scores["1"] += 1.0
            adjusted_scores["3"] += 0.5

    action_id = max(adjusted_scores, key=adjusted_scores.get)
    action_type = ACTION_CHOICE_MAP[action_id]
    if action_type == "write_file":
        if deterministic_write:
            action = build_training_action("1", obs, variant_index=max(obs.remaining_steps, 0))
            return action, adjusted_scores, "[deterministic write template]"
        action, raw_code_output = generate_write_action(model, tokenizer, obs)
        return action, adjusted_scores, raw_code_output
    if action_type == "read_file":
        return Zero960Action(action_type="read_file", path="eval.py"), adjusted_scores, None
    return Zero960Action(action_type=action_type), adjusted_scores, None


def _extract_balanced_json_objects(text: str) -> list[str]:
    """Return brace-balanced JSON object candidates from free-form model output."""
    candidates: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escape = False

    for index, char in enumerate(text):
        if start is None:
            if char == "{":
                start = index
                depth = 1
                in_string = False
                escape = False
            continue

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidates.append(text[start:index + 1])
                start = None

    return candidates


def parse_llm_output(text: str) -> Zero960Action:
    """Best-effort parse of LLM output into a Zero960Action."""
    cleaned = strip_reasoning(text)
    fenced_match = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    for candidate in fenced_match + _extract_balanced_json_objects(cleaned):
        try:
            data = json.loads(candidate)
            return Zero960Action(**data)
        except (json.JSONDecodeError, ValueError):
            continue

    action_match = re.search(r'"action_type"\s*:\s*"([^"]+)"', cleaned)
    if action_match:
        action_type = action_match.group(1)
        if action_type in {"run_static_eval", "run_match", "finish"}:
            return Zero960Action(action_type=action_type)

    lowered = cleaned.lower()
    if "run_match" in lowered:
        return Zero960Action(action_type="run_match")
    if "write_file" in lowered and "eval.py" in lowered:
        return Zero960Action(action_type="read_file", path="eval.py")
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
    """Quick demo: apply a tiny valid edit, run a match, then finish."""
    with Zero960Client(base_url=base_url) as client:
        result = client.reset()
        obs = result.observation

        current_code = obs.file_contents["eval.py"]
        edited_code = current_code.replace("score += 15 *", "score += 20 *", 1)
        result = client.step(
            Zero960Action(
                action_type="write_file",
                path="eval.py",
                content=edited_code,
            )
        )
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
    tokenizer_name: str | None = None,
    max_episode_steps: int = 6,
    deterministic_write: bool = True,
) -> RolloutSummary:
    """Run a single episode with Qwen generating actions against the live env."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto",
    )

    with Zero960Client(base_url=base_url) as client:
        result = client.reset()
        obs = result.observation

        for step_i in range(max_episode_steps):
            if result.done:
                break

            print(f"\n--- Step {step_i + 1} ---")
            action, action_scores, raw_code_output = choose_structured_action(
                model,
                tokenizer,
                obs,
                deterministic_write=deterministic_write,
            )
            score_text = ", ".join(
                f"{choice}:{score:.3f}" for choice, score in sorted(action_scores.items())
            )
            print(f"Action scores: {score_text}")
            print(f"Parsed action: {action.action_type}", end="")
            if action.path:
                print(f" path={action.path}", end="")
            print()
            if raw_code_output is not None:
                print(f"Write output: {raw_code_output[:300]}...")

            result = client.step(action)
            obs = result.observation
            print(f"Status: {obs.status_message}")
            print(f"Step reward: {result.reward}")

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
            msgs = format_action_selection_messages(result.observation)
            prompt_text = apply_action_chat_template(tokenizer, msgs)
            prompts.append(prompt_text)
        return Dataset.from_dict({"prompt": prompts})

    dataset = make_dataset(num_generations * num_train_steps)

    # Observability: log every completion + action + reward to JSONL
    log_dir = Path("./zero960_logs")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"rollouts_{int(time.time())}.jsonl"
    step_counter = {"n": 0}

    def _log_entry(entry: dict) -> None:
        with open(log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    # Reward function: run multi-turn episode from the model's completion
    def env_reward_func(completions, **kwargs):
        """Score each completion by running a full episode in the env."""
        rewards = []
        step_counter["n"] += 1
        step_n = step_counter["n"]

        for i, completion in enumerate(completions):
            t0 = time.time()
            entry = {
                "step": step_n,
                "gen": i,
                "completion_preview": completion[:500],
                "completion_len": len(completion),
                "step_rewards": [],
            }
            try:
                result = env.reset()
                obs = result.observation

                choice_id = parse_action_choice(completion)
                action = build_training_action(choice_id, obs, variant_index=step_n + i)
                entry["choice_id"] = choice_id
                entry["parsed_action"] = action.action_type
                entry["parsed_path"] = action.path
                if action.action_type == "write_file" and action.content:
                    entry["code_preview"] = action.content[:300]
                    entry["code_len"] = len(action.content)
                    entry["code_changed"] = action.content != obs.file_contents.get("eval.py", "")

                result = env.step(action)
                obs = result.observation
                entry["env_status_1"] = obs.status_message
                entry["step_rewards"].append(float(result.reward or 0.0))

                # If the model wrote code, run a match to get a real score
                if not result.done and action.action_type == "write_file":
                    result = env.step(Zero960Action(action_type="run_match"))
                    obs = result.observation
                    entry["match_score"] = obs.last_match_score
                    entry["step_rewards"].append(float(result.reward or 0.0))

                # Finish to get terminal reward
                if not result.done:
                    result = env.step(Zero960Action(action_type="finish"))
                    entry["step_rewards"].append(float(result.reward or 0.0))

                reward = float(result.reward or 0.0) + TRAIN_ACTION_REWARD_BIAS[action.action_type]
                rewards.append(reward)
                entry["reward"] = reward
                entry["reward_bias"] = TRAIN_ACTION_REWARD_BIAS[action.action_type]
            except Exception as exc:
                rewards.append(0.0)
                entry["reward"] = 0.0
                entry["error"] = str(exc)

            entry["elapsed_s"] = round(time.time() - t0, 2)
            _log_entry(entry)

            # Print compact summary to stdout
            act = entry.get("parsed_action", "?")
            rew = entry["reward"]
            elapsed = entry["elapsed_s"]
            code_info = ""
            if act == "write_file":
                code_info = f" [{entry.get('code_len', 0)}b]"
            print(f"  [{step_n}.{i}] {act}{code_info} → reward={rew:.3f} ({elapsed:.1f}s)")

        avg = sum(rewards) / len(rewards) if rewards else 0
        print(f"  step {step_n} avg_reward={avg:.3f}")
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

    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    print(f"Loading {model_name} in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto",
    )
    print(f"Model loaded. GPU mem: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    config = GRPOConfig(
        output_dir="./zero960_grpo_output",
        num_train_epochs=1,
        max_steps=num_train_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=num_generations,
        learning_rate=5e-6,
        logging_steps=1,
        num_generations=num_generations,
        max_completion_length=4,
        bf16=True,
        gradient_checkpointing=False,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[env_reward_func],
        train_dataset=dataset,
        args=config,
        peft_config=lora_config,
    )

    print(f"Starting GRPO training: {model_name}")
    print(f"  steps={num_train_steps}, generations={num_generations}")
    print(f"  env={base_url}")

    trainer.train()

    trainer.save_model("./zero960_grpo_final")
    print("Training complete. Model saved to ./zero960_grpo_final")
    print(f"Rollout logs: {log_path}")

    # Print summary stats from logs
    action_counts: dict[str, int] = {}
    reward_by_action: dict[str, list[float]] = {}
    with open(log_path) as f:
        for line in f:
            entry = json.loads(line)
            act = entry.get("parsed_action", "unknown")
            action_counts[act] = action_counts.get(act, 0) + 1
            reward_by_action.setdefault(act, []).append(entry.get("reward", 0))
    print("\n=== Training Summary ===")
    for act, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        avg_r = sum(reward_by_action[act]) / len(reward_by_action[act])
        print(f"  {act}: {count}x, avg_reward={avg_r:.3f}")

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
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional tokenizer path/name for infer mode when loading a checkpoint without tokenizer files.",
    )
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
            tokenizer_name=args.tokenizer,
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
