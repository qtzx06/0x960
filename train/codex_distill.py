"""Collect teacher trajectories from Codex for 0x960 and emit SFT-ready samples.

This script keeps the teacher inside the same bounded action space as the student:
the model sees the current observation and returns exactly one JSON action per turn.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from zero960_env.client import Zero960Client
from zero960_env.models import Zero960Action, Zero960Observation

from train.minimal_trl_openenv import SYSTEM_PROMPT, format_observation_as_prompt

ACTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "action_type": {
            "type": "string",
            "enum": ["read_file", "write_file", "run_static_eval", "run_match", "finish"],
        },
        "path": {"type": ["string", "null"]},
        "content": {"type": ["string", "null"]},
    },
    "required": ["action_type", "path", "content"],
}

TEACHER_INSTRUCTIONS = """You are the teacher policy for 0x960.

Return exactly one JSON action object that matches the provided schema.

Constraints:
- Act only through the bounded action schema. Do not describe actions.
- Do not use shell commands or external tools.
- The current eval.py contents are already included in the observation.
- Prefer the high-reward loop: write_file -> run_match -> finish.
- Avoid repeated run_static_eval unless it is truly necessary.
- Always include all three JSON keys: action_type, path, content.
- Use null for unused fields. Example: {"action_type":"run_match","path":null,"content":null}
- If you choose write_file, return a full valid replacement for eval.py.
"""


@dataclass(slots=True)
class TeacherTurn:
    action: Zero960Action
    raw_response: str
    elapsed_s: float


def _action_payload(action: Zero960Action) -> dict:
    return {
        "action_type": action.action_type,
        "path": action.path,
        "content": action.content,
    }


def _find_codex_binary(explicit_path: str | None) -> str:
    if explicit_path:
        return explicit_path
    codex_bin = shutil.which("codex")
    if codex_bin is None:
        raise RuntimeError("codex CLI not found on PATH; install or pass --codex-bin")
    return codex_bin


def _teacher_prompt(observation: Zero960Observation) -> str:
    return (
        f"{TEACHER_INSTRUCTIONS}\n"
        "Use the same environment contract as the student prompt below.\n\n"
        f"System prompt:\n{SYSTEM_PROMPT}\n\n"
        f"Observation:\n{format_observation_as_prompt(observation)}\n"
    )


def _run_codex_turn(
    codex_bin: str,
    model: str,
    workdir: Path,
    observation: Zero960Observation,
    timeout_s: int,
) -> TeacherTurn:
    prompt = _teacher_prompt(observation)

    with tempfile.TemporaryDirectory(prefix="zero960_codex_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        schema_path = temp_dir / "action.schema.json"
        output_path = temp_dir / "action.json"
        schema_path.write_text(json.dumps(ACTION_SCHEMA))

        command = [
            codex_bin,
            "exec",
            "--model",
            model,
            "--cd",
            str(workdir),
            "--ephemeral",
            "--color",
            "never",
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(output_path),
            "-",
        ]

        started = time.time()
        result = subprocess.run(
            command,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        elapsed_s = round(time.time() - started, 2)

        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "refresh_token_reused" in stderr:
                raise RuntimeError(
                    "codex auth is stale; run `codex logout` then `codex login` and retry"
                )
            if "usage limit" in stderr.lower():
                raise RuntimeError("codex usage limit reached; stop the batch and retry later")
            raise RuntimeError(f"codex exec failed with exit code {result.returncode}: {stderr}")
        if not output_path.exists():
            raise RuntimeError("codex exec did not write an output message")

        raw_response = output_path.read_text().strip()
        if not raw_response:
            raise RuntimeError("codex exec returned an empty final message")
        action = Zero960Action.model_validate_json(raw_response)
        return TeacherTurn(action=action, raw_response=raw_response, elapsed_s=elapsed_s)


def _append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def _sft_sample(observation: Zero960Observation, action: Zero960Action, metadata: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_observation_as_prompt(observation)},
            {"role": "assistant", "content": json.dumps(_action_payload(action))},
        ],
        "metadata": metadata,
    }


def collect_teacher_rollouts(
    base_url: str,
    model: str,
    episodes: int,
    max_turns: int,
    timeout_s: int,
    output_dir: Path,
    min_reward: float,
    codex_bin: str | None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / f"teacher_rollouts_{int(time.time())}.jsonl"
    sft_path = output_dir / f"sft_samples_{int(time.time())}.jsonl"
    trace_path.touch()
    sft_path.touch()

    codex_executable = _find_codex_binary(codex_bin)
    workdir = Path(__file__).resolve().parents[1]

    with Zero960Client(base_url=base_url) as client:
        stop_reason: str | None = None
        for episode_index in range(episodes):
            reset_result = client.reset()
            observation = reset_result.observation
            episode_turns: list[dict] = []
            forced_finish = False

            for turn_index in range(max_turns):
                if reset_result.done:
                    break

                pre_action_observation = observation
                try:
                    teacher_turn = _run_codex_turn(
                        codex_bin=codex_executable,
                        model=model,
                        workdir=workdir,
                        observation=pre_action_observation,
                        timeout_s=timeout_s,
                    )
                except RuntimeError as exc:
                    if "usage limit reached" in str(exc):
                        stop_reason = str(exc)
                        break
                    raise

                step_result = client.step(teacher_turn.action)
                observation = step_result.observation

                turn_payload = {
                    "episode_index": episode_index,
                    "turn_index": turn_index,
                    "teacher_model": model,
                    "elapsed_s": teacher_turn.elapsed_s,
                    "raw_response": teacher_turn.raw_response,
                    "action": _action_payload(teacher_turn.action),
                    "observation_before": pre_action_observation.model_dump(),
                    "observation_after": observation.model_dump(),
                    "reward": step_result.reward,
                    "done": step_result.done,
                }
                episode_turns.append(turn_payload)

                if step_result.done:
                    reset_result = step_result
                    break
                reset_result = step_result

            if stop_reason is not None:
                break

            if not reset_result.done:
                forced_finish = True
                finish_result = client.step(Zero960Action(action_type="finish"))
                observation = finish_result.observation
                episode_turns.append(
                    {
                        "episode_index": episode_index,
                        "turn_index": len(episode_turns),
                        "teacher_model": model,
                        "elapsed_s": 0.0,
                        "raw_response": json.dumps({"action_type": "finish"}),
                        "action": {"action_type": "finish"},
                        "observation_before": reset_result.observation.model_dump(),
                        "observation_after": observation.model_dump(),
                        "reward": finish_result.reward,
                        "done": finish_result.done,
                        "forced_finish": True,
                    }
                )
                reset_result = finish_result

            final_reward = float(reset_result.reward or 0.0)
            accepted = (
                final_reward >= min_reward
                and observation.has_valid_edit
                and observation.has_run_match
            )

            episode_payload = {
                "episode_index": episode_index,
                "teacher_model": model,
                "forced_finish": forced_finish,
                "accepted_for_sft": accepted,
                "final_reward": final_reward,
                "final_status": observation.status_message,
                "turns": episode_turns,
            }
            _append_jsonl(trace_path, episode_payload)

            if accepted:
                for turn in episode_turns:
                    if turn.get("forced_finish"):
                        continue
                    sample = _sft_sample(
                        observation=Zero960Observation.model_validate(turn["observation_before"]),
                        action=Zero960Action.model_validate(turn["action"]),
                        metadata={
                            "episode_index": episode_index,
                            "turn_index": turn["turn_index"],
                            "teacher_model": model,
                            "final_reward": final_reward,
                        },
                    )
                    _append_jsonl(sft_path, sample)

            print(
                {
                    "episode": episode_index,
                    "final_reward": final_reward,
                    "accepted_for_sft": accepted,
                    "turns": len(episode_turns),
                    "final_status": observation.status_message,
                }
            )

        if stop_reason is not None:
            print({"stopped_early": True, "reason": stop_reason})

    return trace_path, sft_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Codex teacher rollouts for 0x960.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--min-reward", type=float, default=0.4)
    parser.add_argument("--codex-bin", default=None)
    parser.add_argument("--output-dir", default="outputs/codex_distill")
    args = parser.parse_args()

    trace_path, sft_path = collect_teacher_rollouts(
        base_url=args.base_url,
        model=args.model,
        episodes=args.episodes,
        max_turns=args.max_turns,
        timeout_s=args.timeout_s,
        output_dir=Path(args.output_dir),
        min_reward=args.min_reward,
        codex_bin=args.codex_bin,
    )
    print({"trace_path": str(trace_path), "sft_path": str(sft_path)})


if __name__ == "__main__":
    main()
