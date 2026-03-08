"""Supervised fine-tuning for a bounded-action 0x960 student policy."""

from __future__ import annotations

import argparse
import glob
import json
import random
from collections import Counter
from pathlib import Path

from datasets import Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


ALLOWED_ACTION_KEYS = {"action_type", "path", "content"}


def _resolve_input_paths(explicit_paths: list[str], data_glob: str) -> list[Path]:
    paths = [Path(path) for path in explicit_paths]
    paths.extend(Path(path) for path in glob.glob(data_glob))
    unique_paths = sorted({path.resolve() for path in paths if path.exists()})
    if not unique_paths:
        raise FileNotFoundError(
            "no SFT data files found; pass --data-path or adjust --data-glob"
        )
    return unique_paths


def _validate_record(payload: dict, source_path: Path, line_number: int) -> dict | None:
    messages = payload.get("messages")
    metadata = payload.get("metadata", {})
    if not isinstance(messages, list) or len(messages) != 3:
        return None

    roles = [message.get("role") for message in messages if isinstance(message, dict)]
    if roles != ["system", "user", "assistant"]:
        return None

    assistant_content = messages[-1].get("content")
    if not isinstance(assistant_content, str):
        return None

    try:
        action_payload = json.loads(assistant_content)
    except json.JSONDecodeError:
        return None

    if not isinstance(action_payload, dict):
        return None
    if set(action_payload) != ALLOWED_ACTION_KEYS:
        return None
    if action_payload["action_type"] not in {
        "read_file",
        "write_file",
        "run_static_eval",
        "run_match",
        "finish",
    }:
        return None

    final_reward = metadata.get("final_reward")
    if final_reward is not None:
        final_reward = float(final_reward)

    return {
        "messages": messages,
        "metadata": {
            "source_path": str(source_path),
            "line_number": line_number,
            "episode_index": metadata.get("episode_index"),
            "turn_index": metadata.get("turn_index"),
            "teacher_model": metadata.get("teacher_model"),
            "final_reward": final_reward,
        },
        "action_type": action_payload["action_type"],
    }


def load_sft_records(
    input_paths: list[Path],
    min_final_reward: float,
    max_examples: int | None,
    seed: int,
) -> tuple[list[dict], dict]:
    records: list[dict] = []
    skipped_invalid = 0
    skipped_low_reward = 0
    dedupe_keys: set[str] = set()

    for input_path in input_paths:
        for line_number, line in enumerate(input_path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            record = _validate_record(payload, input_path, line_number)
            if record is None:
                skipped_invalid += 1
                continue
            final_reward = record["metadata"]["final_reward"]
            if final_reward is not None and final_reward < min_final_reward:
                skipped_low_reward += 1
                continue

            dedupe_key = json.dumps(record["messages"], sort_keys=True)
            if dedupe_key in dedupe_keys:
                continue
            dedupe_keys.add(dedupe_key)
            records.append(record)

    random.Random(seed).shuffle(records)
    if max_examples is not None:
        records = records[:max_examples]

    stats = {
        "input_files": [str(path) for path in input_paths],
        "records_kept": len(records),
        "skipped_invalid": skipped_invalid,
        "skipped_low_reward": skipped_low_reward,
        "action_counts": dict(Counter(record["action_type"] for record in records)),
    }
    return records, stats


def split_records(records: list[dict], eval_fraction: float) -> tuple[list[dict], list[dict]]:
    if not records:
        return [], []
    if eval_fraction <= 0 or len(records) < 10:
        return records, []
    eval_size = max(1, int(len(records) * eval_fraction))
    if eval_size >= len(records):
        eval_size = len(records) - 1
    return records[eval_size:], records[:eval_size]


def build_dataset(records: list[dict]) -> Dataset:
    return Dataset.from_list(
        [
            {
                "messages": record["messages"],
                "metadata": record["metadata"],
            }
            for record in records
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT a bounded-action 0x960 student model.")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--data-path", action="append", default=[])
    parser.add_argument("--data-glob", default="outputs/codex_distill/sft_samples_*.jsonl")
    parser.add_argument("--output-dir", default="outputs/sft_student")
    parser.add_argument("--min-final-reward", type=float, default=0.4)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--assistant-only-loss", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = _resolve_input_paths(args.data_path, args.data_glob)
    records, stats = load_sft_records(
        input_paths=input_paths,
        min_final_reward=args.min_final_reward,
        max_examples=args.max_examples,
        seed=args.seed,
    )
    if not records:
        raise RuntimeError("no usable SFT rows found after validation and filtering")

    train_records, eval_records = split_records(records, args.eval_fraction)
    stats["train_records"] = len(train_records)
    stats["eval_records"] = len(eval_records)
    print(stats)

    if args.dry_run:
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    model_kwargs = {"torch_dtype": torch.bfloat16} if use_bf16 else {}
    if tokenizer.padding_side != "right":
        tokenizer.padding_side = "right"

    train_dataset = build_dataset(train_records)
    eval_dataset = build_dataset(eval_records) if eval_records else None

    trainer = SFTTrainer(
        model=AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs),
        args=SFTConfig(
            output_dir=str(output_dir),
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps,
            logging_steps=args.logging_steps,
            save_strategy="epoch",
            eval_strategy="epoch" if eval_dataset is not None else "no",
            save_total_limit=args.save_total_limit,
            report_to="none",
            bf16=use_bf16,
            gradient_checkpointing=use_cuda,
            assistant_only_loss=args.assistant_only_loss,
            max_length=args.max_length,
            remove_unused_columns=False,
            dataset_num_proc=1,
            seed=args.seed,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))


if __name__ == "__main__":
    main()
