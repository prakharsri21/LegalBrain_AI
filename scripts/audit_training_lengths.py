from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen3-4B"
DEFAULT_DATA_DIR = Path("data/processed")
DEFAULT_MAX_LENGTH = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit token lengths before MLX-LM fine-tuning."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
    )

    args = parser.parse_args()

    if args.max_seq_length < 1:
        parser.error("--max-seq-length must be greater than zero")

    return args


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"Dataset file not found: {path}")

    records = []

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"Invalid JSON in {path}, line {line_number}: {exc}"
                ) from exc

            messages = record.get("messages")

            if not isinstance(messages, list) or len(messages) < 2:
                raise SystemExit(
                    f"Invalid messages in {path}, line {line_number}"
                )

            if any(
                not isinstance(message, dict)
                or not isinstance(message.get("role"), str)
                or not isinstance(message.get("content"), str)
                for message in messages
            ):
                raise SystemExit(
                    f"Invalid message structure in {path}, line {line_number}"
                )

            records.append(record)

    return records


def tokenize_messages(tokenizer, messages: list[dict]) -> list[int]:
    """
    Apply the model's actual chat template and return token IDs.

    return_dict=False is important: otherwise len() may measure the
    number of dictionary keys instead of the number of tokens.
    """
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
    )

    if not isinstance(token_ids, list):
        raise TypeError(
            "Expected a list of token IDs from apply_chat_template; "
            f"received {type(token_ids).__name__}."
        )

    if token_ids and not isinstance(token_ids[0], int):
        raise TypeError("Tokenizer output does not contain integer token IDs.")

    return token_ids


def audit_split(tokenizer, path: Path, max_length: int) -> None:
    records = read_jsonl(path)

    if not records:
        print(f"\n{path.name}: no records")
        return

    measurements = []

    for index, record in enumerate(records, start=1):
        messages = record["messages"]

        assistant_indices = [
            i
            for i, message in enumerate(messages)
            if message["role"] == "assistant"
        ]

        if not assistant_indices:
            raise SystemExit(
                f"No assistant answer in {path.name}, record {index}"
            )

        final_index = assistant_indices[-1]
        prompt_messages = messages[:final_index]
        assistant_message = messages[final_index]

        if not assistant_message["content"].strip():
            raise SystemExit(
                f"Empty assistant answer in {path.name}, record {index}"
            )

        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )

        full_ids = tokenize_messages(tokenizer, messages)

        if not isinstance(prompt_ids, list) or not isinstance(full_ids, list):
            raise TypeError("Expected token ID lists from tokenizer.")

        prompt_length = len(prompt_ids)
        full_length = len(full_ids)

        # A length of two or less indicates that something is still wrong
        # with the tokenizer/template configuration for these examples.
        if full_length <= 2 or prompt_length <= 2:
            raise RuntimeError(
                f"Suspiciously short tokenization in {path.name}, "
                f"record {index}: prompt={prompt_length}, full={full_length}. "
                "Check the model tokenizer and its chat template."
            )

        # Approximate the number of completion tokens. This difference is
        # diagnostic; special-token boundaries can make it inexact.
        completion_length = max(0, full_length - prompt_length)
        completion_room = max(0, max_length - prompt_length)
        retained_estimate = min(completion_length, completion_room)

        measurements.append(
            {
                "record": index,
                "prompt_tokens": prompt_length,
                "full_tokens": full_length,
                "completion_tokens_est": completion_length,
                "completion_retained_est": retained_estimate,
                "completion_lost_est": completion_length - retained_estimate,
            }
        )

    total = len(measurements)

    over_limit = [
        row for row in measurements if row["full_tokens"] > max_length
    ]
    long_prompts = [
        row for row in measurements if row["prompt_tokens"] >= max_length
    ]
    limited_room = [
        row for row in measurements
        if row["prompt_tokens"] >= max_length - 128
    ]
    zero_completion_room = [
        row for row in measurements
        if row["completion_retained_est"] == 0
    ]

    print(f"\n{'=' * 60}")
    print(f"Split: {path.name}")
    print(f"Records: {total}")
    print(f"Maximum sequence length: {max_length}")

    print(
        f"Full sequences over limit: {len(over_limit)}/{total} "
        f"({len(over_limit) / total:.1%})"
    )
    print(
        f"Prompts at or over limit: {len(long_prompts)}/{total} "
        f"({len(long_prompts) / total:.1%})"
    )
    print(
        f"Prompts with <=128 tokens of completion room: "
        f"{len(limited_room)}/{total}"
    )
    print(
        f"Estimated zero completion tokens retained: "
        f"{len(zero_completion_room)}/{total}"
    )

    print("\nLongest examples:")
    for row in sorted(
        measurements,
        key=lambda item: item["full_tokens"],
        reverse=True,
    )[:10]:
        print(
            f"row={row['record']:>3} "
            f"prompt={row['prompt_tokens']:>4} "
            f"full={row['full_tokens']:>4} "
            f"completion≈{row['completion_tokens_est']:>4} "
            f"retained≈{row['completion_retained_est']:>4} "
            f"lost≈{row['completion_lost_est']:>4}"
        )


def main() -> None:
    args = parse_args()

    print(f"Loading tokenizer only: {args.model}")
    print("The full model weights will not be loaded.")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    except Exception as exc:
        raise SystemExit(
            f"Could not load tokenizer: {type(exc).__name__}: {exc}"
        ) from exc

    for filename in ("train.jsonl", "valid.jsonl", "test.jsonl"):
        audit_split(
            tokenizer,
            args.data_dir / filename,
            args.max_seq_length,
        )

    print("\nToken-length audit complete.")


if __name__ == "__main__":
    main()