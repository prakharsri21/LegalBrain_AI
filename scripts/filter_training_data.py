from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen3-4B"
DEFAULT_INPUT = Path("data/processed")
DEFAULT_OUTPUT = Path("data/processed_safe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter examples that may be truncated during MLX-LM training."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=896,
        help="Keep at least 128 tokens of sequence capacity for the answer.",
    )

    args = parser.parse_args()

    if args.max_seq_length < 1:
        parser.error("--max-seq-length must be positive")
    if not 1 <= args.max_prompt_tokens < args.max_seq_length:
        parser.error(
            "--max-prompt-tokens must be positive and smaller than "
            "--max-seq-length"
        )

    return args


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"Required file not found: {path}")

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

            if (
                not isinstance(messages, list)
                or len(messages) < 2
                or messages[-1].get("role") != "assistant"
                or not messages[-1].get("content", "").strip()
            ):
                raise SystemExit(
                    f"Invalid chat example in {path}, line {line_number}"
                )

            records.append(record)

    return records


def token_lengths(tokenizer, record: dict) -> tuple[int, int, int]:
    messages = record["messages"]

    # Tokenize the prompt with the assistant generation marker.
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )

    # Tokenize the entire conversation using the model's chat template.
    full_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
    )

    if not isinstance(prompt_ids, list) or not isinstance(full_ids, list):
        raise TypeError("Expected token ID lists from the tokenizer.")

    prompt_length = len(prompt_ids)
    full_length = len(full_ids)
    estimated_answer_length = full_length - prompt_length

    return prompt_length, full_length, estimated_answer_length


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def filter_split(
    tokenizer,
    input_path: Path,
    output_path: Path,
    max_seq_length: int,
    max_prompt_tokens: int,
) -> dict:
    records = load_jsonl(input_path)
    kept = []

    counts = {
        "input": len(records),
        "kept": 0,
        "full_sequence_too_long": 0,
        "prompt_too_long": 0,
        "invalid_token_lengths": 0,
    }

    for index, record in enumerate(records, start=1):
        prompt_len, full_len, answer_len = token_lengths(tokenizer, record)

        if prompt_len <= 0 or full_len <= prompt_len or answer_len <= 0:
            counts["invalid_token_lengths"] += 1
            print(
                f"Warning: skipped {input_path.name} record {index} "
                f"with suspicious lengths: "
                f"prompt={prompt_len}, full={full_len}, answer≈{answer_len}"
            )
            continue

        if full_len > max_seq_length:
            counts["full_sequence_too_long"] += 1
            continue

        if prompt_len > max_prompt_tokens:
            counts["prompt_too_long"] += 1
            continue

        kept.append(record)

    counts["kept"] = len(kept)

    if not kept:
        raise SystemExit(
            f"Filtering removed all records from {input_path}. "
            "Review the tokenization and configured limits."
        )

    write_jsonl(output_path, kept)

    print(f"\n{input_path.name}")
    print(f"  Input records: {counts['input']}")
    print(f"  Retained: {counts['kept']}")
    print(f"  Full sequence too long: {counts['full_sequence_too_long']}")
    print(f"  Prompt too long: {counts['prompt_too_long']}")
    print(f"  Invalid token lengths: {counts['invalid_token_lengths']}")
    print(f"  Output: {output_path}")

    return counts


def main() -> None:
    args = parse_args()

    print(f"Tokenizer: {args.model}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Maximum sequence length: {args.max_seq_length}")
    print(f"Maximum prompt length: {args.max_prompt_tokens}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    except Exception as exc:
        raise SystemExit(
            f"Failed to load tokenizer: {type(exc).__name__}: {exc}"
        ) from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "model": args.model,
        "max_seq_length": args.max_seq_length,
        "max_prompt_tokens": args.max_prompt_tokens,
        "splits": {},
    }

    for split in ("train", "valid"):
        summary["splits"][split] = filter_split(
            tokenizer=tokenizer,
            input_path=args.input_dir / f"{split}.jsonl",
            output_path=args.output_dir / f"{split}.jsonl",
            max_seq_length=args.max_seq_length,
            max_prompt_tokens=args.max_prompt_tokens,
        )

    # Keep the held-out test set identical to the original split.
    test_records = load_jsonl(args.input_dir / "test.jsonl")
    test_output = args.output_dir / "test.jsonl"
    write_jsonl(test_output, test_records)

    summary["test"] = {
        "copied_unchanged": True,
        "records": len(test_records),
        "path": str(test_output),
    }

    summary_path = args.output_dir / "filter_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(f"\nTest split copied unchanged: {len(test_records)} records")
    print(f"Summary saved to: {summary_path}")
    print("\nFiltering complete.")


if __name__ == "__main__":
    main()