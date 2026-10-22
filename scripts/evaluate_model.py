from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

DEFAULT_MODEL = "Qwen/Qwen3-4B"
DEFAULT_DATA = Path("data/processed/test.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a model on a fixed sample of held-out legal QA examples."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model ID or local model path.",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Optional MLX-LM LoRA adapter directory.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="JSONL file containing held-out chat examples.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of examples to evaluate (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for selecting a reproducible evaluation sample.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=160,
        help="Maximum generated tokens per answer.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature; zero requests greedy decoding.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path. A path is generated if omitted.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing evaluation output.",
    )

    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be greater than zero")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be greater than zero")
    if args.temperature < 0:
        parser.error("--temperature cannot be negative")

    return args


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read and validate the JSONL records from a held-out split."""
    if not path.is_file():
        raise SystemExit(f"Evaluation data file not found: {path}")

    records: list[dict[str, Any]] = []

    try:
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(
                        f"Invalid JSON at {path}:{line_number}: {exc}"
                    ) from exc

                messages = record.get("messages")

                if not isinstance(messages, list) or len(messages) < 2:
                    raise SystemExit(
                        f"Expected chat messages at {path}:{line_number}"
                    )

                user_messages = [
                    message
                    for message in messages
                    if message.get("role") == "user"
                ]
                assistant_messages = [
                    message
                    for message in messages
                    if message.get("role") == "assistant"
                ]

                if not user_messages or not assistant_messages:
                    raise SystemExit(
                        f"Missing user or assistant message at "
                        f"{path}:{line_number}"
                    )

                input_text = user_messages[-1].get("content", "")
                reference = assistant_messages[-1].get("content", "")

                if not isinstance(input_text, str) or not input_text.strip():
                    raise SystemExit(
                        f"Empty user prompt at {path}:{line_number}"
                    )
                if not isinstance(reference, str) or not reference.strip():
                    raise SystemExit(
                        f"Empty reference answer at {path}:{line_number}"
                    )

                records.append(
                    {
                        "source_row": line_number,
                        "input_text": input_text,
                        "reference_answer": reference,
                    }
                )

    except OSError as exc:
        raise SystemExit(f"Could not read {path}: {exc}") from exc

    if not records:
        raise SystemExit(f"No usable evaluation records found in {path}")

    return records


def choose_sample(
    records: list[dict[str, Any]],
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Choose the same examples whenever the data, limit, and seed match."""
    rng = random.Random(seed)
    count = min(limit, len(records))
    return rng.sample(records, count)


def output_path_for(args: argparse.Namespace) -> Path:
    """Choose a separate output file for the base model or each adapter."""
    if args.output is not None:
        return args.output

    if args.adapter_path:
        adapter_name = Path(args.adapter_path).name
        run_name = f"{adapter_name}_seed{args.seed}"
    else:
        run_name = f"base_qwen3_4b_seed{args.seed}"

    return Path("outputs/evaluations") / f"{run_name}.jsonl"


def clean_response(text: str) -> str:
    """Remove a completed Qwen thinking section from the saved answer."""
    if "</think>" in text:
        text = text.rsplit("</think>", maxsplit=1)[-1]

    return text.strip()


def run_evaluation(args: argparse.Namespace) -> None:
    output_path = output_path_for(args)
    sampler = make_sampler(temp=args.temperature)
    if output_path.exists() and not args.overwrite:
        raise SystemExit(
            f"Output already exists: {output_path}\n"
            "Choose another --output path or pass --overwrite."
        )

    records = read_records(args.data)
    selected_records = choose_sample(
        records,
        limit=args.limit,
        seed=args.seed,
    )

    print(f"Model: {args.model}")
    print(f"Adapter: {args.adapter_path or '(base model)'}")
    print(f"Evaluation file: {args.data}")
    print(f"Available examples: {len(records)}")
    print(f"Selected examples: {len(selected_records)}")
    print(f"Selection seed: {args.seed}")
    print(f"Maximum generated tokens: {args.max_tokens}")

    print("\nLoading model...")
    load_kwargs: dict[str, Any] = {}

    if args.adapter_path:
        load_kwargs["adapter_path"] = args.adapter_path

    try:
        model, tokenizer = load(args.model, **load_kwargs)
    except Exception as exc:
        raise SystemExit(
            "Could not load the model or adapter. Check the model ID, "
            f"adapter path, and MLX-LM configuration.\n{type(exc).__name__}: {exc}"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save each completed result immediately so an interruption does not
    # lose answers generated earlier in the run.
    try:
        with output_path.open("w", encoding="utf-8") as output_file:
            for index, record in enumerate(selected_records, start=1):
                # Use the same prompt for base-model and adapter runs.
                # /no_think asks Qwen3 for a direct answer.
                user_prompt = record["input_text"].rstrip() + "\n/no_think"

                chat_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )

                prediction_raw = generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=chat_prompt,
                    max_tokens=args.max_tokens,
                    sampler=sampler,
                    verbose=False,
                )

                prediction = clean_response(prediction_raw)

                result = {
                    "source_row": record["source_row"],
                    "model": args.model,
                    "adapter_path": args.adapter_path,
                    "seed": args.seed,
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "input": record["input_text"],
                    "reference_answer": record["reference_answer"],
                    "prediction": prediction,
                }

                output_file.write(
                    json.dumps(result, ensure_ascii=False) + "\n"
                )
                output_file.flush()

                print(f"\n[{index}/{len(selected_records)}] Test row "
                      f"{record['source_row']}")
                print(f"Reference: {record['reference_answer'][:220]}")
                print(f"Prediction: {prediction[:350]}")

                # Release unused generation buffers between examples.
                gc.collect()
                mx.clear_cache()

    except KeyboardInterrupt:
        print(
            "\nEvaluation interrupted. Completed answers were saved to "
            f"{output_path}"
        )
        raise SystemExit(130)

    except Exception as exc:
        raise SystemExit(
            f"Evaluation failed. Partial results, if any, are at "
            f"{output_path}.\n{type(exc).__name__}: {exc}"
        ) from exc

    print(f"\nEvaluation complete. Results saved to: {output_path}")


def main() -> None:
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()