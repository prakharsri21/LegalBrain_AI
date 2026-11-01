import json
import random
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

SOURCE = Path("data/processed/stage4-export-v1")
OUTPUT = Path("data/training_pilot_v1")

SEED = 42
TARGETS = {"train": 500, "validation": 100}

if OUTPUT.exists() and any(OUTPUT.iterdir()):
    raise SystemExit(f"Output directory is not empty: {OUTPUT}")

if not SOURCE.joinpath("manifest.jsonl").exists():
    raise SystemExit("Stage 4 manifest not found.")

OUTPUT.mkdir(parents=True, exist_ok=True)

rng = random.Random(SEED)
reservoirs = {s: [] for s in TARGETS}
group_counts = Counter()
manifest_path = SOURCE / "manifest.jsonl"

# Pass 1: Reservoir-sample distinct question groups.
print("Pass 1: Selecting distinct question groups...")

with tempfile.TemporaryDirectory() as tmp:
    db = sqlite3.connect(Path(tmp) / "groups.sqlite")
    db.execute("""
        CREATE TABLE seen (
            split TEXT,
            group_id TEXT,
            PRIMARY KEY (split, group_id)
        ) WITHOUT ROWID
    """)

    with manifest_path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            row = json.loads(line)
            split = row["split"]

            if split not in TARGETS:
                continue

            gid = row["group_id"]
            cursor = db.execute(
                "INSERT OR IGNORE INTO seen VALUES (?, ?)",
                (split, gid)
            )

            if cursor.rowcount == 0:
                continue

            group_counts[split] += 1
            n = group_counts[split]
            k = TARGETS[split]

            if n <= k:
                reservoirs[split].append(gid)
            else:
                j = rng.randrange(n)
                if j < k:
                    reservoirs[split][j] = gid

            if line_number % 500000 == 0:
                print(f"Scanned {line_number:,} manifest rows")

    db.close()

for split, target in TARGETS.items():
    if group_counts[split] < target:
        raise RuntimeError(
            f"Only {group_counts[split]} groups available for {split}"
        )

selected = {s: set(reservoirs[s]) for s in TARGETS}

# Pass 2: Stream JSONL files in manifest order.
print("Pass 2: Writing selected records...")

source_files = {
    "train": SOURCE / "train.jsonl",
    "validation": SOURCE / "validation.jsonl",
}

output_files = {
    "train": OUTPUT / "train.jsonl",
    "validation": OUTPUT / "valid.jsonl",
}

record_counts = Counter()
selected_group_records = {s: Counter() for s in TARGETS}
token_lengths = {s: [] for s in TARGETS}
consumed = Counter()

with manifest_path.open(encoding="utf-8") as manifest, \
     source_files["train"].open(encoding="utf-8") as train_in, \
     source_files["validation"].open(encoding="utf-8") as valid_in, \
     output_files["train"].open("w", encoding="utf-8") as train_out, \
     output_files["validation"].open("w", encoding="utf-8") as valid_out, \
     (OUTPUT / "pilot_manifest.jsonl").open("w", encoding="utf-8") as pilot_manifest:

    inputs = {
        "train": train_in,
        "validation": valid_in,
    }

    outputs = {
        "train": train_out,
        "validation": valid_out,
    }

    for line_number, line in enumerate(manifest, 1):
        meta = json.loads(line)
        split = meta["split"]

        # Never open or read test.jsonl.
        if split not in TARGETS:
            continue

        data_line = inputs[split].readline()
        if not data_line:
            raise RuntimeError(f"Unexpected EOF in {split}.jsonl")

        record = json.loads(data_line)
        if "messages" not in record:
            raise ValueError("Unexpected JSONL schema")

        consumed[split] += 1
        gid = meta["group_id"]

        if gid not in selected[split]:
            continue

        outputs[split].write(data_line)
        pilot_manifest.write(line)

        record_counts[split] += 1
        selected_group_records[split][gid] += 1
        token_lengths[split].append(meta["token_length"])

    for split, f in inputs.items():
        if f.readline():
            raise RuntimeError(
                f"Source {split}.jsonl has extra records beyond manifest"
            )

def length_stats(values):
    values = sorted(values)

    def percentile(p):
        index = round((len(values) - 1) * p)
        return values[index]

    return {
        "median": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "maximum": max(values),
    }

summary = {
    "seed": SEED,
    "source": str(SOURCE),
    "sampling_method": "Reservoir sampling over distinct question groups",
    "selected_question_groups": {
        split: len(selected[split]) for split in TARGETS
    },
    "available_question_groups": dict(group_counts),
    "records": dict(record_counts),
    "multiple_response_groups": {
        split: sum(n > 1 for n in selected_group_records[split].values())
        for split in TARGETS
    },
    "token_length_statistics": {
        split: length_stats(token_lengths[split])
        for split in TARGETS
    },
    "outputs": {
        "train": "train.jsonl",
        "validation": "valid.jsonl",
        "manifest": "pilot_manifest.jsonl",
    },
}

(OUTPUT / "pilot_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n",
    encoding="utf-8"
)

print("\nPILOT COMPLETE")
print(json.dumps(summary, indent=2))
