"""Create deterministic, label-stratified content-group splits.

The dataset has no patient identifiers. A content hash is therefore the
strongest leakage boundary currently available, but the resulting evaluation
must not be described as patient-separated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def manifest_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_rows(rows: list[dict[str, str]], seed: int = 20260922, validation_fraction: float = 0.15, test_fraction: float = 0.15) -> list[dict[str, str]]:
    eligible = [row for row in rows if row.get("label_status") == "agreement" and row.get("consensus_label", "") != ""]
    if len({row["image_hash"] for row in eligible}) != len(eligible):
        raise ValueError("Manifest contains duplicate image hashes among eligible rows.")
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in eligible:
        by_label[row["consensus_label"]].append(row)
    result: list[dict[str, str]] = []
    for label, label_rows in sorted(by_label.items()):
        rng.shuffle(label_rows)
        n_test = max(1, round(len(label_rows) * test_fraction))
        n_validation = max(1, round(len(label_rows) * validation_fraction))
        for index, row in enumerate(label_rows):
            split = "test" if index < n_test else "validation" if index < n_test + n_validation else "train"
            result.append({**row, "split": split})
    rng.shuffle(result)
    return result


def write_split(manifest: Path, output: Path, seed: int) -> dict[str, object]:
    with manifest.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    split = split_rows(rows, seed=seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(split[0]) if split else ["image_hash", "consensus_label", "split"]
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(split)
    summary = {
        "source_manifest": manifest.as_posix(),
        "source_manifest_sha256": manifest_hash(manifest),
        "seed": seed,
        "patient_separation": "not_available_no_patient_identifiers",
        "rows": len(split),
        "split_counts": dict(Counter(row["split"] for row in split)),
        "label_counts": dict(Counter(row["consensus_label"] for row in split)),
        "split_label_counts": {name: dict(Counter(row["consensus_label"] for row in split if row["split"] == name)) for name in ("train", "validation", "test")},
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/digital-knee-xray-v1.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/manifests/digital-knee-xray-v1.splits.csv"))
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    print(json.dumps(write_split(args.manifest, args.output, args.seed), indent=2))


if __name__ == "__main__":
    main()

