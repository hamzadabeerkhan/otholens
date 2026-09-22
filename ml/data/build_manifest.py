"""Build a leakage-aware manifest for the Digital Knee X-ray dataset.

The downloaded images stay under data/ and are ignored by Git. This script
only writes a CSV manifest and a JSON summary under data/manifests/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
from collections import Counter, defaultdict
from pathlib import Path


LABELS = {"0Normal": 0, "1Doubtful": 1, "2Mild": 2, "3Moderate": 3, "4Severe": 4}
EXPERTS = ("MedicalExpert-I", "MedicalExpert-II")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG file: {path}")
    return struct.unpack(">II", header[16:24])


def collect(root: Path) -> dict[str, dict[str, list[dict[str, object]]]]:
    by_hash: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for expert in EXPERTS:
        expert_root = root / expert
        if not expert_root.is_dir():
            raise FileNotFoundError(f"Expected dataset directory: {expert_root}")
        for path in sorted(expert_root.rglob("*.png")):
            label_name = path.parent.name
            if label_name not in LABELS:
                raise ValueError(f"Unknown label directory: {path.parent}")
            width, height = png_dimensions(path)
            by_hash[sha256_file(path)][expert].append(
                {
                    "path": path.relative_to(root.parent).as_posix(),
                    "filename": path.name,
                    "label_name": label_name,
                    "label": LABELS[label_name],
                    "width": width,
                    "height": height,
                }
            )
    return by_hash


def build(root: Path, output: Path) -> dict[str, object]:
    grouped = collect(root)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    status_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    expert_counts: dict[str, Counter[str]] = {expert: Counter() for expert in EXPERTS}
    dimensions: Counter[str] = Counter()
    duplicate_groups: dict[str, int] = {}

    for image_hash in sorted(grouped):
        experts = grouped[image_hash]
        first = next(iter(experts.values()))[0]
        expert_values: dict[str, dict[str, object] | None] = {}
        for expert in EXPERTS:
            values = experts.get(expert, [])
            expert_values[expert] = values[0] if values else None
            if values:
                duplicate_groups[f"{expert}:{image_hash}"] = len(values)
                expert_counts[expert][str(values[0]["label"])] += 1
        labels = [item["label"] for item in expert_values.values() if item is not None]
        status = "missing_expert" if len(labels) != 2 else "agreement" if labels[0] == labels[1] else "disagreement"
        consensus = labels[0] if status == "agreement" else ""
        status_counts[status] += 1
        if consensus != "":
            label_counts[str(consensus)] += 1
        dimensions[f"{first['width']}x{first['height']}"] += 1
        rows.append(
            {
                "image_hash": image_hash,
                "canonical_path": first["path"],
                "expert_i_label": expert_values[EXPERTS[0]]["label"] if expert_values[EXPERTS[0]] else "",
                "expert_ii_label": expert_values[EXPERTS[1]]["label"] if expert_values[EXPERTS[1]] else "",
                "consensus_label": consensus,
                "label_status": status,
                "width": first["width"],
                "height": first["height"],
                "expert_i_duplicate_count": len(experts.get(EXPERTS[0], [])),
                "expert_ii_duplicate_count": len(experts.get(EXPERTS[1], [])),
            }
        )

    fields = list(rows[0]) if rows else []
    with (output / "digital-knee-xray-v1.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset_root": root.as_posix(),
        "experts": list(EXPERTS),
        "source_png_files": {expert: sum(len(group.get(expert, [])) for group in grouped.values()) for expert in EXPERTS},
        "unique_image_hashes": len(grouped),
        "label_status_counts": dict(status_counts),
        "consensus_label_counts": dict(label_counts),
        "expert_label_counts": {expert: dict(counts) for expert, counts in expert_counts.items()},
        "dimensions": dict(dimensions),
        "duplicate_hash_groups": sum(count > 1 for count in duplicate_groups.values()),
        "duplicate_files_beyond_first": sum(count - 1 for count in duplicate_groups.values() if count > 1),
        "patient_separation": "not_available_no_patient_identifiers_found",
        "training_warning": "Do not split by file path. Group by image_hash and exclude disagreement rows from the first supervised baseline.",
    }
    (output / "digital-knee-xray-v1.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/manifests"))
    args = parser.parse_args()
    summary = build(args.root, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

