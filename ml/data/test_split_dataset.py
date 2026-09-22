import csv
from pathlib import Path

from ml.data.split_dataset import split_rows


def rows_for_test() -> list[dict[str, str]]:
    return [
        {"image_hash": f"hash-{i}", "consensus_label": str(i % 5), "label_status": "agreement"}
        for i in range(50)
    ] + [
        {"image_hash": "ambiguous", "consensus_label": "", "label_status": "disagreement"}
    ]


def test_split_excludes_disagreements_and_is_deterministic() -> None:
    first = split_rows(rows_for_test(), seed=7)
    second = split_rows(rows_for_test(), seed=7)
    assert first == second
    assert all(row["label_status"] == "agreement" for row in first)
    assert {row["split"] for row in first} == {"train", "validation", "test"}


def test_hashes_do_not_cross_splits() -> None:
    rows = split_rows(rows_for_test(), seed=11)
    by_split = {split: {row["image_hash"] for row in rows if row["split"] == split} for split in ("train", "validation", "test")}
    assert not (by_split["train"] & by_split["validation"])
    assert not (by_split["train"] & by_split["test"])
    assert not (by_split["validation"] & by_split["test"])
