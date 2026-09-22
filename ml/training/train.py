"""Train the first research KL baseline.

This command is intentionally explicit: it does not silently download data,
weights, or publish a model. Use --pretrained only when the weight licence and
download provenance have been recorded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .metrics import classification_metrics, expected_calibration_error
from .model import build_model


class KneeDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], root: Path, transform: transforms.Compose) -> None:
        self.rows = rows
        self.root = root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(self.root.parent / row["canonical_path"]).convert("RGB")
        return self.transform(image), int(row["consensus_label"])


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train(training); total_loss = 0.0; targets: list[int] = []; predictions: list[int] = []; probabilities: list[list[float]] = []
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images); loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            total_loss += loss.item() * len(labels); targets.extend(labels.cpu().tolist()); predictions.extend(probs.argmax(1).tolist()); probabilities.extend(probs.tolist())
    metrics = classification_metrics(targets, predictions)
    metrics.update({"loss": total_loss / len(loader.dataset), "ece": expected_calibration_error(targets, probabilities), "confusion_matrix": confusion_matrix(targets, predictions, labels=list(range(5))).tolist()})
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=Path, default=Path("data/manifests/digital-knee-xray-v1.splits.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/models/ortholens-kl-resnet18-v0.1.0.pt"))
    parser.add_argument("--epochs", type=int, default=10); parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args(); seed_everything(args.seed)
    rows = read_rows(args.splits); train_rows = [r for r in rows if r["split"] == "train"]; val_rows = [r for r in rows if r["split"] == "validation"]; test_rows = [r for r in rows if r["split"] == "test"]
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_loader = DataLoader(KneeDataset(train_rows, args.data_root, transform), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(KneeDataset(val_rows, args.data_root, transform), batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(KneeDataset(test_rows, args.data_root, transform), batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model = build_model(pretrained=args.pretrained).to(device); criterion = nn.CrossEntropyLoss(); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best = None; history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, True); val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, False); history.append({"epoch": epoch, "train": train_metrics, "validation": val_metrics})
        print(json.dumps({"epoch": epoch, "train": train_metrics, "validation": val_metrics}))
        if best is None or val_metrics["quadratic_weighted_kappa"] > best["score"]:
            best = {"score": val_metrics["quadratic_weighted_kappa"], "state_dict": model.state_dict(), "epoch": epoch, "validation": val_metrics}
    if best is None: raise RuntimeError("No model checkpoint was produced")
    model.load_state_dict(best["state_dict"]); test_metrics = run_epoch(model, test_loader, criterion, optimizer, device, False)
    args.output.parent.mkdir(parents=True, exist_ok=True); manifest_digest = hashlib.sha256(args.splits.read_bytes()).hexdigest()
    torch.save({"model_state_dict": model.cpu().state_dict(), "model_version": "ortholens-kl-resnet18-v0.1.0", "classes": list(range(5)), "manifest_sha256": manifest_digest, "seed": args.seed, "device": str(device), "pretrained": args.pretrained, "best_epoch": best["epoch"], "validation_metrics": best["validation"], "test_metrics": test_metrics, "history": history, "research_only": True}, args.output)
    print(json.dumps({"model": args.output.as_posix(), "test_metrics": test_metrics}, indent=2))


if __name__ == "__main__":
    main()

