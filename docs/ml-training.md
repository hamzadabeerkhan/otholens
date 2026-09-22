# First KL Baseline Training

The first real model is a five-class ResNet18 transfer-learning baseline. It is a research model, not a clinical device.

## Prepare the data

From the repository root:

```bash
python ml/data/build_manifest.py
python ml/data/split_dataset.py
```

The split script keeps one row per image content hash, excludes expert disagreements, and writes a deterministic seed into the summary. Because the dataset has no patient identifiers, evaluation must be described as content-grouped rather than patient-separated.

## Train

The training image is intentionally separate from the application image. It is allowed to write only to ignored local artifacts.

```bash
docker build -t ortholens-training ./ml/training
docker run --rm \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/artifacts:/app/artifacts" \
  ortholens-training \
  --epochs 10
```

Use `--pretrained` only after recording the pretrained weight source and its licence. A CPU smoke test may use `--epochs 1 --batch-size 4`.

## Promotion rules

Do not connect a produced checkpoint to the web API until the model card, leakage checks, confusion matrix, macro F1, balanced accuracy, quadratic weighted kappa, calibration error, failure review, and dataset limitations are recorded. Predictions must support abstention and must retain model and manifest versions.

