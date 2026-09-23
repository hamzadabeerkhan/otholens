# OrthoLens

OrthoLens is an early research prototype for analysing eligible knee radiographs. It is not for clinical diagnosis, triage, treatment planning, or patient use.

The first vertical slice contains:

- a Next.js browser interface;
- a Python FastAPI service with health and guarded analysis endpoints;
- deterministic PNG and JPEG validation and preprocessing;
- a versioned research-result contract;
- Docker Compose development and test commands.

The API keeps placeholder probabilities enabled by default. A locally mounted,
research-only PyTorch checkpoint can be enabled explicitly after review:

```bash
ORTHOLENS_ENABLE_MODEL=true docker compose up --build
```

The checkpoint is read from `artifacts/models/` and is never committed to Git.
If the model is below the confidence threshold, the API abstains instead of
reporting a KL grade. Every response remains marked research-only.

## Start locally

Docker with Docker Compose is required.

```bash
cp .env.example .env
docker compose up --build
```

Open [http://localhost:3000](http://localhost:3000). The API health endpoint is available at [http://localhost:8000/health](http://localhost:8000/health), and interactive API documentation is at [http://localhost:8000/docs](http://localhost:8000/docs).

## Run tests

```bash
docker compose run --build api-tests
```

## Repository safety

Never commit medical images, datasets, patient identifiers, credentials, restricted model weights, or notebooks containing embedded source images. Dataset access and licence terms must be approved and recorded before training.

## Current scope

The first milestone is one reproducible flow from a cleared or synthetic knee X-ray to validated preprocessing, placeholder KL-grade probabilities, structured JSON, and browser presentation. Model training, DICOM ingestion, persistence, segmentation, landmarks, measurements, and estimated 3D reconstruction follow only after their evidence gates are satisfied.
