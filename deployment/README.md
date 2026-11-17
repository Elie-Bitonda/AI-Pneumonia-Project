# Deployment Starter

This folder contains the first real deployment package for the pediatric pneumonia research model.

> Research use only. The model is not clinically validated and must not be used for patient-care decisions.

## Frozen deployment model

- Architecture: **DenseNet-121**
- Representative checkpoint: **Fold 4**
- BEST epoch: **3**
- Geometry: **G1**
- Contrast: **C1**
- Training augmentation: **A1**
- Input size: **224 x 224**
- Class order:
  1. NORMAL
  2. BACTERIAL_PNEUMONIA
  3. VIRAL_PNEUMONIA
- Checkpoint SHA-256:
  `404deb43f00463371c6898360e7b89231846c9c5684b4284891db2d87cb480e4`

The full machine-readable contract is in `model_contract.json`.

## API routes

- `GET /` — project/model metadata
- `GET /health` — checkpoint readiness and hash status
- `POST /predict` — real DenseNet-121 inference

## Put the checkpoint in place

The checkpoint is intentionally not committed directly to GitHub.

Create:

```text
deployment/model/fold4_best.pt
```

or set another location with:

```bash
MODEL_PATH=/path/to/fold4_best.pt
```

The API verifies the checkpoint SHA-256 before loading it.

## Build locally

From `deployment/`:

```bash
docker build -t pneumonia-api .
```

If the checkpoint is stored in `deployment/model/fold4_best.pt`, run:

```bash
docker run --rm -p 8000:8000 pneumonia-api
```

If the checkpoint is outside the image, mount it:

```bash
docker run --rm \
  -p 8000:8000 \
  -v /absolute/path/to/fold4_best.pt:/app/model/fold4_best.pt:ro \
  pneumonia-api
```

Open:

```text
http://localhost:8000/
http://localhost:8000/health
http://localhost:8000/docs
```

## Test prediction

Using curl:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@example_xray.jpeg"
```

The response contains the predicted research label, confidence, and all three softmax probabilities.

## Azure deployment direction

The intended cloud target is **Azure Container Apps**.

For a quick first deployment, the model checkpoint can be included in the deployment build context or mounted/downloaded through a controlled storage mechanism. Before any public deployment, keep the checkpoint hash verification enabled.

The cloud service will expose the same `/health` and `/predict` endpoints over HTTPS.
