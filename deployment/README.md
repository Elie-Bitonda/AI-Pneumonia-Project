# Deployment Starter

This folder begins the deployment phase for the pediatric pneumonia research project.

> Research use only. The model is not clinically validated.

## Current state

- Architecture direction: **DenseNet-121**
- API framework: **FastAPI**
- Containerization: **Docker**
- Planned cloud target: **Azure Container Apps**
- `GET /health`: available now
- `POST /predict`: route created; real model inference will be connected after the exact checkpoint and preprocessing contract are frozen

No mock medical predictions are returned.

## Run locally

From this folder:

```bash
docker build -t pneumonia-api .
docker run --rm -p 8000:8000 pneumonia-api
```

Then open:

```text
http://localhost:8000/
http://localhost:8000/health
http://localhost:8000/docs
```

## Model checkpoint

The exact deployment checkpoint is intentionally not hard-coded into the repository.

When the checkpoint is frozen, the container will receive its location through:

```bash
MODEL_PATH=/path/to/checkpoint.pt
```

The inference adapter will then load the model once at application startup.

## Intended API response

After the inference adapter is connected, `POST /predict` will return the three research-label probabilities:

```json
{
  "prediction": "BACTERIAL_PNEUMONIA",
  "confidence": 0.84,
  "probabilities": {
    "NORMAL": 0.03,
    "BACTERIAL_PNEUMONIA": 0.84,
    "VIRAL_PNEUMONIA": 0.13
  },
  "research_use_only": true
}
```

## Azure direction

Once local Docker testing passes, the same container can be deployed to Azure Container Apps and exposed through a public HTTPS endpoint.

The next implementation step is to connect the frozen DenseNet-121 checkpoint and its exact preprocessing pipeline to `/predict`.
