from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import os

app = FastAPI(
    title="Pediatric Pneumonia Research API",
    version="0.1.0",
    description=(
        "Research-only API for the pediatric chest X-ray project. "
        "Not clinically validated."
    ),
)

MODEL_PATH = os.getenv("MODEL_PATH")
MODEL_READY = bool(MODEL_PATH and os.path.exists(MODEL_PATH))

CLASS_NAMES = [
    "NORMAL",
    "BACTERIAL_PNEUMONIA",
    "VIRAL_PNEUMONIA",
]


@app.get("/")
def root():
    return {
        "project": "Explainable Deep Learning for Multiclass Pediatric Pneumonia Classification",
        "status": "deployment starter",
        "research_use_only": True,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_ready": MODEL_READY,
        "architecture": "DenseNet-121",
        "classes": CLASS_NAMES,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        payload = await file.read()
        image = Image.open(io.BytesIO(payload)).convert("L")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a readable image.",
        ) from exc

    if not MODEL_READY:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model checkpoint is not configured yet. "
                "Set MODEL_PATH after the deployment checkpoint is frozen."
            ),
        )

    # The real preprocessing + DenseNet-121 inference will be connected here
    # once the exact deployment checkpoint and preprocessing contract are frozen.
    #
    # We intentionally do not return mock medical predictions.

    return JSONResponse(
        status_code=503,
        content={
            "detail": "Inference adapter not connected yet.",
            "image_size": list(image.size),
            "research_use_only": True,
        },
    )
