from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageChops
import hashlib
import io
import os
import warnings

import numpy as np
import torch
from torchvision import models


APP_TITLE = "Pediatric Pneumonia Research API"
MODEL_PATH = os.getenv("MODEL_PATH", "/app/model/fold4_best.pt")

EXPECTED_SHA256 = "404deb43f00463371c6898360e7b89231846c9c5684b4284891db2d87cb480e4"

CLASS_NAMES = [
    "NORMAL",
    "BACTERIAL_PNEUMONIA",
    "VIRAL_PNEUMONIA",
]

MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(
    title=APP_TITLE,
    version="0.2.0",
    description=(
        "Research-only API for multiclass pediatric chest X-ray classification. "
        "Not clinically validated."
    ),
)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def construct_model():
    model = models.densenet121(weights=None)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 3)
    return model


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, f"Checkpoint not found at {MODEL_PATH}"

    observed_hash = sha256_file(MODEL_PATH)
    if observed_hash != EXPECTED_SHA256:
        return None, (
            "Checkpoint SHA-256 mismatch. "
            f"Expected {EXPECTED_SHA256}, got {observed_hash}"
        )

    payload = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(payload, dict) or "model" not in payload:
        return None, "Unexpected checkpoint structure."

    model = construct_model()
    model.load_state_dict(payload["model"], strict=True)
    model = model.to(DEVICE).eval()

    return model, None


MODEL, MODEL_ERROR = load_model()


def resize_geometry(width: int, height: int, size: int = 224):
    scale = min(size / width, size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))

    left = (size - resized_width) // 2
    top = (size - resized_height) // 2

    return (resized_width, resized_height), (left, top, 0, 0)


def read_grayscale_bytes(payload: bytes) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)

            with Image.open(io.BytesIO(payload)) as check:
                check.verify()

            with Image.open(io.BytesIO(payload)) as source:
                source.load()

                if source.mode == "L":
                    return source.copy()

                if source.mode == "RGB":
                    red, green, blue = source.split()

                    if (
                        ImageChops.difference(red, green).getbbox()
                        or ImageChops.difference(red, blue).getbbox()
                    ):
                        raise ValueError(
                            "RGB channels differ; this research pipeline only "
                            "accepts grayscale or identical-channel RGB images."
                        )

                    return red.copy()

    except Exception as exc:
        raise ValueError("Unreadable or unsupported image.") from exc

    raise ValueError("Unsupported image mode.")


def preprocess_image(image: Image.Image) -> torch.Tensor:
    # C1 contrast policy: unchanged grayscale.
    width, height = image.size

    resized, pad = resize_geometry(width, height, 224)
    scaled = image.resize(resized, Image.Resampling.BILINEAR)

    canvas = Image.new("L", (224, 224), 0)
    canvas.paste(scaled, pad[:2])

    values = np.asarray(canvas, dtype=np.float32) / np.float32(255.0)

    rgb = np.repeat(values[None, :, :], 3, axis=0)
    rgb = (rgb - MEAN[:, None, None]) / STD[:, None, None]

    tensor = torch.from_numpy(rgb.astype(np.float32)).unsqueeze(0)
    return tensor


@app.get("/")
def root():
    return {
        "project": "Explainable Deep Learning for Multiclass Pediatric Pneumonia Classification",
        "architecture": "DenseNet-121",
        "checkpoint": "Fold 4 BEST epoch 3",
        "research_use_only": True,
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if MODEL is not None else "model_not_ready",
        "model_ready": MODEL is not None,
        "model_error": MODEL_ERROR,
        "device": str(DEVICE),
        "architecture": "DenseNet-121",
        "geometry": "G1",
        "contrast": "C1",
        "image_size": "224x224",
        "checkpoint_sha256": EXPECTED_SHA256,
        "classes": CLASS_NAMES,
        "research_use_only": True,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail=MODEL_ERROR or "Model is not ready.",
        )

    payload = await file.read()

    try:
        image = read_grayscale_bytes(payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    tensor = preprocess_image(image).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

    prediction_index = int(np.argmax(probabilities))
    prediction = CLASS_NAMES[prediction_index]
    confidence = float(probabilities[prediction_index])

    result = {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": {
            CLASS_NAMES[i]: float(probabilities[i])
            for i in range(len(CLASS_NAMES))
        },
        "model": {
            "architecture": "DenseNet-121",
            "fold": 4,
            "best_epoch": 3,
            "geometry": "G1",
            "contrast": "C1",
            "checkpoint_sha256": EXPECTED_SHA256,
        },
        "research_use_only": True,
        "warning": (
            "Research prototype only. This output is not a clinical diagnosis "
            "and must not be used for patient-care decisions."
        ),
    }

    return JSONResponse(content=result)
