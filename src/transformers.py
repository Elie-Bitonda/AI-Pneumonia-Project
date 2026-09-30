"""Read-only image inspection and conservative filename parsing."""

import hashlib
import re
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

LABELS = ("NORMAL", "BACTERIAL_PNEUMONIA", "VIRAL_PNEUMONIA")
EXTENSIONS = {".jpg", ".jpeg", ".png"}
METRICS = (
    "width", "height", "aspect_ratio", "file_size_bytes",
    "mean_intensity", "std_intensity",
)


def is_filesystem_metadata(path):
    """Exclude packaging metadata from modeling even if it can be decoded."""
    path = Path(path)
    return path.name.startswith("._") or "__MACOSX" in path.parts


def parse_identity(relative_path):
    """Require one class folder and an unambiguous whole subtype token.

    personNNN is a candidate patient identifier, never a confirmed identity.
    IM/NORMAL2-IM codes are not assumed to be globally unique patient IDs.
    """
    path = Path(relative_path)
    folders = {p.upper() for p in path.parts[:-1]} & {"NORMAL", "PNEUMONIA"}
    splits = {p.lower() for p in path.parts[:-1]} & {"train", "val", "test"}
    tokens = set(re.split(r"[^a-z0-9]+", path.stem.lower()))
    subtypes = tokens & {"bacteria", "virus"}
    label = None
    reason = "ambiguous_or_missing_class_folder"
    if folders == {"NORMAL"} and not subtypes:
        label, reason = "NORMAL", "normal_folder"
    elif folders == {"PNEUMONIA"} and len(subtypes) == 1:
        label = ("BACTERIAL_PNEUMONIA" if "bacteria" in subtypes
                 else "VIRAL_PNEUMONIA")
        reason = "whole_filename_subtype_token"
    elif len(folders) == 1:
        reason = "missing_or_conflicting_subtype"
    match = re.fullmatch(
        r"person([0-9]+)_(bacteria|virus)_([0-9]+)", path.stem.lower()
    )
    candidate = f"person{int(match[1])}" if match else None
    return {
        "original_split": (
            next(iter(splits)) if len(splits) == 1 else "UNKNOWN"
        ),
        "original_folder": path.parent.name,
        "target_label": label,
        "patient_id_if_parseable": candidate,
        "patient_id_status": (
            "CANDIDATE_UNVERIFIED" if candidate else "UNPARSEABLE"
        ),
        "label_parse_status": "OK" if label else "FAILED",
        "label_parse_reason": reason,
    }


def inspect_image(path, root):
    """Hash file bytes and decode without saving, resizing or changing pixels.

    Statistics use float64 over stored channels; palette images are expanded
    to RGB for meaningful intensities. Alpha is excluded. ddof=0 describes
    the full image population. EXIF orientation is not applied.
    """
    row = {
        "filepath": path.relative_to(root).as_posix(), "filename": path.name,
        **parse_identity(path.relative_to(root)),
        "width": None, "height": None, "aspect_ratio": None,
        "image_mode": None, "channel_count": None, "file_size_bytes": None,
        "mean_intensity": None, "std_intensity": None,
        "min_intensity": None, "max_intensity": None,
        "readable": False, "sha256_hash": None,
        "intensity_basis": None, "read_error": None, "hash_error": None,
        "is_macos_sidecar": is_filesystem_metadata(path),
    }
    try:
        row["file_size_bytes"] = path.stat().st_size
        with path.open("rb") as stream:
            row["sha256_hash"] = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
    except OSError as exc:
        row["hash_error"] = f"{type(exc).__name__}: {exc}"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as check:
                check.verify()
            with Image.open(path) as img:
                img.load()
                mode = img.mode
                diagnostic = img.convert("RGB") if mode == "P" else img
                pixels = np.asarray(diagnostic, dtype=np.float64)
                if "A" in diagnostic.getbands() and pixels.ndim == 3:
                    pixels = pixels[..., :-1]
                row.update(
                    width=img.width, height=img.height,
                    aspect_ratio=img.width / img.height, image_mode=mode,
                    channel_count=len(img.getbands()),
                    mean_intensity=float(pixels.mean()),
                    std_intensity=float(pixels.std(ddof=0)),
                    min_intensity=float(pixels.min()),
                    max_intensity=float(pixels.max()), readable=True,
                    intensity_basis=(
                        f"{diagnostic.mode};alpha_excluded;native_scale"
                    ),
                )
    except Exception as exc:  # One damaged file must not abort the inventory.
        row["read_error"] = f"{type(exc).__name__}: {exc}"
    return row
