"""I keep on-demand image preprocessing separate from metadata diagnostics."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import warnings

import numpy as np
from PIL import Image, ImageChops
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.utils.validation import check_is_fitted

from .preprocessing_summary import calculate_vif
from .statistical_audit import mi_diagnostic
from .transformers import METRICS, is_filesystem_metadata

NUMERIC_PREDICTORS = tuple(METRICS)


def resize_geometry(width, height, size=224):
    if min(width, height, size) <= 0:
        raise ValueError("Positive dimensions required.")
    scale = min(size / width, size / height)
    resized = (max(1, round(width * scale)), max(1, round(height * scale)))
    dx, dy = size - resized[0], size - resized[1]
    return resized, (dx // 2, dy // 2, dx - dx // 2, dy - dy // 2)


@dataclass(frozen=True)
class AugmentationConfig:
    """I use seed-controlled rotation and translation only during training."""
    rotation_degrees: float = 5.0
    translation_fraction: float = 0.02

    def __post_init__(self):
        if not 0 <= self.rotation_degrees <= 5 or not 0 <= self.translation_fraction <= .02:
            raise ValueError("Augmentation exceeds the Phase 1 conservative bounds.")


def augment_training(image, config, seed, role="train"):
    if role != "train":
        raise ValueError("Augmentation is restricted to the actual training role.")
    if seed is None:
        raise ValueError("An explicit sample/epoch seed is required.")
    rng = np.random.default_rng(seed)
    angle = rng.uniform(-config.rotation_degrees, config.rotation_degrees)
    translation = tuple(rng.uniform(-config.translation_fraction, config.translation_fraction, 2)
                        * np.asarray(image.size))
    return image.rotate(float(angle), resample=Image.Resampling.BILINEAR,
                        translate=translation, expand=False, fillcolor=0)


def preprocess_image(path, size=224, augmentation=None, seed=None, role="evaluation"):
    """I return H x W float32 intensities without writing a transformed image.

    I use Pillow's antialiased bilinear downsampling, stored orientation, and
    black padding. I reject unexpected color rather than silently discarding it.
    """
    if is_filesystem_metadata(path):
        raise ValueError("Filesystem metadata is not a modeling image.")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(path) as check:
            check.verify()
        with Image.open(path) as source:
            source.load()
            if source.mode == "L":
                intensity = source.copy()
            elif source.mode == "RGB":
                r, g, b = source.split()
                if ImageChops.difference(r, g).getbbox() or ImageChops.difference(r, b).getbbox():
                    raise ValueError("RGB channels differ; channel policy requires review.")
                intensity = r
            else:
                raise ValueError(f"Unreviewed image mode: {source.mode}")
    resized, padding = resize_geometry(*intensity.size, size)
    intensity = intensity.resize(resized, Image.Resampling.BILINEAR)
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(intensity, padding[:2])
    if augmentation is not None:
        canvas = augment_training(canvas, augmentation, seed, role)
    return np.asarray(canvas, dtype=np.float32) / np.float32(255)


def load_canonical_image(root, row, **kwargs):
    root = Path(root).resolve()
    path = (root / row["filepath"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Canonical image must exist inside the dataset root.")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != row["sha256_hash"]:
            raise ValueError("Canonical image bytes differ from the manifest.")
    return preprocess_image(path, **kwargs)


def pretrained_representation(intensity, *, published_mean=None, published_std=None):
    """I replicate to CHW; optional normalization must name published constants.

    I leave the selected weights' normalization explicit. I do not run a
    weights preset that could silently introduce a second resize/center crop.
    """
    values = np.asarray(intensity, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise ValueError("Expected a finite H x W intensity array in [0,1].")
    rgb = np.repeat(values[None], 3, axis=0)
    if (published_mean is None) != (published_std is None):
        raise ValueError("Both published mean and standard deviation are required.")
    if published_mean is not None:
        mean = np.asarray(published_mean, dtype=np.float32)
        std = np.asarray(published_std, dtype=np.float32)
        if mean.shape != (3,) or std.shape != (3,) or not np.isfinite([mean, std]).all() or (std <= 0).any():
            raise ValueError("Three finite constants and positive scales required.")
        rgb = (rgb - mean[:, None, None]) / std[:, None, None]
    return rgb


class RankVIFReducer(TransformerMixin, BaseEstimator):
    """I learn the documented redundancy rule on training diagnostics only."""
    def __init__(self, columns=NUMERIC_PREDICTORS, threshold=10.0):
        self.columns = columns
        self.threshold = threshold

    def fit(self, X, y=None):
        frame = pd_frame(X, self.columns)
        self.n_features_in_ = frame.shape[1]
        self.fit_rows_ = len(frame)
        self.raw_vif_ = calculate_vif(frame)
        self.initial_rank_vif_ = calculate_vif(frame.rank(method="average"))
        retained = list(self.columns)
        self.removals_ = []
        if self.initial_rank_vif_.VIF.max() > self.threshold and "aspect_ratio" in retained:
            retained.remove("aspect_ratio")
            self.removals_.append({"column": "aspect_ratio", "reason": "structurally derived redundancy first"})
        while len(retained) > 1:
            vif = calculate_vif(frame[retained].rank(method="average")).set_index("variable").VIF
            maximum = vif.max()
            if maximum <= self.threshold:
                break
            tied = [c for c in retained if (np.isinf(maximum) and np.isinf(vif[c]))
                    or (np.isfinite(maximum) and maximum - vif[c] <= .01)]
            if len(tied) > 1 and y is None:
                raise ValueError("Training labels are required for the documented MI tie break.")
            chosen = min(tied, key=lambda c: (mi_diagnostic(frame[c], y, True)["mi_nats"] if len(tied) > 1 else 0, c))
            retained.remove(chosen)
            self.removals_.append({"column": chosen, "reason": "largest rank-VIF > threshold; MI then alphabetical ties"})
        self.selected_columns_ = retained
        self.indices_ = [list(self.columns).index(c) for c in retained]
        self.final_rank_vif_ = calculate_vif(frame[retained].rank(method="average"))
        return self

    def transform(self, X):
        check_is_fitted(self, "indices_")
        values = pd_frame(X, self.columns).to_numpy()
        return values[:, self.indices_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "selected_columns_")
        return np.asarray(self.selected_columns_, dtype=object)


def pd_frame(values, columns):
    import pandas as pd
    frame = pd.DataFrame(np.asarray(values, dtype=float), columns=columns)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("I do not impute unexpected missing/nonfinite diagnostic predictors.")
    return frame


def require_complete_predictors(values):
    """I reject unexpected missingness before fitting any diagnostic scaler."""
    if not np.isfinite(np.asarray(values, dtype=float)).all():
        raise ValueError("Selected diagnostic predictors must be complete and finite.")
    return values


def build_metadata_pipeline(columns=NUMERIC_PREDICTORS, reduce_collinearity=True):
    """I exclude identifiers/targets with a numeric allowlist and remainder=drop."""
    columns = tuple(columns)
    if not columns or len(set(columns)) != len(columns) or not set(columns) <= set(NUMERIC_PREDICTORS):
        raise ValueError("Only the six documented numeric diagnostic columns are allowed.")
    numeric = Pipeline([("validate", FunctionTransformer(require_complete_predictors,
                                                        feature_names_out="one-to-one")),
                        ("scale", StandardScaler())])
    selection = ColumnTransformer([("numeric", numeric, list(columns))],
                                  remainder="drop", verbose_feature_names_out=False)
    steps = [("columns", selection)]
    if reduce_collinearity:
        steps.append(("rank_vif", RankVIFReducer(columns)))
    return Pipeline(steps)
