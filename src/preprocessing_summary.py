"""Generate reproducible Phase 1 preprocessing summary tables.

This module reads existing audit manifests and writes the metadata inventory,
missingness summary, canonical outlier bounds, and diagnostic VIF table used in
the preprocessing strategy. It does not read or modify raw image bytes and does
not fit a predictive model.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .transformers import METRICS

TEXT = "text/path/hash/identifier"
NOMINAL = "nominal categorical"
DESCRIPTIONS = {
    "filepath": ("Selected/source path relative to dataset root", TEXT),
    "filename": ("Original file basename", TEXT),
    "original_split": ("Source train/val/test or UNKNOWN; provenance only", NOMINAL),
    "original_folder": ("Source parent folder name", NOMINAL),
    "target_label": ("Nominal three-class target Y", NOMINAL),
    "patient_id_if_parseable": ("Unverified filename-derived ID candidate", TEXT),
    "patient_id_status": ("Candidate-ID parse/confidence status", NOMINAL),
    "label_parse_status": ("Success or failure of target parsing", NOMINAL),
    "label_parse_reason": ("Categorical rule/reason used by label parser", NOMINAL),
    "width": ("Stored image width in pixels", "discrete"),
    "height": ("Stored image height in pixels", "discrete"),
    "aspect_ratio": ("Width divided by height", "continuous"),
    "image_mode": ("Pillow storage mode, e.g. L or RGB", NOMINAL),
    "channel_count": ("Number of stored image bands", "discrete"),
    "file_size_bytes": ("Original file size in bytes", "discrete"),
    "mean_intensity": ("Mean decoded intensity, native scale", "continuous"),
    "std_intensity": ("Population SD of decoded intensities", "continuous"),
    "min_intensity": ("Minimum intensity; integer for observed 8-bit images", "discrete"),
    "max_intensity": ("Maximum intensity; integer for observed 8-bit images", "discrete"),
    "readable": ("Full image verification/decode succeeded", "boolean"),
    "sha256_hash": ("SHA-256 digest of original file bytes", TEXT),
    "intensity_basis": ("Mode/alpha/native-scale measurement convention", NOMINAL),
    "read_error": ("Read failure text; not applicable on success", TEXT),
    "hash_error": ("Hash failure text; not applicable on success", TEXT),
    "is_macos_sidecar": ("Path/name identifies filesystem metadata", "boolean"),
    "all_filepaths": ("JSON array of all paths with this byte hash", TEXT),
    "all_original_splits": ("JSON array of source split names", TEXT),
    "all_target_labels": ("JSON array of source annotations", TEXT),
    "all_filenames": ("JSON array of source basenames", TEXT),
    "source_file_count": ("Number of source paths for this hash", "discrete"),
    "alternative_filepaths": ("JSON array of non-selected source paths", TEXT),
    "manifest_status": ("Provisional inclusion/selection status", NOMINAL),
    "patient_grouping_validated": ("Whether patient groups are validated", "boolean"),
}


def read_table(path):
    """Read audit CSVs while treating only empty fields as missing."""
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def variable_inventory(canonical, original):
    rows = []
    for position, column in enumerate(canonical.columns, 1):
        description, kind = DESCRIPTIONS[column]
        rows.append({
            "column_position": position,
            "column_name": column,
            "semantic_description": description,
            "storage_dtype": str(canonical[column].dtype),
            "assignment_data_type": kind,
            "original_audit_storage_dtype": (
                str(original[column].dtype) if column in original else "ABSENT"
            ),
            "role": "TARGET_Y" if column == "target_label" else "DIAGNOSTIC_METADATA",
            "used_as_CNN_predictor": False,
        })
    return pd.DataFrame(rows)


def missingness(table, name):
    """Summarize column and matrix completeness including structural blanks."""
    mask = table.isna()
    rows = [{
        "dataset": name,
        "record_type": "COLUMN",
        "column_name": column,
        "missing_count": int(mask[column].sum()),
        "missing_percentage": float(mask[column].mean() * 100),
    } for column in table]
    optional = ["read_error", "hash_error"]
    rows.append({
        "dataset": name,
        "record_type": "MATRIX_SUMMARY",
        "column_name": "ALL",
        "missing_count": int(mask.to_numpy().sum()),
        "missing_percentage": float(mask.to_numpy().mean() * 100),
        "total_rows": len(table),
        "total_columns": len(table.columns),
        "total_matrix_cells": table.size,
        "total_missing_cells": int(mask.to_numpy().sum()),
        "overall_missing_percentage": float(mask.to_numpy().mean() * 100),
        "complete_rows": int((~mask.any(axis=1)).sum()),
        "rows_with_missing_values": int(mask.any(axis=1).sum()),
        "complete_rows_excluding_optional_errors": int(
            table.drop(columns=optional).notna().all(axis=1).sum()
        ),
    })
    return pd.DataFrame(rows)


def outlier_bounds(frame):
    rows = []
    shape_reason = {
        "width": "Resolution mixture and right tail; robust IQR preferred",
        "height": "Right tail and different class distributions; IQR preferred",
        "aspect_ratio": "Bounded positive ratios with long right tail; IQR preferred",
        "file_size_bytes": "Strong skew and long storage-size tail; IQR preferred",
        "mean_intensity": (
            "Near-symmetric pooled shape; Z-score is a sensitivity check, "
            "while IQR remains the class-blind review rule"
        ),
        "std_intensity": (
            "Mild pooled skew; Z-score is a sensitivity check, while IQR "
            "remains the class-blind review rule"
        ),
    }
    masks = []
    for metric in METRICS:
        x = frame[metric]
        q1, q3 = x.quantile([0.25, 0.75], interpolation="linear")
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = x.lt(lower) | x.gt(upper)
        masks.append(mask.to_numpy())
        rows.append({
            "variable": metric,
            "N": int(x.count()),
            "mean": x.mean(),
            "median": x.median(),
            "skewness": x.skew(),
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "lower_fence": lower,
            "upper_fence": upper,
            "number_below": int(x.lt(lower).sum()),
            "number_above": int(x.gt(upper).sum()),
            "total_flagged": int(mask.sum()),
            "preferred_method": "IQR_REVIEW_ONLY",
            "method_rationale": shape_reason[metric],
        })
    return pd.DataFrame(rows), int(np.logical_or.reduce(masks).sum())


def calculate_vif(frame):
    """Compute diagnostic VIF by auxiliary OLS with an intercept."""
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("VIF requires complete finite diagnostics.")
    scales = values.std(axis=0, ddof=0)
    if (scales == 0).any():
        raise ValueError("VIF undefined for a constant diagnostic.")
    z = (values - values.mean(axis=0)) / scales
    rows = []
    for j, variable in enumerate(frame.columns):
        y = z[:, j]
        design = np.column_stack([np.ones(len(z)), np.delete(z, j, axis=1)])
        coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
        sse = float(np.sum((y - design @ coefficients) ** 2))
        sst = float(np.sum((y - y.mean()) ** 2))
        ratio = sse / sst
        vif = float("inf") if ratio <= 1e-14 else 1 / ratio
        rows.append({
            "variable": variable,
            "N": len(frame),
            "auxiliary_R_squared": 1 - ratio,
            "VIF": vif,
            "VIF_gt_10": vif > 10,
            "auxiliary_design_rank": int(rank),
            "intercept_included": True,
            "used_as_CNN_feature": False,
        })
    return pd.DataFrame(rows)


def run(output):
    canonical = read_table(output / "canonical_image_manifest.csv")
    original = read_table(output / "image_metadata_audit.csv")
    if not canonical.sha256_hash.is_unique:
        raise ValueError("Canonical contents must be unique.")
    if canonical[list(METRICS)].isna().any().any():
        raise ValueError("Canonical diagnostics unexpectedly missing.")

    inventory = variable_inventory(canonical, original)
    missing = pd.concat([
        missingness(original, "original_audit"),
        missingness(canonical, "canonical_manifest"),
    ], ignore_index=True)
    bounds, union = outlier_bounds(canonical)
    vif = calculate_vif(canonical[list(METRICS)])

    outputs = {
        "phase1_variable_inventory.csv": inventory,
        "phase1_missingness_summary.csv": missing,
        "outlier_bounds_and_skewness.csv": bounds,
        "metadata_vif.csv": vif,
    }
    for name, table in outputs.items():
        table.to_csv(output / name, index=False)

    print("Canonical N x P:", canonical.shape)
    print("Variable types:", inventory.assignment_data_type.value_counts().to_dict())
    print(missing[missing.record_type == "MATRIX_SUMMARY"].to_string(index=False))
    print(bounds.drop(columns=["method_rationale"]).to_string(index=False))
    print("Unique contents flagged by canonical IQR fences:", union)
    print(vif.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
