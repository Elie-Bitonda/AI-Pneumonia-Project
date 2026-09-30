"""Deterministic Phase 1 audit CLI. Does not build or fit a model."""

import argparse
import json
import platform
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL

from .transformers import EXTENSIONS, LABELS, METRICS, inspect_image

matplotlib.use("Agg")


def quality_audit(frame):
    """Linear quartiles and Tukey 1.5 IQR fences flag, never exclude, rows."""
    summaries, flags = [], []
    for metric in METRICS:
        values = frame.loc[frame.readable, metric].dropna()
        if values.empty:
            continue
        q1, median, q3 = values.quantile(
            [0.25, 0.5, 0.75], interpolation="linear"
        )
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        summaries.append({
            "metric": metric, "n": len(values), "min": values.min(),
            "q1": q1, "median": median, "q3": q3, "max": values.max(),
            "iqr": iqr, "lower_fence": lower, "upper_fence": upper,
        })
        mask = frame.readable & (
            (frame[metric] < lower) | (frame[metric] > upper)
        )
        for _, row in frame.loc[mask].iterrows():
            flags.append({
                "filepath": row.filepath, "metric": metric,
                "value": row[metric], "lower_fence": lower,
                "upper_fence": upper, "reason": "outside_1.5_IQR_fences",
            })
    return pd.DataFrame(summaries), pd.DataFrame(flags, columns=[
        "filepath", "metric", "value", "lower_fence", "upper_fence", "reason",
    ])


def class_table(frame):
    """Show explicit denominators; never silently deduplicate."""
    counts = frame.target_label.value_counts().reindex(LABELS, fill_value=0)
    readable = frame.loc[frame.readable, "target_label"].value_counts()
    valid_n = int(counts.sum())
    result = pd.DataFrame({"target_label": LABELS, "count": counts.values})
    result["percent_of_all_discovered"] = result["count"] / len(frame) * 100
    result["percent_of_parsed_labels"] = (
        result["count"] / valid_n * 100 if valid_n else np.nan
    )
    result["readable_count"] = [int(readable.get(k, 0)) for k in LABELS]
    result["ratio_to_smallest_class"] = (
        result["count"] / counts.min() if counts.min() else np.nan
    )
    return result


def write_reports(frame, output, figures, root):
    """Emit detailed review tables and a compact machine-readable summary."""
    frame.to_csv(output / "image_metadata_audit.csv", index=False)
    missing = frame.isna().sum().rename("missing_count").to_frame()
    missing["missing_percent"] = missing.missing_count / len(frame) * 100
    missing.to_csv(output / "missing_values.csv", index_label="column")
    core = [c for c in frame if c not in {"read_error", "hash_error"}]
    duplicates = frame[
        frame.sha256_hash.notna() & frame.duplicated("sha256_hash", keep=False)
    ]
    duplicates.to_csv(output / "duplicate_hashes.csv", index=False)
    frame[frame.duplicated("filename", keep=False)].to_csv(
        output / "duplicate_filenames.csv", index=False
    )
    frame[frame.label_parse_status == "FAILED"].to_csv(
        output / "label_failures.csv", index=False
    )
    frame[~frame.readable].to_csv(output / "unreadable_files.csv", index=False)
    conflicts = frame.groupby("sha256_hash").target_label.nunique()
    frame[frame.sha256_hash.isin(conflicts[conflicts > 1].index)].to_csv(
        output / "duplicate_label_conflicts.csv", index=False
    )
    robust, flags = quality_audit(frame)
    robust.to_csv(output / "quality_summary.csv", index=False)
    flags.to_csv(output / "image_quality_flags.csv", index=False)
    classes = class_table(frame)
    classes.to_csv(output / "class_distribution.csv", index=False)
    unique = frame[frame.readable & frame.sha256_hash.notna()].drop_duplicates(
        "sha256_hash", keep="first"
    )
    if not unique.empty:
        class_table(unique).to_csv(
            output / "class_distribution_unique_readable_hashes.csv",
            index=False,
        )
    patients = frame[frame.patient_id_if_parseable.notna()].groupby(
        "patient_id_if_parseable", sort=True
    ).agg(
        file_count=("filepath", "size"),
        distinct_byte_images=("sha256_hash", "nunique"),
        source_split_count=(
            "original_split", lambda s: s[s != "UNKNOWN"].nunique()
        ),
        source_splits=("original_split", lambda s: "|".join(sorted(set(s)))),
        label_count=("target_label", "nunique"),
    )
    patients.to_csv(output / "patient_candidate_audit.csv")
    frame.groupby(["image_mode", "channel_count"], dropna=False).size().to_csv(
        output / "image_mode_channels.csv", header=["count"]
    )
    frame.original_split.value_counts().sort_index().to_csv(
        output / "original_split_counts.csv", header=["count"]
    )
    summary = {
        "dataset_root_name": root.name,
        "total_discovered": len(frame), "readable": int(frame.readable.sum()),
        "unreadable": int((~frame.readable).sum()),
        "unreadable_macos_sidecars": int(
            ((~frame.readable) & frame.is_macos_sidecar).sum()
        ),
        "unique_readable_sha256": int(
            frame.loc[frame.readable, "sha256_hash"].nunique()
        ),
        "readable_hash_label_conflict_groups": int(
            frame[frame.readable].groupby("sha256_hash")
            .target_label.nunique().gt(1).sum()
        ),
        "readable_hash_groups_cross_source_splits": int(
            frame[frame.readable].groupby("sha256_hash")
            .original_split.nunique().gt(1).sum()
        ),
        "unique_sha256": int(frame.sha256_hash.nunique()),
        "duplicate_hash_groups": int(duplicates.sha256_hash.nunique()),
        "files_in_duplicate_hash_groups": len(duplicates),
        "redundant_byte_copies": int(
            frame.sha256_hash.count() - frame.sha256_hash.nunique()
        ),
        "duplicate_filename_groups": int(frame.loc[
            frame.duplicated("filename", keep=False), "filename"].nunique()),
        "label_failures": int((frame.label_parse_status == "FAILED").sum()),
        "hash_label_conflict_groups": int((conflicts > 1).sum()),
        "complete_metadata_rows_all_columns": int(
            frame.notna().all(axis=1).sum()
        ),
        "rows_with_missing_values_all_columns": int(
            frame.isna().any(axis=1).sum()
        ),
        "complete_rows_excluding_optional_error_fields": int(
            frame[core].notna().all(axis=1).sum()),
        "unique_candidate_patient_ids": len(patients),
        "confirmed_patient_ids": 0,
        "files_with_candidate_patient_id": int(
            frame.patient_id_if_parseable.notna().sum()
        ),
        "candidate_patients_multiple_files": int(
            (patients.file_count > 1).sum()
        ),
        "candidate_patients_multiple_distinct_hashes": int(
            (patients.distinct_byte_images > 1).sum()),
        "candidate_patients_cross_source_splits": int(
            (patients.source_split_count > 1).sum()),
        "candidate_patients_multiple_labels": int(
            (patients.label_count > 1).sum()
        ),
        "grouping_assessment": (
            "Duplicate isolation required; patient grouping pending ID "
            "validation. Partial candidate IDs do not justify independent "
            "per-image folds."
        ),
        "quality_flag_rows": len(flags),
        "quality_flagged_images": int(flags.filepath.nunique()),
        "class_max_min_ratio": (
            float(classes["count"].max() / classes["count"].min())
            if classes["count"].min() else None
        ),
        "original_split_counts": frame.original_split.value_counts().to_dict(),
        "versions": {
            "python": platform.python_version(), "pillow": PIL.__version__,
            "numpy": np.__version__, "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    (output / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    bars = ax.bar(["Normal", "Bacterial pneumonia", "Viral pneumonia"],
                  classes["count"], color=["#3477a5", "#d28336", "#6f5495"])
    ax.bar_label(bars, padding=3)
    ax.set(
        ylabel="Discovered files with parsed labels",
        title="Source inventory: includes duplicate and unreadable files",
    )
    ax.set_ylim(0, max(1, classes["count"].max()) * 1.15)
    fig.savefig(figures / "class_distribution.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


def run_audit(root, output, figures):
    """Reject destinations that could overwrite raw image directories."""
    root, output, figures = [p.resolve() for p in (root, output, figures)]
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    for destination in (output, figures):
        if (destination == root or root in destination.parents
                or destination in root.parents):
            raise ValueError("Output directories must be disjoint from raw.")
    paths = sorted(
        (p for p in root.rglob("*")
         if p.is_file() and p.suffix.lower() in EXTENSIONS),
        key=lambda p: p.relative_to(root).as_posix(),
    )
    if not paths:
        raise ValueError("No JPEG/JPG/PNG files; no audit outputs written.")
    rows = []
    for index, path in enumerate(paths, 1):
        rows.append(inspect_image(path, root))
        if index % 500 == 0:
            print(f"Inspected {index}/{len(paths)} files", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    write_reports(pd.DataFrame(rows), output, figures, root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed")
    )
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    run_audit(args.dataset_root, args.output_dir, args.figures_dir)


if __name__ == "__main__":
    main()
