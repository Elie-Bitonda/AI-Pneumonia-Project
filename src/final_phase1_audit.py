"""Reproduce final Phase 1 statistics and human-review artifacts only."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import scipy
from scipy import stats

from .image_review import (
    inspect_canonical, near_duplicate_pairs, review_sheets,
)
from .integrity_verification import verify_sources
from .statistical_audit import analyze_metadata, quality_class_analysis
from .transformers import LABELS, METRICS, is_filesystem_metadata

matplotlib.use("Agg")


def validate_manifest(frame):
    """Reject duplicate content, missing diagnostics and ineligible files."""
    if frame.empty or not frame.sha256_hash.is_unique:
        raise ValueError("Canonical manifest requires unique image hashes.")
    if (not frame.readable.all()
            or frame.filepath.map(is_filesystem_metadata).any()):
        raise ValueError("Manifest contains unreadable/metadata files.")
    if not frame.target_label.isin(LABELS).all():
        raise ValueError("Unresolved target labels.")
    if not np.isfinite(frame[list(METRICS)].to_numpy(dtype=float)).all():
        raise ValueError("Missing or nonfinite diagnostic measurements.")
    return frame.sort_values("sha256_hash", kind="stable").reset_index(
        drop=True
    )


def statistical_figures(frame, correlation, figures):
    labels = ["Width", "Height", "Aspect ratio", "File size",
              "Mean intensity", "Intensity SD"]
    values = correlation.to_numpy()
    visible = np.ma.array(values, mask=np.tril(np.ones_like(values), k=-1))
    fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("white")
    display = ax.imshow(visible, vmin=-1, vmax=1, cmap=cmap)
    ax.set_xticks(range(6), labels, rotation=35, ha="right")
    ax.set_yticks(range(6), labels)
    for i in range(6):
        for j in range(i, 6):
            mark = "*" if i != j and abs(values[i, j]) >= 0.85 else ""
            ax.text(j, i, f"{values[i, j]:.2f}{mark}",
                    ha="center", va="center",
                    color="white" if abs(values[i, j]) > 0.6 else "black")
    ax.set_title("Spearman metadata correlations: canonical unique contents\n"
                 "* |rho| >= 0.85; diagnostic variables, not CNN predictors")
    fig.colorbar(display, ax=ax, label="Spearman rho", shrink=0.8)
    fig.savefig(figures / "fig3_1_correlation_matrix.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), layout="constrained")
    for metric, ax in zip(METRICS, axes.flat):
        edges = np.histogram_bin_edges(frame[metric], bins="fd")
        if len(edges) > 61:
            edges = np.linspace(frame[metric].min(), frame[metric].max(), 61)
        for label in LABELS:
            x = frame.loc[frame.target_label == label, metric]
            ax.hist(x, bins=edges, density=True, histtype="step",
                    linewidth=1.5,
                    label=label)
        ax.set(title=metric, ylabel="Density")
    axes[0, 0].legend(fontsize=8)
    fig.savefig(figures / "metadata_class_distributions.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(6, 3, figsize=(12, 19), layout="constrained")
    for i, metric in enumerate(METRICS):
        for j, label in enumerate(LABELS):
            stats.probplot(frame.loc[frame.target_label == label, metric],
                           dist="norm", fit=True, plot=axes[i, j])
            axes[i, j].set_title(f"{metric}\n{label}", fontsize=9)
    fig.savefig(figures / "metadata_normal_qq.png", dpi=120)
    plt.close(fig)


def run(root, output, figures):
    root, output, figures = [p.resolve() for p in (root, output, figures)]
    for destination in [output, figures]:
        if (destination == root or root in destination.parents
                or destination in root.parents):
            raise ValueError("Audit destinations must be disjoint from raw.")
    frame = validate_manifest(pd.read_csv(
        output / "canonical_image_manifest.csv"
    ))
    inventory = pd.read_csv(output / "image_metadata_audit.csv")
    verify_sources(inventory, root)
    print("All raw inventory hashes verified before analysis.", flush=True)
    associations, correlation = analyze_metadata(frame, output)
    statistical_figures(frame, correlation, figures)
    flags = pd.read_csv(output / "quality_flags_unique_content.csv")
    quality = quality_class_analysis(frame, flags)
    quality.to_csv(output / "quality_flag_class_analysis.csv", index=False)
    channels = inspect_canonical(frame, root, output)
    pairs = near_duplicate_pairs(frame, channels, radius=4)
    pairs.to_csv(output / "near_duplicate_candidates.csv", index=False)
    print(f"Perceptual-hash candidates: {len(pairs)}", flush=True)
    review_sheets(frame, channels, pairs, root, figures,
                  pd.read_csv(output / "quality_top30_unique_images.csv"))
    verify_sources(inventory, root)
    print("All raw hashes verified unchanged after analysis.", flush=True)
    channels.groupby([
        "target_label", "image_mode", "actual_channel_differences",
    ], dropna=False).size().rename("count").to_csv(
        output / "channel_class_counts.csv"
    )
    manifest_digest = hashlib.sha256(
        (output / "canonical_image_manifest.csv").read_bytes()
    ).hexdigest()
    summary = {
        "canonical_n": len(frame), "manifest_sha256": manifest_digest,
        "raw_hashes_checked_before_and_after": len(inventory),
        "raw_bytes_unchanged": True,
        "L_count": int(channels.image_mode.eq("L").sum()),
        "RGB_count": int(channels.image_mode.eq("RGB").sum()),
        "RGB_identical_channels": int(
            channels.rgb_channels_identical.eq(True).sum()
        ),
        "RGB_differing_channels": int(
            channels.actual_channel_differences.sum()
        ),
        "near_duplicate_pairs": len(pairs),
        "cross_class_pairs": int(pairs.cross_class.sum()),
        "cross_original_split_pairs": int(pairs.cross_original_split.sum()),
        "phash_distance_counts": pairs.phash_distance.value_counts().to_dict(),
        "phash_radius": 4, "mi_permutations": 499, "random_seed": 42,
        "quality_class_rates": (
            quality.set_index("target_class").flag_rate.to_dict()
        ),
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
            "pillow": PIL.__version__, "matplotlib": matplotlib.__version__,
        },
        "limitations": "Patient independence and candidates are unverified.",
    }
    (output / "final_phase1_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(associations[["variable", "effect_size", "p_value", "mi_nats"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed")
    )
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    run(args.dataset_root, args.output_dir, args.figures_dir)


if __name__ == "__main__":
    main()
