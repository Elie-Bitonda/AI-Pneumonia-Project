"""I execute train-only metadata diagnostics and preserve the raw evidence."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .phase2_grouping import load_manifest, constrain_groups, make_partitions, CLASS_INDEX, audit_constraint_graph
from .phase2_pipeline import NUMERIC_PREDICTORS, build_metadata_pipeline, load_canonical_image, pretrained_representation
from .preprocessing_summary import read_table, missingness
from .statistical_audit import mi_diagnostic
from .transformers import LABELS

FIGURES = (
    "fig1_1_missingness_heatmap.png", "fig1_2_target_distribution.png",
    "fig2_1_outlier_boxplots.png", "fig3_1_correlation_matrix.png",
    "fig3_2_collinear_scatterplots.png", "fig3_3_target_importance.png",
    "fig4_1_transformation_grid.png", "fig5_1_pca_scree_plot.png",
)
UNITS = {"width": "pixels", "height": "pixels", "aspect_ratio": "width/height",
         "file_size_bytes": "bytes", "mean_intensity": "intensity (0–255)",
         "std_intensity": "intensity SD (0–255)"}


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def raw_snapshot(root):
    """I fingerprint every raw file, including non-image packaging metadata."""
    return {p.relative_to(root).as_posix(): file_hash(p)
            for p in sorted(Path(root).rglob("*")) if p.is_file()}


def snapshot_fingerprint(snapshot):
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()


def verify_raw_inventory(root, inventory, snapshot):
    actual = {p for p in snapshot if Path(p).suffix.lower() in {".jpeg", ".jpg", ".png"}}
    if actual != set(inventory.filepath) or not inventory.filepath.is_unique:
        raise ValueError("Raw image membership differs from the Phase 1 inventory.")
    if any(snapshot[r.filepath] != r.sha256_hash for r in inventory.itertuples()):
        raise ValueError("Raw image bytes differ from Phase 1.")


def outlier_diagnostic(training):
    """I fit IQR fences and Z-score references on the actual training rows."""
    x = training[list(NUMERIC_PREDICTORS)].astype(float)
    q1, q3 = x.quantile(.25), x.quantile(.75)
    lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    capped = x.clip(lower=lower, upper=upper, axis=1)
    masks = x.lt(lower) | x.gt(upper)
    records = []
    for col in x:
        sd = x[col].std(ddof=0)
        z_count = int(((x[col] - x[col].mean()).abs() > 3 * sd).sum()) if sd else 0
        records.append({"variable": col, "training_n": len(x), "lower_fence": lower[col],
                        "upper_fence": upper[col], "iqr_flagged": int(masks[col].sum()),
                        "z_gt_3_sensitivity": z_count if col in {"mean_intensity", "std_intensity"} else None,
                        "skew_before": x[col].skew(), "skew_after_diagnostic_cap": capped[col].skew(),
                        "pixels_capped": False, "cap_used_in_pipeline": False})
    return pd.DataFrame(records), capped, int(masks.any(axis=1).sum())


def transformation_diagnostic(training):
    """I compare raw bytes, log1p(raw), MinMax(raw), and StandardScaler(raw).

    I fit the two scalers independently on training raw bytes. These panels are
    parallel alternatives, not a chain of fitted scalers. log1p is fixed.
    """
    raw = training[["file_size_bytes"]].to_numpy(float)
    if not np.isfinite(raw).all() or (raw <= 0).any():
        raise ValueError("Positive finite file sizes are required for this log diagnostic.")
    mm, ss = MinMaxScaler().fit(raw), StandardScaler().fit(raw)
    values = pd.DataFrame({"Raw": raw[:, 0], "Log1p(raw)": np.log1p(raw[:, 0]),
                           "MinMax(raw)": mm.transform(raw)[:, 0],
                           "Standardized(raw)": ss.transform(raw)[:, 0]})
    stats = pd.DataFrame({"panel": values.columns, "skewness": values.skew().to_numpy(),
                          "mean": values.mean().to_numpy(),
                          "population_sd": values.std(ddof=0).to_numpy(),
                          "minimum": values.min().to_numpy(), "maximum": values.max().to_numpy()})
    return values, stats


def pca_diagnostic(training, *, fitted_pipeline=None):
    """Compare candidate-space PCA with PCA on an already fitted pipeline output.

    The supplied pipeline is never refitted here. Both PCA spaces are fitted
    only on the active training rows; neither becomes a retained predictor stage.
    """
    if fitted_pipeline is None:
        pipeline = build_metadata_pipeline(reduce_collinearity=False)
        z = pipeline.fit_transform(training[list(NUMERIC_PREDICTORS)])
    else:
        pipeline = fitted_pipeline
        z = pipeline.transform(training[list(NUMERIC_PREDICTORS)])
    pca = PCA(svd_solver="full").fit(z)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    summary = pd.DataFrame({"component": np.arange(1, len(cumulative) + 1),
                            "explained_variance_ratio": pca.explained_variance_ratio_,
                            "cumulative_explained_variance": cumulative})
    counts = {str(t): int(np.searchsorted(cumulative + 1e-12, t) + 1) for t in (.9, .95)}
    return summary, counts, pipeline, pca


def summarize_associations(associations):
    """Summarize overlapping training subsets; SD is descriptive, not a CI."""
    if (associations.duplicated(["outer_fold", "variable"]).any()
            or not np.isfinite(associations.mi_nats).all()):
        raise ValueError("MI records must be unique and finite for each fold/variable.")
    for _, part in associations.groupby("variable"):
        if set(part.outer_fold) != set(range(1, 6)):
            raise ValueError("Five independently calculated training-fold MI values are required.")
    result = associations.groupby("variable").mi_nats.agg(
        mean_mi_nats="mean", sd_mi_nats="std", min_mi_nats="min", max_mi_nats="max")
    first = associations[associations.outer_fold.eq(1)].set_index("variable").mi_nats
    result["fold1_mi_nats"] = first
    result["fold_count"] = 5
    return result.reset_index().sort_values("mean_mi_nats", ascending=False).reset_index(drop=True)


def analyze_training(training):
    if training[list(NUMERIC_PREDICTORS)].isna().any().any():
        raise ValueError("Unexpected predictor missingness requires a documented decision.")
    pipe = build_metadata_pipeline()
    x = training[list(NUMERIC_PREDICTORS)]
    transformed = pipe.fit_transform(x, training.target_label)
    scaler = pipe.named_steps["columns"].named_transformers_["numeric"].named_steps["scale"]
    if int(scaler.n_samples_seen_) != len(training) or not np.allclose(scaler.mean_, x.mean()):
        raise AssertionError("Scaler was not fitted exclusively on the supplied training rows.")
    associations = pd.DataFrame([{"variable": c, **mi_diagnostic(training[c], training.target_label, True)}
                                 for c in NUMERIC_PREDICTORS])
    bounds, capped, union = outlier_diagnostic(training)
    transformations, transform_stats = transformation_diagnostic(training)
    candidates, candidate_counts, _, _ = pca_diagnostic(training)
    pca, counts, pca_pipeline, pca_model = pca_diagnostic(training, fitted_pipeline=pipe)
    return {"pipeline": pipe, "input_features": len(NUMERIC_PREDICTORS),
            "output_features": transformed.shape[1], "associations": associations,
            "correlation": x.corr(method="spearman"), "outliers": bounds, "capped": capped,
            "outlier_union": union, "transformations": transformations, "transform_stats": transform_stats,
            "pca": pca, "pca_counts": counts, "pca_pipeline": pca_pipeline, "pca_model": pca_model,
            "pca_candidates": candidates, "pca_candidate_counts": candidate_counts,
            "fit_rows": len(training), "scaler_mean": scaler.mean_.tolist()}


def missingness_categories(canonical):
    """I distinguish optional error blanks, absent identity, and measurements."""
    cols = list(NUMERIC_PREDICTORS) + ["target_label", "patient_id_if_parseable", "read_error", "hash_error"]
    values = np.zeros((len(canonical), len(cols)), dtype=np.uint8)
    for j, col in enumerate(cols):
        code = 2 if col in {"read_error", "hash_error"} else 3 if col == "patient_id_if_parseable" else 1
        values[canonical[col].isna().to_numpy(), j] = code
    return cols, values


def generate_figures(canonical, training, diagnostics, directory):
    """Plot canonical descriptive views, fold-1 diagnostics, and across-fold MI."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
                         "figure.dpi": 120, "savefig.dpi": 300})

    def save(fig, name, caption):
        footer = .28 if name == FIGURES[0] else .12
        fig.set_layout_engine("constrained", rect=(0, footer, 1, 1-footer))
        fig.text(.5, .02, caption, ha="center", va="bottom", fontsize=9)
        fig.savefig(directory / name, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    cols, values = missingness_categories(canonical)
    fig, ax = plt.subplots(figsize=(12, 7), layout="constrained")
    colors = ["#e7eef2", "#d73027", "#4575b4", "#fdae61"]
    ax.imshow(values, aspect="auto", interpolation="nearest", cmap=ListedColormap(colors),
              norm=BoundaryNorm(np.arange(-.5, 4), 4))
    ax.set_xticks(range(len(cols)), [c.replace("_", "\n") for c in cols], fontsize=8)
    ax.set(ylabel="Canonical rows (SHA-256 order)", title="Canonical missingness: predictor completeness and structural blanks")
    fig.legend(handles=[Patch(color=c, label=t) for c, t in zip(colors,
              ["Observed", "Missing measurement", "Structural error-field blank", "Unavailable identity token"])],
              loc="lower center", bbox_to_anchor=(.5, .07), ncol=2, fontsize=9)
    save(fig, FIGURES[0], "I observed complete selected predictors; I did not impute structural blanks or identity tokens.")

    counts = canonical.target_label.value_counts().reindex(LABELS)
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    bars = ax.bar(["Normal", "Bacterial pneumonia", "Viral pneumonia"], counts, color=["#326b8e", "#db9345", "#7c6ba6"])
    ax.bar_label(bars, labels=[f"{n:,}\n{100*n/len(canonical):.2f}%" for n in counts], padding=4)
    ax.set(ylabel="Canonical unique byte contents", title=f"Canonical target distribution (N = {len(canonical):,})", ylim=(0, counts.max()*1.22))
    save(fig, FIGURES[1], "I counted canonical SHA-256 contents, not verified independent patients.")

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    for col, ax in zip(NUMERIC_PREDICTORS, axes.flat):
        ax.boxplot([training[col].to_numpy(), diagnostics["capped"][col].to_numpy()],
                   tick_labels=["Raw", "Diagnostic cap"], flierprops={"markersize": 2})
        ax.set(title=col, ylabel=UNITS[col])
    fig.suptitle("Fold 1 training metadata: before/after training-fitted IQR capping")
    save(fig, FIGURES[2], "Metadata comparison only: no radiographs deleted or CNN pixels capped. The diagnostic pipeline remains uncapped.")

    corr = diagnostics["correlation"]
    fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
    masked = np.ma.array(corr.to_numpy(), mask=np.tril(np.ones(corr.shape), k=-1).astype(bool))
    cmap = plt.get_cmap("RdBu_r").with_extremes(bad="white")
    im = ax.imshow(masked, vmin=-1, vmax=1, cmap=cmap)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(i, len(corr)):
            ax.text(j, i, f"{corr.iloc[i,j]:.3f}", ha="center", va="center",
                    color="white" if abs(corr.iloc[i,j])>.7 else "black")
    ax.set_title("Fold 1 training metadata: upper-triangle Spearman correlation")
    fig.colorbar(im, ax=ax, label="Spearman rho", shrink=.75)
    save(fig, FIGURES[3], "I interpret rank associations as diagnostic redundancy, not clinical or causal evidence.")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), layout="constrained")
    pairs = [("width", "height"), ("width", "file_size_bytes"), ("height", "file_size_bytes")]
    for (a,b), ax in zip(pairs, axes):
        ax.scatter(training[a], training[b], s=6, alpha=.22, color="#326b8e", rasterized=True)
        ax.set(xlabel=f"{a} ({UNITS[a]})", ylabel=f"{b} ({UNITS[b]})",
               title=f"Spearman rho = {corr.loc[a,b]:.3f}")
    fig.suptitle("Fold 1 training metadata: resolution and storage relationships")
    save(fig, FIGURES[4], "I retained every radiograph; these relationships do not justify removing image pixels.")

    assoc = diagnostics["association_across_folds"].sort_values("mean_mi_nats")
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    bars = ax.barh(assoc.variable, assoc.mean_mi_nats, xerr=assoc.sd_mi_nats,
                   capsize=4, color="#326b8e")
    ax.bar_label(bars, fmt="%.4f", padding=12)
    ax.set(xlabel="Ten-quantile-bin MI (nats); mean ± sample SD across five training subsets",
           title="Training-only metadata–target association across five outer folds",
           xlim=(0, (assoc.mean_mi_nats + assoc.sd_mi_nats).max()*1.2))
    save(fig, FIGURES[5], "Training subsets overlap: SD bars are descriptive, not confidence intervals. Association is not causation.")

    fig, axes = plt.subplots(1, 4, figsize=(15, 5), layout="constrained")
    for (col, values), ax in zip(diagnostics["transformations"].items(), axes):
        ax.hist(values, bins=40, color="#326b8e", alpha=.85)
        ax.set(title=f"{col}\nskew = {values.skew():.3f}", xlabel=col + (" (bytes)" if col=="Raw" else ""), ylabel="Training images")
    fig.suptitle("Fold 1 training file size: Raw → Log → MinMax → Standardized comparison")
    save(fig, FIGURES[6], "I compared log1p(raw), MinMax(raw), and StandardScaler(raw) independently; the arrows indicate panel order.")

    pca = diagnostics["pca"]
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    ax.bar(pca.component, pca.explained_variance_ratio, label="Individual variance", color="#326b8e", alpha=.75)
    ax.plot(pca.component, pca.cumulative_explained_variance, "o-", color="#b85824", label="Cumulative variance")
    for threshold, color in [(.9,"#638743"),(.95,"#8061a3")]:
        ax.axhline(threshold, color=color, linestyle="--", label=f"{threshold:.0%} threshold")
    ax.set(xticks=pca.component, xlabel="Principal component", ylabel="Explained variance fraction", ylim=(0,1.06),
           title=f"Fold 1 training PCA: retained metadata ({len(pca)} variables; post-collinearity)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, .5), fontsize=8)
    save(fig, FIGURES[7], "PCA describes the retained diagnostic pipeline output; it is not fitted to CNN pixels or retained automatically.")
    return [str(directory / name) for name in FIGURES]


def preserve_phase1_figure(project):
    """I preserve the committed figure once and verify the backup on reruns."""
    import subprocess
    project = Path(project)
    relative = "figures/fig3_1_correlation_matrix.png"
    original = subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=project)
    backup = project / "figures/phase1_fig3_1_correlation_matrix.png"
    if backup.exists():
        if backup.read_bytes() != original:
            raise ValueError("Phase 1 figure backup differs from the committed evidence.")
    else:
        if (project / relative).read_bytes() != original:
            raise ValueError("Existing correlation figure has unreviewed changes.")
        shutil.copyfile(project / relative, backup)
    return hashlib.sha256(original).hexdigest()


def verify_phase1_preservation(project):
    """Check committed contents and retain byte fingerprints of local evidence.

    Git's Windows checkout uses CRLF while repository blobs use LF. I compare
    text contents with newline normalization and separately fingerprint actual
    working bytes before/after execution; no evidence is rewritten.
    """
    import subprocess
    project = Path(project)
    paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", "HEAD", "--",
                                     "data/processed", "phase1_strategy", "src", "tests", "figures"],
                                    cwd=project, text=True).splitlines()
    paths = [name for name in paths if name != "figures/fig3_1_correlation_matrix.png"]
    fingerprints = {}
    newline_only = 0
    for name in paths:
        current = (project / name).read_bytes()
        committed = subprocess.check_output(["git", "show", "HEAD:" + name], cwd=project)
        if current != committed:
            text = Path(name).suffix in {".csv", ".json", ".md", ".py"}
            if not text or current.replace(b"\r\n", b"\n") != committed.replace(b"\r\n", b"\n"):
                raise AssertionError(f"Authoritative Phase 1 contents changed: {name}")
            newline_only += 1
        fingerprints[name] = hashlib.sha256(current).hexdigest()
    old_requirements = subprocess.check_output(["git", "show", "HEAD:requirements.txt"], cwd=project).decode().splitlines()
    current_requirements = (project / "requirements.txt").read_text().splitlines()
    if current_requirements[:len(old_requirements)] != old_requirements:
        raise AssertionError("Existing Phase 1 dependency pins changed.")
    return {"protected_files_checked": len(paths), "committed_contents_match": True,
            "checkout_newline_only_differences": newline_only,
            "comparison": "Committed text contents normalized for CRLF/LF; working byte hashes checked before/after execution.",
            "working_file_sha256": fingerprints, "phase1_dependency_pins_preserved": True,
            "other_phase1_image_figures_byte_identical": True}


def excluded_columns(canonical):
    reasons = {}
    for col in canonical.columns:
        if col in NUMERIC_PREDICTORS:
            continue
        if col == "target_label":
            reason = "I use this nominal label as Y only."
        elif col in {"image_mode", "channel_count", "intensity_basis"}:
            reason = "I omit redundant storage/channel encodings after the equal-channel audit."
        elif col in {"min_intensity", "max_intensity"}:
            reason = "I retain the six Phase 1 diagnostic candidates rather than adding extrema."
        else:
            reason = "I exclude identifiers, provenance, parsing/quality status, alternative annotations, and structural error fields."
        reasons[col] = reason
    return reasons


def write_decisions(project, summary):
    first = summary["fold_diagnostics"][0]
    rows = [
        ("One canonical path per readable SHA-256", f"I verified {summary['canonical_n']} rows and unchanged manifest bytes.", "I retained all canonical contents for this checkpoint.", "I found no new adjudicated exclusion evidence.", "Membership is reproducible; patient independence remains unverified."),
        ("Filename-derived relation groups", f"I reproduced {summary['base_relation_groups']} base groups.", "I grouped pneumonia tokens across subtypes and kept Normal filename families distinct.", "I followed the documented relation policy without asserting patient identity.", "I retained the Phase 1 base structure."),
        ("Unresolved pHash edges stay within roles", f"I found {summary['unresolved_phash_edges']} unresolved edges and {summary['leakage_control_groups']} constraint components.", "I retained every unresolved edge as a split-only constraint.", "I have no definitive adjudication establishing duplicates or rejected edges.", "The partition group count differs from the filename-only count by design; I did not merge image contents."),
        ("Five group-aware outer folds; approximately 72/8/20 roles", "I recorded all actual role/class/group counts in phase2_fold_summary.csv.", "I used seed 42 for outer folds and seed 42+fold for the first inner ten-fold partition.", "I prioritized component integrity and did not search favorable seeds.", "Exact shares vary; all roles and candidate endpoints remain isolated."),
        ("224×224 aspect-preserving common intensity input", "I verified deterministic float32 [0,1] outputs for one L and one RGB canonical image.", "I implemented antialiased bilinear resize, black symmetric padding, and bottom/right odd pixels.", "I preserved stored geometry and equal-channel intensity.", "Raw images remain read-only; inputs are produced on demand."),
        ("Conservative training-only augmentation", "I tested bounded seeded rotation/translation; I did not train models.", "I configured ±5° and up to 2% translation with explicit sample/epoch seeds.", "I avoided another augmentation dependency.", "Validation/evaluation reject augmentation requests."),
        ("No predictor imputation or pixel capping", f"I observed {summary['missingness']['selected_predictor_missing_cells']} missing predictor cells.", "I retained structural blanks and used IQR caps for metadata comparison only.", "Optional error fields and unverified IDs are not predictors.", "No imputer, automatic exclusion, or pixel Winsorization is used."),
        ("Train-derived normalization and class weights", "I independently fitted five training-only pipelines and computed five sets of class weights.", "I calculate N_train/(3*n_class) for each active training subset.", "I isolate fitted state from internal validation and outer evaluation.", "Each fold records its training-hash fingerprint and scaler sample count."),
        ("Structural redundancy first, then rank-VIF >10", f"I removed {first['removed_columns']} in fold 1; I recorded all fold-specific decisions.", "I implemented the deterministic aspect-ratio-first and rank-VIF/MI tie rule.", "I used training-only metadata and retained raw/rank VIF evidence.", "Only the metadata diagnostic feature space changes; no spatial pixels are removed."),
        ("Skew/transformation diagnostics", f"Fold-1 raw/log skew is {first['raw_file_size_skew']:.4f}/{first['log_file_size_skew']:.4f}.", "I compare raw/log1p/MinMax/standardized representations independently.", "Affine scaling changes units and spread, not skew.", "Log and capping remain diagnostic comparisons rather than retained pipeline stages."),
        ("PCA diagnostics on standardized metadata", f"Fold-1 candidate-space PCA reaches 90%/95% at {first['pca_candidate_90_components']}/{first['pca_candidate_95_components']} components; retained-space PCA reaches them at {first['pca_90_components']}/{first['pca_95_components']}.", "I selected post-collinearity PCA for the required figure and retained both spaces in phase2_pca_comparison.csv.", "The four-variable PCA describes the actual retained diagnostic pipeline output; selection was based on pipeline alignment, not the variance result.", "The required PCA figure changed scope; PCA remains exploratory and is not added to the predictor pipeline."),
        ("Training-only feature/target association", "All five training-fold MI calculations are summarized in phase2_associations_across_folds.csv.", "I replaced the representative fold-1 figure with the across-fold mean and sample SD.", "A five-subset descriptive summary exposes stability; overlapping training sets preclude interpreting SD as independent uncertainty or a confidence interval.", "Only the presentation changes; no full-cohort MI fitting or causal inference is introduced."),
        ("Preserve Phase 1 evidence", "I found an exact required-filename collision for the correlation figure.", "I preserved the committed bytes as figures/phase1_fig3_1_correlation_matrix.png.", "I needed the exact Phase 2 deliverable filename.", "Both the historical and current correlation figures remain available."),
        ("Reproduce the environment", f"Phase 1 recorded Python 3.14.7; Phase 2 executed with Python {summary['versions']['python']}.", "I retained the Phase 1 dependency pins and tests, and added four necessary direct requirements.", "The restored environment uses a different interpreter; I record actual versions rather than claiming interpreter equivalence.", "The runtime deviation is explicit; complete test and dependency checks accompany execution."),
    ]
    text = "# Phase 2 checkpoint decisions\n\nI implemented two separate branches: image preprocessing for future CNN inputs and metadata diagnostics for acquisition/shortcut investigation. I did not train a CNN, finalize patient identities, or write a final Phase 2 report.\n\n"
    categories = {"One canonical path per readable SHA-256": "Confirmed",
                  "Filename-derived relation groups": "Confirmed",
                  "224×224 aspect-preserving common intensity input": "Confirmed",
                  "No predictor imputation or pixel capping": "Confirmed",
                  "PCA diagnostics on standardized metadata": "Phase 2 methodological refinement",
                  "Training-only feature/target association": "Phase 2 presentation refinement",
                  "Reproduce the environment": "Genuine runtime deviation"}
    text += "| Classification | Phase 1 plan | Phase 2 observed evidence | Implemented decision | Technical reason | Impact |\n|---|---|---|---|---|---|\n"
    text += "\n".join("| " + " | ".join(str(c).replace("|", "\\|") for c in (categories.get(row[0], "Operationalized"), *row)) + " |" for row in rows)
    text += "\n\nI treat these assignments as the reproducible checkpoint partition under the all-unresolved-edge constraint policy. I will version any later adjudication and regenerate partitions before model fitting. I do not claim patient-independent evaluation. The canonical membership is retained, while clinical and identity uncertainty remain unresolved.\n\n"
    text += "I used the documented scikit-learn [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html) and [Pipeline/ColumnTransformer](https://scikit-learn.org/stable/modules/compose.html) interfaces.\n"
    (Path(project) / "phase2_implementation/phase2_decision_log.md").write_text(text, encoding="utf-8")


def run_checkpoint(project):
    """I execute the checkpoint without fitting a predictive model or saving pixels."""
    project = Path(project).resolve()
    root, output, figures = project / "chest_xray", project / "data/processed", project / "figures"
    if not root.is_dir():
        raise ValueError("Verified dataset root is absent.")
    for dest in (output, figures):
        if dest.resolve().is_relative_to(root.resolve()) or root.resolve().is_relative_to(dest.resolve()):
            raise ValueError("Outputs must be disjoint from raw images.")
    print("I am fingerprinting every raw file before checkpoint execution.", flush=True)
    preservation = verify_phase1_preservation(project)
    before = raw_snapshot(root)
    inventory = read_table(output / "image_metadata_audit.csv")
    verify_raw_inventory(root, inventory, before)
    evidence = {p.name: file_hash(p) for p in output.iterdir() if p.is_file() and not p.name.startswith("phase2_")}
    canonical = load_manifest(project)
    manifest_hash = file_hash(output / "canonical_image_manifest.csv")
    expected_hash = json.loads((output / "final_phase1_summary.json").read_text())["manifest_sha256"]
    if manifest_hash != expected_hash:
        raise ValueError("Canonical manifest fingerprint changed.")
    candidates = read_table(output / "near_duplicate_candidates.csv")
    if len(candidates) != 223:
        raise ValueError("Phase 1 pHash candidate membership changed.")
    grouped = constrain_groups(canonical, candidates)
    if grouped.base_relation_group.nunique() != 3117:
        raise ValueError("Filename base does not reproduce Phase 1.")
    assignments, folds, weights, checks = make_partitions(grouped, candidates)
    graph_audit = audit_constraint_graph(grouped, candidates)
    counterfactual, _, _, _ = make_partitions(grouped.assign(original_split="IGNORED_PROVENANCE"), candidates)
    pd.testing.assert_frame_equal(assignments, counterfactual)
    forbidden = {"target_label", "filepath", "sha256_hash", "filename", "base_relation_group",
                 "leakage_control_group", "original_split", "patient_id_if_parseable"}
    if set(NUMERIC_PREDICTORS) & forbidden:
        raise AssertionError("A forbidden identifier/target entered the predictor allowlist.")
    assignments.to_csv(output / "phase2_fold_assignments.csv", index=False)
    folds.to_csv(output / "phase2_fold_summary.csv", index=False)
    weights.to_csv(output / "phase2_class_weights.csv", index=False)
    grouped[["sha256_hash", "filepath", "base_relation_group", "leakage_control_group"]].to_csv(output / "phase2_grouping_manifest.csv", index=False)
    edges = candidates[["sha256_1", "sha256_2", "phash_distance", "review_status"]].copy()
    edges["handling"] = "UNRESOLVED_SPLIT_CONSTRAINT_ONLY"
    edges.to_csv(output / "phase2_phash_constraints.csv", index=False)
    fold_results, combined = [], {k: [] for k in ["associations", "outliers", "transform_stats", "pca", "vif", "correlation"]}
    pca_comparison = []
    first_training = first_diagnostics = None
    for fold in range(1, 6):
        keys = assignments[(assignments.outer_fold == fold) & (assignments.role == "train")].sha256_hash
        training = grouped[grouped.sha256_hash.isin(keys)].copy()
        d = analyze_training(training)
        reducer = d["pipeline"].named_steps["rank_vif"]
        for name in ["associations", "outliers", "transform_stats", "pca"]:
            table = d[name].copy(); table.insert(0, "outer_fold", fold); combined[name].append(table)
        for scope, table, counts in [("six_candidate_variables_pre_collinearity", d["pca_candidates"], d["pca_candidate_counts"]),
                                     ("retained_pipeline_output_post_collinearity", d["pca"], d["pca_counts"])]:
            comparison = table.copy()
            comparison.insert(0, "outer_fold", fold)
            comparison.insert(1, "scope", scope)
            comparison["feature_count"] = len(table)
            comparison["components_90"] = counts["0.9"]
            comparison["components_95"] = counts["0.95"]
            pca_comparison.append(comparison)
        for kind, vif in [("raw_initial", reducer.raw_vif_), ("rank_initial", reducer.initial_rank_vif_),
                          ("rank_final", reducer.final_rank_vif_)]:
            table = vif.copy(); table.insert(0, "outer_fold", fold); table.insert(1, "stage", kind); combined["vif"].append(table)
        corr = d["correlation"].rename_axis("variable_1").reset_index().melt(id_vars="variable_1", var_name="variable_2", value_name="spearman_rho")
        corr.insert(0, "outer_fold", fold); combined["correlation"].append(corr)
        fold_results.append({"outer_fold": fold, "fit_rows": len(training),
            "training_hash_fingerprint": hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest(),
            "input_features": d["input_features"], "output_features": d["output_features"],
            "retained_columns": reducer.selected_columns_, "removed_columns": reducer.removals_,
            "scaler_mean": d["scaler_mean"], "scaler_fit_count_verified": True,
            "iqr_flagged_training_images": d["outlier_union"],
            "raw_file_size_skew": float(d["transformations"]["Raw"].skew()),
            "log_file_size_skew": float(d["transformations"]["Log1p(raw)"].skew()),
            "pca_90_components": d["pca_counts"]["0.9"], "pca_95_components": d["pca_counts"]["0.95"]})
        fold_results[-1].update({"pca_candidate_90_components": d["pca_candidate_counts"]["0.9"],
                                 "pca_candidate_95_components": d["pca_candidate_counts"]["0.95"]})
        if fold == 1:
            first_training, first_diagnostics = training, d
        print(f"I fitted fold {fold} diagnostics on {len(training)} actual training rows.", flush=True)
    for name, tables in combined.items():
        pd.concat(tables, ignore_index=True).to_csv(output / f"phase2_{name}_summary.csv", index=False)
    across_folds = summarize_associations(pd.concat(combined["associations"], ignore_index=True))
    across_folds.to_csv(output / "phase2_associations_across_folds.csv", index=False)
    first_diagnostics["association_across_folds"] = across_folds
    pd.concat(pca_comparison, ignore_index=True).to_csv(output / "phase2_pca_comparison.csv", index=False)
    pd.concat([missingness(inventory, "original_audit"), missingness(canonical, "canonical_manifest")], ignore_index=True).to_csv(output / "phase2_missingness_summary.csv", index=False)
    backup_hash = preserve_phase1_figure(project)
    figure_paths = generate_figures(canonical, first_training, first_diagnostics, figures)
    image_results = []
    for mode in ["L", "RGB"]:
        row = canonical[canonical.image_mode == mode].iloc[0]
        a = load_canonical_image(root, row)
        b = load_canonical_image(root, row)
        rgb = pretrained_representation(a)
        assert a.shape == (224,224) and a.dtype == np.float32 and 0 <= a.min() <= a.max() <= 1
        assert np.array_equal(a,b) and np.array_equal(rgb[0],rgb[1]) and np.array_equal(rgb[1],rgb[2])
        image_results.append({"mode": mode, "shape": list(a.shape), "dtype": str(a.dtype),
                              "minimum": float(a.min()), "maximum": float(a.max()), "deterministic": True,
                              "replicated_channels_equal": True})
    print("I am verifying that every raw file and Phase 1 table is unchanged.", flush=True)
    after = raw_snapshot(root)
    if before != after:
        raise AssertionError("Raw dataset changed during checkpoint execution.")
    if any(file_hash(output / name) != h for name, h in evidence.items()):
        raise AssertionError("Authoritative Phase 1 processed evidence changed.")
    if any(file_hash(project / name) != h for name,h in preservation["working_file_sha256"].items()):
        raise AssertionError("Phase 1 working evidence bytes changed during execution.")
    preservation["working_bytes_unchanged_during_execution"] = True
    checks.update({"train_only_scaler_fits": True, "raw_dataset_unchanged": True, "phase1_tables_unchanged": True})
    summary = {"canonical_n": len(canonical), "class_counts": canonical.target_label.value_counts().to_dict(),
        "base_relation_groups": int(grouped.base_relation_group.nunique()),
        "constraint_graph_audit": graph_audit,
        "leakage_control_groups": int(grouped.leakage_control_group.nunique()),
        "largest_constraint_component_images": int(grouped.leakage_control_group.value_counts().max()),
        "unresolved_phash_edges": len(candidates), "phash_policy": "all unresolved edges are split-only constraints; no identity/duplicate adjudication",
        "class_index": CLASS_INDEX, "outer_seed": 42, "inner_seed": "42 + outer_fold; first of ten inner folds",
        "predictors": list(NUMERIC_PREDICTORS), "excluded_columns": excluded_columns(canonical),
        "missingness": {"selected_predictor_missing_cells": int(canonical[list(NUMERIC_PREDICTORS)].isna().sum().sum()),
                        "canonical_missing_cells": int(canonical.isna().sum().sum()),
                        "original_missing_cells": int(inventory.isna().sum().sum()),
                        "structural_error_blanks": int(canonical[["read_error", "hash_error"]].isna().sum().sum()),
                        "unavailable_candidate_ids": int(canonical.patient_id_if_parseable.isna().sum())},
        "fold_diagnostics": fold_results, "leakage_assertions": checks, "image_checks": image_results,
        "raw_file_count": len(before), "raw_fingerprint_before": snapshot_fingerprint(before),
        "raw_fingerprint_after": snapshot_fingerprint(after), "manifest_sha256": manifest_hash,
        "phase1_figure_backup_sha256": backup_hash, "figure_paths": figure_paths,
        "versions": {"python": platform.python_version(), **{p: importlib.metadata.version(p) for p in ["numpy","pandas","Pillow","matplotlib","scipy","scikit-learn","nbformat","nbclient","ipykernel"]}},
        "limitations": "I retained uncertain relations as constraints; I do not claim verified patient independence, clinical causation, or model performance.",
        "pca_scope": "post-collinearity retained pipeline output; fold-1 required figure; six-candidate comparison saved separately; exploratory only",
        "mi_figure_scope": "mean and sample SD of independently recomputed MI in five overlapping actual training subsets; descriptive variability, not confidence intervals",
        "metadata_pipeline": "explicit numeric ColumnTransformer/StandardScaler followed by training rank-VIF reduction; no imputation, capping, log transform, categorical predictors, or PCA retained",
        "augmentation": {"rotation_degrees": 5, "translation_fraction": .02, "role": "train only", "seed_policy": "explicit sample/epoch seed", "model_training_performed": False}}
    (output / "phase2_pipeline_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    research_audit = {"phase1_preservation": preservation, "constraint_graph": graph_audit,
                      "leakage_assertions": checks, "source_split_counterfactual_identical": True,
                      "predictor_allowlist_excludes_target_identifiers": True,
                      "preserved_phase1_correlation_sha256": file_hash(figures / "phase1_fig3_1_correlation_matrix.png"),
                      "current_phase2_correlation_sha256": file_hash(figures / "fig3_1_correlation_matrix.png"),
                      "pca_decision": summary["pca_scope"], "mi_decision": summary["mi_figure_scope"]}
    (output / "phase2_research_audit.json").write_text(json.dumps(research_audit, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    write_decisions(project, summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    print(json.dumps(run_checkpoint(parser.parse_args().project_root), indent=2))
