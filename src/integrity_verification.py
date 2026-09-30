"""Verify Phase 1 integrity from local files; never assign folds or fit models.

Usage: python -m src.integrity_verification --dataset-root chest_xray
The earlier metadata is reused only after file membership and every SHA-256
are checked again. Sidecar signatures are read directly from the source.
"""

import argparse
import hashlib
import itertools
import json
import re
import struct
import zipfile
from pathlib import Path

import pandas as pd

from .transformers import (
    EXTENSIONS, LABELS, METRICS, is_filesystem_metadata, parse_identity,
)


def json_list(values):
    """Stable JSON arrays in CSV cells, without lossy delimiter joining."""
    return json.dumps(sorted(set(values)), ensure_ascii=False)


def source_location(filepath):
    """Separate source-tree prefix from split/class/filename suffix."""
    parts = Path(filepath).parts
    for i, part in enumerate(parts):
        if part.lower() in {"train", "val", "test"}:
            return "/".join(parts[:i]) or ".", "/".join(parts[i:])
    return "UNKNOWN", filepath


def verify_sources(frame, root):
    """Reject stale inventory, changed bytes or changed label parsing."""
    actual = {
        p.relative_to(root).as_posix() for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    }
    if not frame.filepath.is_unique or actual != set(frame.filepath):
        raise ValueError("Inventory changed. Rerun the full Phase 1 audit.")
    for row in frame.itertuples():
        path = (root / row.filepath).resolve()
        if root not in path.parents:
            raise ValueError("Inventory path resolves outside the raw root.")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != row.sha256_hash:
            raise ValueError(f"Changed source bytes: {row.filepath}")
        parsed = parse_identity(row.filepath)
        label = None if pd.isna(row.target_label) else row.target_label
        if parsed["target_label"] != label:
            raise ValueError(f"Changed label parsing: {row.filepath}")


def build_groups(frame):
    """Return overlapping A/B/C/D classifications and a safe manifest.

    D requires same split/class/filename suffix and hash in distinct trees.
    Priority: shortest relative path depth, then case-sensitive lexical path.
    Original split and class never influence selection priority.
    """
    groups, canonical, exclusions = [], [], []
    for digest, group in frame.groupby("sha256_hash", sort=True):
        labels = set(group.target_label.dropna())
        splits = set(group.original_split) & {"train", "val", "test"}
        conflict = len(labels) > 1
        locations = [source_location(p) for p in group.filepath]
        suffix_trees = {}
        for tree, suffix in locations:
            suffix_trees.setdefault(suffix, set()).add(tree)
        directory_copy = any(len(v) > 1 for v in suffix_trees.values())
        codes = []
        if len(labels) == 1:
            codes.append("A")
        if len(splits) > 1:
            codes.append("B")
        if conflict:
            codes.append("C")
        if directory_copy:
            codes.append("D")
        alternatives = {
            "all_filepaths": json_list(group.filepath),
            "all_original_splits": json_list(group.original_split),
            "all_target_labels": json_list(group.target_label.dropna()),
            "all_filenames": json_list(group.filename),
        }
        if len(group) > 1:
            groups.append({
                "sha256_hash": digest, "duplicate_group_size": len(group),
                "excess_duplicate_files": len(group) - 1, **alternatives,
                "classification_codes": "|".join(codes),
                "same_label": len(labels) == 1,
                "across_original_splits": len(splits) > 1,
                "conflicting_labels": conflict,
                "duplicated_dataset_directories": directory_copy,
                "readable_file_count": int(group.readable.sum()),
                "filesystem_metadata_count": int(
                    group.filepath.map(is_filesystem_metadata).sum()
                ),
            })
        eligible = group[
            group.readable
            & ~group.filepath.map(is_filesystem_metadata)
            & (group.label_parse_status == "OK")
            & group.target_label.isin(LABELS)
        ]
        status = None
        if conflict:
            status = "CONFLICT_REQUIRES_REVIEW"
        elif group.target_label.isna().any():
            status = "UNRESOLVED_LABEL_REQUIRES_REVIEW"
        elif eligible.empty:
            status = "NO_READABLE_MODELING_IMAGE"
        if status:
            exclusions.append({
                "sha256_hash": digest, "status": status, **alternatives,
            })
            continue
        selected = min(
            eligible.filepath,
            key=lambda p: (len(Path(p).parts), p),
        )
        row = eligible.loc[eligible.filepath == selected].iloc[0].to_dict()
        row.update(alternatives)
        row.update(
            source_file_count=len(group),
            alternative_filepaths=json_list(
                p for p in group.filepath if p != selected
            ),
            manifest_status="PROVISIONAL_CONTENT_DEDUPLICATED",
            patient_grouping_validated=False,
        )
        canonical.append(row)
    return (
        pd.DataFrame(groups), pd.DataFrame(canonical),
        pd.DataFrame(exclusions),
    )


def directory_inventory(frame, root):
    """Recursive counts include all file types; directory rows overlap."""
    directories = [root] + sorted(p for p in root.rglob("*") if p.is_dir())
    all_files = [p.relative_to(root).as_posix()
                 for p in root.rglob("*") if p.is_file()]
    rows = []
    for directory in directories:
        relative = directory.relative_to(root).as_posix()
        prefix = "" if relative == "." else relative + "/"
        subset = frame[frame.filepath.str.startswith(prefix)]
        files = [p for p in all_files if p.startswith(prefix)]
        rows.append({
            "directory_path": relative, "count_scope": "recursive",
            "hash_and_class_scope": "image_extension_files_only",
            "number_of_files": len(files),
            "number_of_image_extension_files": len(subset),
            "number_of_readable_images": int(subset.readable.sum()),
            "number_of_sidecars": sum(map(is_filesystem_metadata, files)),
            "class_counts": json.dumps({
                k: int((subset.target_label == k).sum()) for k in LABELS
            }),
            "unique_sha256_count": int(subset.sha256_hash.nunique()),
            "unique_readable_sha256_count": int(
                subset.loc[subset.readable, "sha256_hash"].nunique()
            ),
        })
    return pd.DataFrame(rows)


def compare_trees(frame):
    """Pair corresponding readable suffixes; counts alone are not proof."""
    readable = frame[frame.readable].copy()
    locations = readable.filepath.map(source_location)
    readable["tree"] = locations.map(lambda item: item[0])
    readable["suffix"] = locations.map(lambda item: item[1])
    rows = []
    trees = sorted(readable.tree.unique())
    for left, right in itertools.combinations(trees, 2):
        a = readable[readable.tree == left].set_index("suffix")
        b = readable[readable.tree == right].set_index("suffix")
        for suffix in sorted(set(a.index) | set(b.index)):
            arow = a.loc[suffix] if suffix in a.index else None
            brow = b.loc[suffix] if suffix in b.index else None
            matched = arow is not None and brow is not None
            rows.append({
                "tree_a": left, "tree_b": right, "relative_suffix": suffix,
                "filepath_a": arow.filepath if arow is not None else None,
                "filepath_b": brow.filepath if brow is not None else None,
                "sha256_a": arow.sha256_hash if arow is not None else None,
                "sha256_b": brow.sha256_hash if brow is not None else None,
                "same_hash": matched and arow.sha256_hash == brow.sha256_hash,
                "same_label": (
                    matched and arow.target_label == brow.target_label
                ),
            })
    return pd.DataFrame(rows)


def inspect_sidecars(frame, root):
    """Check AppleDouble magic/version and bounded entry descriptors."""
    rows = []
    for row in frame[frame.filepath.map(is_filesystem_metadata)].itertuples():
        data = (root / row.filepath).read_bytes()
        magic = data[:4].hex()
        version = data[4:8].hex()
        entry_count = (
            struct.unpack(">H", data[24:26])[0] if len(data) >= 26 else 0
        )
        valid = (
            magic == "00051607" and version == "00020000"
            and entry_count > 0 and len(data) >= 26 + 12 * entry_count
        )
        if valid:
            for i in range(entry_count):
                _, offset, size = struct.unpack_from(">III", data, 26 + i * 12)
                valid = valid and offset + size <= len(data)
        rows.append({
            "filepath": row.filepath, "filename": row.filename,
            "readable": row.readable, "magic_hex": magic,
            "version_hex": version, "entry_count": entry_count,
            "appledouble_header_verified": valid,
            "classification": (
                "FILESYSTEM_METADATA" if valid else "SIDECAR_REQUIRES_REVIEW"
            ),
        })
    return pd.DataFrame(rows)


def patient_semantics(canonical):
    """Emit all canonical names and >=20 examples per class."""
    rows = []
    for row in canonical.itertuples():
        stem = Path(row.filename).stem
        if re.fullmatch(r"person\d+_bacteria_\d+", stem):
            convention = "person<number>_bacteria_<image>"
        elif re.fullmatch(r"person\d+_virus_\d+", stem):
            convention = "person<number>_virus_<image>"
        elif re.fullmatch(r"NORMAL2-IM-\d+-\d+(?:-\d+)?", stem):
            convention = "NORMAL2-IM-<number>-<number>[-<number>]"
        elif re.fullmatch(r"IM-\d+-\d+(?:-\d+)?", stem):
            convention = "IM-<number>-<number>[-<number>]"
        else:
            convention = "OTHER_REQUIRES_REVIEW"
        rows.append({
            "filename": row.filename,
            "parsed_candidate_id": row.patient_id_if_parseable,
            "class": row.target_label, "naming_convention": convention,
            "filepath": row.filepath, "sha256_hash": row.sha256_hash,
            "identity_proven": False, "representative_example": False,
        })
    result = pd.DataFrame(rows)
    for _, group in result.groupby("class"):
        chosen = []
        families = list(group.groupby("naming_convention", sort=True))
        per_family = max(1, 20 // len(families))
        for _, family in families:
            chosen.extend(
                family.sort_values("filename").head(per_family).index
            )
        remaining = group.drop(index=chosen).sort_values("filename")
        chosen.extend(remaining.head(max(0, 20 - len(chosen))).index)
        result.loc[chosen, "representative_example"] = True
    return result


def quality_breakdown(frame, flags, canonical):
    """Reuse original fences; count each content/metric pair only once."""
    merged = flags.merge(
        frame[["filepath", "sha256_hash", "target_label"]],
        on="filepath", validate="many_to_one",
    )
    unique = merged.drop_duplicates(["sha256_hash", "metric"]).copy()
    counts = unique.groupby("sha256_hash").metric.nunique()
    rows = []
    for metric in [*METRICS, "other"]:
        if metric != "other":
            part = unique[unique.metric == metric]
        else:
            part = unique[~unique.metric.isin(METRICS)]
        raw = merged[merged.metric.isin(part.metric.unique())]
        rows.append({
            "summary_type": "reason", "category": metric,
            "unique_image_count": part.sha256_hash.nunique(),
            "unique_image_metric_flags": len(part),
            "original_path_metric_flags": len(raw),
        })
    for label in LABELS:
        part = unique[unique.target_label == label]
        rows.append({
            "summary_type": "class", "category": label,
            "unique_image_count": part.sha256_hash.nunique(),
            "unique_image_metric_flags": len(part),
            "original_path_metric_flags": int(
                (merged.target_label == label).sum()
            ),
        })
    for category, count in [
        ("exactly_one_flag", int((counts == 1).sum())),
        ("multiple_flags", int((counts > 1).sum())),
        ("all_flagged", len(counts)),
    ]:
        rows.append({
            "summary_type": "flag_multiplicity", "category": category,
            "unique_image_count": count,
        })
    ranking = unique.groupby("sha256_hash", sort=True).agg(
        flag_count=("metric", "nunique"),
        reasons=("metric", json_list),
    ).reset_index()
    ranking = ranking.merge(
        canonical[["sha256_hash", "filepath", "target_label"]],
        on="sha256_hash", how="left", validate="one_to_one",
    ).sort_values(["flag_count", "sha256_hash"], ascending=[False, True])
    return pd.DataFrame(rows), ranking.head(30), unique


def local_documentation(root, archive):
    """Inventory local evidence; authored audit notes are not provenance."""
    suffixes = {".md", ".txt", ".csv", ".json", ".pdf", ".html", ".xml"}
    docs = [p.relative_to(root).as_posix() for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in suffixes]
    nonimages = sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                       if p.is_file() and p.suffix.lower() not in EXTENSIONS)
    archive_docs = []
    archive_entries = None
    if archive and archive.is_file():
        with zipfile.ZipFile(archive) as zipped:
            archive_entries = len(zipped.namelist())
            archive_docs = sorted(n for n in zipped.namelist()
                                  if Path(n).suffix.lower() in suffixes)
    return {
        "dataset_documentation_candidates": sorted(docs),
        "dataset_non_image_files": nonimages,
        "archive_name": archive.name if archive else None,
        "archive_entry_count": archive_entries,
        "archive_documentation_candidates": archive_docs,
        "provenance_warning": (
            "Project README and reports were authored during this audit; "
            "they do not independently establish patient identity."
        ),
    }


def run(root, output, archive):
    root, output = root.resolve(), output.resolve()
    if root == output or root in output.parents or output in root.parents:
        raise ValueError("Output must be disjoint from raw data.")
    frame = pd.read_csv(output / "image_metadata_audit.csv")
    verify_sources(frame, root)
    print("All inventoried source hashes reverified.", flush=True)
    groups, canonical, exclusions = build_groups(frame)
    comparisons = compare_trees(frame)
    sidecars = inspect_sidecars(frame, root)
    semantics = patient_semantics(canonical)
    flags = pd.read_csv(output / "image_quality_flags.csv")
    quality, top, unique_flags = quality_breakdown(frame, flags, canonical)
    rules = {
        "NORMAL": "NORMAL folder; no bacteria/virus token",
        "BACTERIAL_PNEUMONIA": "PNEUMONIA folder + whole bacteria token",
        "VIRAL_PNEUMONIA": "PNEUMONIA folder + whole virus token",
    }
    provenance = pd.DataFrame([{
        "class": label,
        "example_filename": canonical.loc[
            canonical.target_label == label, "filename"
        ].sort_values().iloc[0],
        "parsing_rule": rules[label],
        "number_of_images": int((canonical.target_label == label).sum()),
        "count_scope": "provisional unique readable hashes",
    } for label in LABELS])
    outputs = {
        "directory_inventory.csv": directory_inventory(frame, root),
        "directory_copy_comparison.csv": comparisons,
        "duplicate_group_audit.csv": groups,
        "canonical_image_manifest.csv": canonical,
        "canonical_manifest_exclusions.csv": exclusions,
        "sidecar_verification.csv": sidecars,
        "patient_id_semantics_audit.csv": semantics,
        "patient_filename_examples.csv": semantics[
            semantics.representative_example
        ].sort_values(["class", "filename"]),
        "label_provenance.csv": provenance,
        "quality_flag_summary.csv": quality,
        "quality_top30_unique_images.csv": top,
        "quality_flags_unique_content.csv": unique_flags,
    }
    for filename, table in outputs.items():
        table.to_csv(output / filename, index=False)
    evidence = local_documentation(root, archive)
    (output / "local_documentation_inventory.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    readable = frame[frame.readable]
    readable_sizes = readable.groupby("sha256_hash").size()
    summary = {
        "raw_image_extension_files": len(frame),
        "readable_file_count": len(readable),
        "verified_filesystem_sidecars": int(
            sidecars.appledouble_header_verified.sum()
        ),
        "total_duplicate_groups_all_files": len(groups),
        "excess_duplicate_files_all_files": int(
            groups.excess_duplicate_files.sum()
        ),
        "readable_duplicate_groups": int((readable_sizes > 1).sum()),
        "excess_readable_duplicate_files": int(
            len(readable) - len(readable_sizes)
        ),
        "across_split_duplicate_groups_all_files": int(
            groups.across_original_splits.sum()
        ),
        "conflicting_label_duplicate_groups_all_files": int(
            groups.conflicting_labels.sum()
        ),
        "readable_conflicting_label_groups": int(
            (groups.conflicting_labels
             & groups.readable_file_count.gt(0)).sum()
        ),
        "readable_across_split_groups": int(
            (groups.across_original_splits
             & groups.readable_file_count.gt(0)).sum()
        ),
        "directory_copy_groups": int(
            groups.duplicated_dataset_directories.sum()
        ),
        "directory_comparison_rows": len(comparisons),
        "directory_comparison_hash_matches": int(comparisons.same_hash.sum()),
        "unique_image_content_count": len(readable_sizes),
        "provisional_modeling_n": len(canonical),
        "final_modeling_n": None,
        "canonical_class_counts": (
            canonical.target_label.value_counts().to_dict()
        ),
        "unique_quality_flagged_images": int(
            unique_flags.sha256_hash.nunique()
        ),
        "patient_grouping_defensible": False,
        "plan": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
    }
    (output / "integrity_verification_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed")
    )
    parser.add_argument("--archive", type=Path, default=Path("archive.zip"))
    args = parser.parse_args()
    run(args.dataset_root, args.output_dir, args.archive)


if __name__ == "__main__":
    main()
