"""I preserve uncertain filename relations and pHash edges as split constraints."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .preprocessing_summary import read_table
from .transformers import LABELS, METRICS, is_filesystem_metadata, parse_identity

CLASS_COUNTS = dict(zip(LABELS, (1579, 2760, 1485)))
CLASS_INDEX = dict(zip(LABELS, range(3)))
SEED = 42


def validate_manifest(frame, expected_counts=None):
    """I reject conflicts, duplicates, metadata paths, and incomplete predictors."""
    if frame.empty or frame.sha256_hash.isna().any() or not frame.sha256_hash.is_unique:
        raise ValueError("Canonical SHA-256 values must be present and unique.")
    if not frame.filepath.is_unique or frame.filepath.map(is_filesystem_metadata).any():
        raise ValueError("Canonical paths must be unique non-metadata paths.")
    if not frame.readable.eq(True).all() or not frame.label_parse_status.eq("OK").all():
        raise ValueError("Every retained image must be readable with a resolved label.")
    if not frame.target_label.isin(LABELS).all():
        raise ValueError("Unknown target label.")
    if not np.isfinite(frame[list(METRICS)].to_numpy(float)).all():
        raise ValueError("Selected numeric diagnostics must be complete and finite.")
    for row in frame.itertuples():
        if parse_identity(row.filepath)["target_label"] != row.target_label:
            raise ValueError(f"Path/target conflict: {row.filepath}")
        if set(json.loads(row.all_target_labels)) != {row.target_label}:
            raise ValueError(f"Alternative-path target conflict: {row.filepath}")
    if expected_counts is not None and frame.target_label.value_counts().to_dict() != expected_counts:
        raise ValueError("Canonical class distribution differs from Phase 1.")
    return frame.sort_values("sha256_hash", kind="stable").reset_index(drop=True).copy()


def load_manifest(project):
    return validate_manifest(read_table(Path(project) / "data/processed/canonical_image_manifest.csv"), CLASS_COUNTS)


def filename_group(filename):
    """I use relation tokens, including the documented extra viral suffix."""
    stem = Path(filename).stem
    person = re.fullmatch(r"person(\d+)_(?:bacteria|virus)_\d+(?:_\d+)?", stem)
    if person:
        return f"person:{int(person[1])}"
    normal = re.fullmatch(r"(IM|NORMAL2-IM)-(\d+)-\d+(?:-\d+)?", stem)
    if normal:
        return f"normal:{normal[1]}:{int(normal[2])}"
    raise ValueError(f"Unreviewed filename convention: {filename}")


def constrain_groups(frame, candidates):
    """I connect unresolved edges for splitting without asserting image identity.

    I accept only the committed unresolved status at this checkpoint. A future
    adjudication requires an explicit policy update rather than silently dropping
    unknown statuses. Component names depend on members, never target or split.
    """
    result = frame.copy()
    result["base_relation_group"] = result.filename.map(filename_group)
    parent = {g: g for g in result.base_relation_group.unique()}

    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    lookup = result.set_index("sha256_hash").base_relation_group.to_dict()
    for row in candidates.itertuples():
        if row.review_status != "CANDIDATE_NOT_CONFIRMED":
            raise ValueError("Candidate review status requires explicit adjudication policy.")
        if row.sha256_1 not in lookup or row.sha256_2 not in lookup:
            raise ValueError("pHash edge endpoint is absent from canonical membership.")
        a, b = find(lookup[row.sha256_1]), find(lookup[row.sha256_2])
        parent[max(a, b)] = min(a, b)
    members = {}
    for g in sorted(parent):
        members.setdefault(find(g), []).append(g)
    names = {k: "constraint:" + hashlib.sha256("\n".join(v).encode()).hexdigest()
             for k, v in members.items()}
    result["leakage_control_group"] = [names[find(g)] for g in result.base_relation_group]
    return result


def class_weights(training_labels):
    """I calculate weights from the active training labels only."""
    labels = pd.Series(training_labels)
    counts = labels.value_counts().reindex(LABELS, fill_value=0)
    if not labels.isin(LABELS).all() or (counts == 0).any():
        raise ValueError("Active training data must contain all three recognized classes.")
    return pd.DataFrame({"target_label": LABELS, "class_index": range(3),
                         "training_count": counts.to_numpy(),
                         "weight": len(labels) / (3 * counts.to_numpy())})


def assert_partitions(frame, assignments, candidates, n_splits=5):
    """I check every role pair, evaluation coverage, and every uncertain edge."""
    hashes = set(frame.sha256_hash)
    evaluation = assignments[assignments.role == "evaluation"].sha256_hash
    if len(evaluation) != len(frame) or set(evaluation) != hashes or not evaluation.is_unique:
        raise ValueError("Each canonical content must be evaluated exactly once.")
    if set(assignments.outer_fold) != set(range(1, n_splits + 1)):
        raise ValueError("Unexpected outer fold identifiers.")
    for _, part in assignments.groupby("outer_fold"):
        if len(part) != len(frame) or set(part.sha256_hash) != hashes or not part.sha256_hash.is_unique:
            raise ValueError("Each fold must partition all canonical contents exactly once.")
        if set(part.role) != {"train", "validation", "evaluation"}:
            raise ValueError("Every fold requires three roles.")
        expected = frame.set_index("sha256_hash")
        for column in ["filepath", "target_label", "base_relation_group", "leakage_control_group"]:
            if not np.array_equal(part[column].to_numpy(), expected.loc[part.sha256_hash, column].to_numpy()):
                raise ValueError(f"Saved assignment differs from canonical/grouping evidence: {column}")
        for col in ["base_relation_group", "leakage_control_group"]:
            if part.groupby(col).role.nunique().max() != 1:
                raise ValueError(f"Leakage across roles: {col}")
        roles = part.set_index("sha256_hash").role
        if any(roles[r.sha256_1] != roles[r.sha256_2] for r in candidates.itertuples()):
            raise ValueError("An unresolved pHash edge crosses roles.")
    return {"sha_role_isolation": True, "group_role_isolation": True,
            "base_relation_isolation": True, "phash_edge_isolation": True,
            "evaluation_coverage_exactly_once": True,
            "source_split_not_used": True, "target_not_a_predictor": True}


def audit_constraint_graph(frame, candidates):
    """Independently reconstruct the base-group graph with sparse components.

    Graph component labels must be in one-to-one correspondence with the saved
    split components. This checks the complete partition, not just its count.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    names = sorted(frame.base_relation_group.unique())
    indices = {name: i for i, name in enumerate(names)}
    by_hash = frame.set_index("sha256_hash")
    rows, columns, touched_hashes = [], [], set()
    for edge in candidates.itertuples():
        if edge.review_status != "CANDIDATE_NOT_CONFIRMED":
            raise ValueError("Unexpected adjudication state.")
        a = indices[by_hash.loc[edge.sha256_1, "base_relation_group"]]
        b = indices[by_hash.loc[edge.sha256_2, "base_relation_group"]]
        rows.extend([a,b]); columns.extend([b,a])
        touched_hashes.update([edge.sha256_1, edge.sha256_2])
    graph = coo_matrix((np.ones(len(rows)), (rows, columns)), shape=(len(names),len(names))).tocsr()
    count, labels = connected_components(graph, directed=False)
    independent = frame.base_relation_group.map({g: int(labels[i]) for g,i in indices.items()})
    comparison = pd.DataFrame({"independent": independent, "saved": frame.leakage_control_group})
    if (comparison.groupby("independent").saved.nunique().max() != 1
            or comparison.groupby("saved").independent.nunique().max() != 1):
        raise ValueError("Saved components differ from the unresolved-edge graph.")
    touched = by_hash.loc[sorted(touched_hashes), "leakage_control_group"].nunique()
    return {"phash_edges": len(candidates), "base_groups": len(names),
            "final_components": int(count), "net_base_group_merges": len(names)-int(count),
            "components_touched_by_edges": int(touched),
            "unique_edge_endpoint_contents": len(touched_hashes),
            "largest_component_images": int(frame.leakage_control_group.value_counts().max()),
            "mixed_class_components": int(frame.groupby("leakage_control_group").target_label.nunique().gt(1).sum()),
            "independent_graph_partition_matches": True,
            "interpretation": "Unresolved similarity edges constrain role allocation; mixed-class components are not verified patient identities."}


def make_partitions(frame, candidates, seed=SEED):
    """I reserve the first deterministic inner ten-fold partition for validation.

    I do not search seeds or folds for favorable balance. Labels are used only
    for stratification; X is a constant placeholder and source split is unused.
    """
    data = frame.sort_values("sha256_hash", kind="stable").reset_index(drop=True)
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    assignments, summaries, weights = [], [], []
    for fold, (dev, evaluation) in enumerate(outer.split(
            np.zeros((len(data), 1)), data.target_label, data.leakage_control_group), 1):
        development = data.iloc[dev]
        inner = StratifiedGroupKFold(10, shuffle=True, random_state=seed + fold)
        train_rel, val_rel = next(inner.split(np.zeros((len(dev), 1)),
                                             development.target_label,
                                             development.leakage_control_group))
        train, validation = dev[train_rel], dev[val_rel]
        role = np.full(len(data), "evaluation", dtype=object)
        role[train], role[validation] = "train", "validation"
        part = data[["sha256_hash", "filepath", "target_label", "base_relation_group",
                     "leakage_control_group"]].copy()
        part["outer_fold"], part["role"] = fold, role
        assignments.append(part)
        for name, indices in [("development", dev), ("train", train),
                              ("validation", validation), ("evaluation", evaluation)]:
            subset = data.iloc[indices]
            counts = subset.target_label.value_counts()
            if set(counts.index) != set(LABELS):
                raise ValueError(f"Fold {fold} {name} is missing a target class.")
            record = {"outer_fold": fold, "role": name, "n": len(subset),
                      "cohort_fraction": len(subset) / len(data),
                      "groups": subset.leakage_control_group.nunique()}
            for label in LABELS:
                record[label] = int(counts.get(label, 0))
                record[label + "_fraction"] = counts.get(label, 0) / len(subset)
            summaries.append(record)
        w = class_weights(data.iloc[train].target_label)
        w.insert(0, "outer_fold", fold)
        weights.append(w)
    assignments = pd.concat(assignments, ignore_index=True)
    checks = assert_partitions(data, assignments, candidates)
    return assignments, pd.DataFrame(summaries), pd.concat(weights, ignore_index=True), checks
