"""I test relation constraints, reproducible roles, and invalid membership."""

import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.phase2_grouping import (load_manifest, validate_manifest, filename_group,
    constrain_groups, make_partitions, assert_partitions, class_weights, CLASS_COUNTS, audit_constraint_graph)
from src.preprocessing_summary import read_table

PROJECT = Path(__file__).resolve().parents[1]


class GroupingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.canonical = load_manifest(PROJECT)
        cls.candidates = read_table(PROJECT / "data/processed/near_duplicate_candidates.csv")
        cls.grouped = constrain_groups(cls.canonical, cls.candidates)
        cls.assignments, cls.summary, cls.weights, cls.checks = make_partitions(cls.grouped, cls.candidates)

    def test_manifest_distribution_and_conflict_detection(self):
        self.assertEqual(self.canonical.target_label.value_counts().to_dict(), CLASS_COUNTS)
        bad = self.canonical.copy()
        bad.loc[0, "all_target_labels"] = json.dumps(["NORMAL", "VIRAL_PNEUMONIA"])
        with self.assertRaises(ValueError):
            validate_manifest(bad)
        with self.assertRaises(ValueError):
            validate_manifest(pd.concat([self.canonical, self.canonical.iloc[:1]]))
        bad = self.canonical.copy(); bad.loc[0, "mean_intensity"] = np.nan
        with self.assertRaises(ValueError):
            validate_manifest(bad)

    def test_relation_namespaces_and_extra_suffix(self):
        self.assertEqual(filename_group("person447_virus_921_1.jpeg"), filename_group("person447_bacteria_1.jpeg"))
        self.assertEqual(filename_group("IM-0001-0001.jpeg"), filename_group("IM-0001-0002.jpeg"))
        self.assertNotEqual(filename_group("IM-0001-0001.jpeg"), filename_group("NORMAL2-IM-0001-0001.jpeg"))
        with self.assertRaises(ValueError): filename_group("unreviewed.jpeg")

    def test_base_and_order_invariant_components(self):
        self.assertEqual(self.grouped.base_relation_group.nunique(), 3117)
        changed = constrain_groups(self.canonical.sample(frac=1, random_state=8), self.candidates.iloc[::-1])
        a = self.grouped.set_index("sha256_hash").leakage_control_group.sort_index()
        b = changed.set_index("sha256_hash").leakage_control_group.sort_index()
        pd.testing.assert_series_equal(a,b)

    def test_unknown_candidate_status_and_endpoint_rejected(self):
        bad = self.candidates.copy(); bad.loc[0,"review_status"] = "CONFIRMED"
        with self.assertRaises(ValueError): constrain_groups(self.canonical, bad)
        bad = self.candidates.copy(); bad.loc[0,"sha256_1"] = "missing"
        with self.assertRaises(ValueError): constrain_groups(self.canonical, bad)

    def test_unresolved_edges_stay_in_same_component(self):
        lookup = self.grouped.set_index("sha256_hash").leakage_control_group
        for r in self.candidates.itertuples():
            self.assertEqual(lookup[r.sha256_1], lookup[r.sha256_2])

    def test_independent_graph_partition_and_tampering_detection(self):
        audit = audit_constraint_graph(self.grouped, self.candidates)
        self.assertEqual(audit["final_components"], self.grouped.leakage_control_group.nunique())
        self.assertEqual(audit["net_base_group_merges"], audit["base_groups"]-audit["final_components"])
        bad=self.grouped.copy(); bad["leakage_control_group"]="all combined"
        with self.assertRaises(ValueError): audit_constraint_graph(bad,self.candidates)

    def test_saved_assignment_groups_cannot_be_relabelled_to_hide_leakage(self):
        bad=self.assignments.copy(); bad.loc[0,"leakage_control_group"]="invented"
        with self.assertRaises(ValueError): assert_partitions(self.grouped,bad,self.candidates)

    def test_reproducible_and_source_split_independent(self):
        changed = self.grouped.sample(frac=1, random_state=91).copy()
        changed["original_split"] = "arbitrary provenance"
        actual, _, _, _ = make_partitions(changed, self.candidates)
        pd.testing.assert_frame_equal(actual, self.assignments)

    def test_coverage_all_roles_and_all_edges(self):
        self.assertTrue(all(assert_partitions(self.grouped, self.assignments, self.candidates).values()))
        for _, part in self.assignments.groupby("outer_fold"):
            groups = {role: set(g.leakage_control_group) for role,g in part.groupby("role")}
            hashes = {role: set(g.sha256_hash) for role,g in part.groupby("role")}
            for a,b in [("train","validation"),("train","evaluation"),("validation","evaluation")]:
                self.assertFalse(groups[a] & groups[b]); self.assertFalse(hashes[a] & hashes[b])
        evaluated = self.assignments[self.assignments.role=="evaluation"]
        self.assertEqual(evaluated.sha256_hash.value_counts().max(),1)
        self.assertEqual(len(evaluated),len(self.canonical))

    def test_assertions_detect_corrupted_roles(self):
        bad = self.assignments.copy()
        row = bad[(bad.outer_fold==1) & (bad.role=="train")].index[0]
        bad.loc[row,"role"] = "evaluation"
        with self.assertRaises(ValueError): assert_partitions(self.grouped,bad,self.candidates)

    def test_class_weights_use_actual_training_labels(self):
        labels = ["NORMAL"]*2+["BACTERIAL_PNEUMONIA"]*3+["VIRAL_PNEUMONIA"]*5
        result = class_weights(labels).set_index("target_label")
        np.testing.assert_allclose(result.weight,[10/6,10/9,10/15])
        for fold, part in self.assignments.groupby("outer_fold"):
            train = part[part.role=="train"]
            expected = class_weights(train.target_label)
            observed = self.weights[self.weights.outer_fold==fold].drop(columns="outer_fold").reset_index(drop=True)
            pd.testing.assert_frame_equal(expected,observed)
        with self.assertRaises(ValueError): class_weights(["NORMAL"])


if __name__ == "__main__": unittest.main()
