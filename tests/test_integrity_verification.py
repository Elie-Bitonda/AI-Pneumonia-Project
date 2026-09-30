"""Canonical selection must fail closed on labels and packaging metadata."""

import json
import unittest

import pandas as pd

from src.integrity_verification import build_groups, compare_trees
from src.transformers import parse_identity


def sample(path, digest, label="NORMAL", readable=True):
    parsed = parse_identity(path)
    return {
        "filepath": path, "filename": path.split("/")[-1],
        "sha256_hash": digest, "readable": readable,
        **parsed, "target_label": label,
    }


class IntegrityTests(unittest.TestCase):
    def test_sidecars_excluded_even_if_readable(self):
        data = pd.DataFrame([
            sample("__MACOSX/train/NORMAL/._a.jpeg", "sidecar"),
            sample("train/NORMAL/._b.jpeg", "sidecar2"),
            sample("train/NORMAL/a.jpeg", "image"),
        ])
        _, canonical, exclusions = build_groups(data)
        self.assertEqual(list(canonical.sha256_hash), ["image"])
        self.assertEqual(len(exclusions), 2)

    def test_groups_and_canonical_priority(self):
        data = pd.DataFrame([
            sample("copy/train/NORMAL/a.jpeg", "a"),
            sample("train/NORMAL/a.jpeg", "a"),
            sample("test/NORMAL/other.jpeg", "a"),
            sample("val/NORMAL/b.jpeg", "b"),
        ])
        groups, canonical, _ = build_groups(data)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups.iloc[0].duplicate_group_size, 3)
        self.assertEqual(groups.iloc[0].classification_codes, "A|B|D")
        self.assertEqual(len(json.loads(groups.iloc[0].all_filepaths)), 3)
        self.assertTrue(canonical.sha256_hash.is_unique)
        self.assertEqual(len(canonical), 2)
        chosen = canonical[canonical.sha256_hash == "a"].iloc[0]
        self.assertEqual(chosen.filepath, "test/NORMAL/other.jpeg")
        _, reordered, _ = build_groups(data.iloc[::-1])
        pd.testing.assert_frame_equal(canonical, reordered)

    def test_conflicts_never_retained(self):
        data = pd.DataFrame([
            sample("train/NORMAL/a.jpeg", "conflict"),
            sample("test/PNEUMONIA/person1_virus_1.jpeg", "conflict",
                   "VIRAL_PNEUMONIA"),
        ])
        groups, canonical, exclusions = build_groups(data)
        self.assertTrue(canonical.empty)
        self.assertTrue(groups.iloc[0].conflicting_labels)
        self.assertEqual(exclusions.iloc[0].status, "CONFLICT_REQUIRES_REVIEW")

    def test_tree_evidence_requires_matching_hashes(self):
        data = pd.DataFrame([
            sample("train/NORMAL/a.jpeg", "a"),
            sample("copy/train/NORMAL/a.jpeg", "different"),
        ])
        comparison = compare_trees(data)
        self.assertFalse(comparison.iloc[0].same_hash)
        self.assertTrue(comparison.iloc[0].same_label)

    def test_class_parsing_is_deterministic(self):
        for path, expected in [
            ("train/NORMAL/IM-1-1.jpeg", "NORMAL"),
            ("test/PNEUMONIA/person1_bacteria_1.jpeg", "BACTERIAL_PNEUMONIA"),
            ("val/PNEUMONIA/person1_virus_1.jpeg", "VIRAL_PNEUMONIA"),
        ]:
            self.assertEqual(parse_identity(path), parse_identity(path))
            self.assertEqual(parse_identity(path)["target_label"], expected)


if __name__ == "__main__":
    unittest.main()
