"""Statistical and screening primitives tested without fitting any model."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.final_phase1_audit import validate_manifest
from src.image_review import (
    BKTree, channel_differences, make_sheet, perceptual_hash,
)
from src.statistical_audit import (
    holm_adjust, mutual_information, quality_class_analysis,
)
from src.transformers import LABELS, METRICS


class FinalPhase1Tests(unittest.TestCase):
    def test_bktree_matches_exhaustive_small_hash_search(self):
        values = [0, 1, 3, 7, 15, 255, 0, 65535]
        tree = BKTree()
        for i, value in enumerate(values):
            tree.add(value, i)
        for query in [0, 3, 23, 1023]:
            for radius in [0, 1, 4]:
                expected = [(i, (v ^ query).bit_count())
                            for i, v in enumerate(values)
                            if (v ^ query).bit_count() <= radius]
                self.assertEqual(tree.query(query, radius), expected)

    def test_rgb_exact_and_unsigned_difference(self):
        a = np.full((8, 8, 3), 128, dtype=np.uint8)
        self.assertTrue(channel_differences(
            Image.fromarray(a)
        )["rgb_channels_identical"])
        a[0, 0] = [0, 255, 0]
        result = channel_differences(Image.fromarray(a))
        self.assertEqual(result["max_channel_difference"], 255)
        self.assertAlmostEqual(result["mean_channel_difference"], 170 / 64)
        self.assertTrue(result["actual_channel_differences"])

    def test_phash_repeated_and_lossless_reencoding(self):
        pixels = np.random.default_rng(42).integers(0, 256, (64, 64), np.uint8)
        image = Image.fromarray(pixels)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "image.png"
            image.save(path)
            with Image.open(path) as encoded:
                self.assertEqual(
                    perceptual_hash(image), perceptual_hash(encoded)
                )

    def test_holm_and_mutual_information(self):
        np.testing.assert_allclose(holm_adjust([0.01, 0.04, 0.03]),
                                   [0.03, 0.06, 0.06])
        x = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(mutual_information(x, x), np.log(2))
        self.assertAlmostEqual(
            mutual_information(x, np.array([0, 1, 0, 1])), 0
        )

    def test_quality_counts_unique_content(self):
        frame = pd.DataFrame({
            "sha256_hash": ["a", "b", "c"], "target_label": LABELS,
        })
        flags = pd.DataFrame({
            "sha256_hash": ["a", "a", "a"],
            "metric": ["width", "width", "height"],
        })
        result = quality_class_analysis(frame, flags).set_index("target_class")
        self.assertEqual(result.loc["NORMAL", "flagged_images"], 1)
        self.assertEqual(result.loc["NORMAL", "multiple_flag_count"], 1)
        self.assertEqual(result.loc["NORMAL", "width_flag_count"], 1)

    def test_manifest_rejects_duplicate_contents(self):
        frame = pd.DataFrame({
            "sha256_hash": ["same", "same"], "readable": [True, True],
            "filepath": ["train/a.png", "test/a.png"],
            "target_label": ["NORMAL", "NORMAL"],
            **{metric: [1, 1] for metric in METRICS},
        })
        with self.assertRaises(ValueError):
            validate_manifest(frame)

    def test_sheet_preserves_raw_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw.png"
            Image.new("L", (30, 50), color=100).save(source)
            before = source.read_bytes()
            make_sheet([("raw.png", "NORMAL", "test review")], root,
                       root / "sheet.png", "Test")
            self.assertEqual(source.read_bytes(), before)
            self.assertTrue((root / "sheet.png").is_file())


if __name__ == "__main__":
    unittest.main()
