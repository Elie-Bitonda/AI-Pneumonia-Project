"""Synthetic audit checks, including source-byte preservation."""

import contextlib
import hashlib
import io
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from src.pipeline_builder import quality_audit, run_audit
from src.transformers import METRICS
from src.transformers import parse_identity


class AuditTests(unittest.TestCase):
    def test_iqr_flags_preserve_values_and_use_strict_fences(self):
        data = pd.DataFrame({
            "filepath": [f"{i}.png" for i in range(5)],
            "readable": [True] * 5,
            **{metric: [10, 10, 10, 10, 100] for metric in METRICS},
        })
        before = data.copy(deep=True)
        summaries, flags = quality_audit(data)
        self.assertEqual(len(flags), len(METRICS))
        self.assertEqual(set(flags.filepath), {"4.png"})
        self.assertTrue((summaries.iqr == 0).all())
        pd.testing.assert_frame_equal(before, data)

    def test_conservative_labels(self):
        examples = {
            "train/NORMAL/IM-0001.jpeg": "NORMAL",
            "test/PNEUMONIA/person1_bacteria_1.jpeg": "BACTERIAL_PNEUMONIA",
            "val/PNEUMONIA/person2_virus_1.PNG": "VIRAL_PNEUMONIA",
            "train/PNEUMONIA/person3_bacterial_1.jpeg": None,
            "train/PNEUMONIA/person3_bacteria_virus.jpeg": None,
            "train/NORMAL/person4_virus_1.jpeg": None,
            "NORMAL/PNEUMONIA/person5_virus_1.jpeg": None,
        }
        for path, expected in examples.items():
            with self.subTest(path=path):
                self.assertEqual(
                    parse_identity(path)["target_label"], expected
                )
        self.assertIsNone(parse_identity(
            "NORMAL/IM-0001-0001.jpeg"
        )["patient_id_if_parseable"])
        self.assertEqual(parse_identity(
            "PNEUMONIA/person01_virus_1.jpeg"
        )["patient_id_if_parseable"], "person1")

    def test_inventory_duplicates_corruption_and_no_raw_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "raw"
            for split in ("train", "val", "test"):
                (root / split / "PNEUMONIA").mkdir(parents=True)
            first = root / "train/PNEUMONIA/person1_bacteria_1.PNG"
            Image.new("L", (12, 8), color=100).save(first)
            copied = root / "test/PNEUMONIA/person1_bacteria_1.PNG"
            copied.write_bytes(first.read_bytes())
            (root / "val/PNEUMONIA/unknown.jpg").write_bytes(b"broken")
            Image.new("RGB", (8, 10), color=(1, 2, 3)).save(
                root / "val/PNEUMONIA/person2_virus_1.png"
            )
            before = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in root.rglob("*") if p.is_file()}
            with contextlib.redirect_stdout(io.StringIO()):
                run_audit(root, base / "out", base / "figures")
            data = pd.read_csv(base / "out/image_metadata_audit.csv")
            self.assertEqual(len(data), 4)
            self.assertEqual(data.readable.sum(), 3)
            self.assertEqual((data.label_parse_status == "FAILED").sum(), 1)
            self.assertEqual(data.sha256_hash.nunique(), 3)
            normal = data[data.filename == first.name].iloc[0]
            self.assertEqual(normal.mean_intensity, 100)
            self.assertEqual(normal.std_intensity, 0)
            self.assertEqual(normal.aspect_ratio, 1.5)
            after = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in root.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            with self.assertRaises(ValueError):
                run_audit(root, root / "processed", base / "figures")


if __name__ == "__main__":
    unittest.main()
