"""Tests for Phase 1 preprocessing summary utilities."""

import unittest

import numpy as np
import pandas as pd

from src.preprocessing_summary import calculate_vif, missingness


class PreprocessingSummaryTests(unittest.TestCase):
    def test_literal_missingness_includes_structural_fields(self):
        data = pd.DataFrame({
            "value": [1, np.nan],
            "read_error": [None, "failure"],
            "hash_error": [None, None],
        })
        result = missingness(data, "fixture").iloc[-1]
        self.assertEqual(result.total_matrix_cells, 6)
        self.assertEqual(result.total_missing_cells, 4)
        self.assertEqual(result.complete_rows, 0)
        self.assertEqual(result.rows_with_missing_values, 2)
        self.assertEqual(result.complete_rows_excluding_optional_errors, 1)

    def test_vif_matches_correlation_inverse_and_is_scale_invariant(self):
        rng = np.random.default_rng(42)
        data = pd.DataFrame(rng.normal(size=(100, 3)), columns=list("abc"))
        data["b"] += data["a"] * 2
        actual = calculate_vif(data).VIF
        expected = np.diag(np.linalg.inv(data.corr()))
        np.testing.assert_allclose(actual, expected, rtol=1e-10)
        changed_units = data * [1000, 0.001, 17] + [31, 200, -51]
        np.testing.assert_allclose(
            actual, calculate_vif(changed_units).VIF, rtol=1e-10
        )

    def test_perfect_collinearity_has_infinite_vif(self):
        data = pd.DataFrame({"a": np.arange(10), "b": np.arange(10) * 2})
        self.assertTrue(np.isinf(calculate_vif(data).VIF).all())

    def test_constant_variable_rejected(self):
        with self.assertRaises(ValueError):
            calculate_vif(pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3]}))


if __name__ == "__main__":
    unittest.main()
