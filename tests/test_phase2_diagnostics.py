"""I verify training diagnostics, figures, and raw-data integrity."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.phase2_diagnostics import (outlier_diagnostic, transformation_diagnostic, pca_diagnostic,
    analyze_training, generate_figures, FIGURES, missingness_categories, raw_snapshot, verify_raw_inventory,
    summarize_associations, verify_phase1_preservation)
from src.phase2_grouping import load_manifest
from src.phase2_pipeline import NUMERIC_PREDICTORS, load_canonical_image
from src.preprocessing_summary import read_table

PROJECT=Path(__file__).resolve().parents[1]


class DiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.canonical=load_manifest(PROJECT)
        cls.training=cls.canonical.iloc[:240].copy()
        cls.diagnostics=analyze_training(cls.training)
        cls.diagnostics["association_across_folds"] = summarize_associations(pd.concat([
            cls.diagnostics["associations"].assign(outer_fold=fold) for fold in range(1,6)], ignore_index=True))

    def test_outlier_thresholds_learned_from_supplied_training(self):
        records,capped,union=outlier_diagnostic(self.training)
        q1=self.training[list(NUMERIC_PREDICTORS)].quantile(.25)
        q3=self.training[list(NUMERIC_PREDICTORS)].quantile(.75)
        np.testing.assert_allclose(records.upper_fence,q3+1.5*(q3-q1))
        self.assertFalse(records.pixels_capped.any()); self.assertFalse(records.cap_used_in_pipeline.any())
        self.assertEqual(len(capped),len(self.training))
        before=self.training.copy(deep=True)
        outlier_diagnostic(self.training)
        pd.testing.assert_frame_equal(before,self.training)

    def test_parallel_transformations_not_scaler_chain(self):
        values,stats=transformation_diagnostic(self.training)
        raw=self.training.file_size_bytes.to_numpy()
        np.testing.assert_allclose(values["Log1p(raw)"],np.log1p(raw))
        np.testing.assert_allclose(values["MinMax(raw)"],(raw-raw.min())/(raw.max()-raw.min()))
        np.testing.assert_allclose(values["Standardized(raw)"],(raw-raw.mean())/raw.std())
        self.assertAlmostEqual(stats.iloc[0].skewness,stats.iloc[2].skewness)
        with self.assertRaises(ValueError): transformation_diagnostic(self.training.assign(file_size_bytes=-1))

    def test_pca_dimensions_and_threshold_consistency(self):
        summary,counts,pipe,pca=pca_diagnostic(self.training)
        self.assertEqual(pca.n_features_in_,6); self.assertEqual(len(summary),6)
        self.assertAlmostEqual(summary.explained_variance_ratio.sum(),1)
        self.assertTrue((np.diff(summary.cumulative_explained_variance)>=0).all())
        for key,count in counts.items():
            self.assertGreaterEqual(summary.iloc[count-1].cumulative_explained_variance+1e-12,float(key))
            if count>1: self.assertLess(summary.iloc[count-2].cumulative_explained_variance,float(key))
        self.assertEqual(pipe.transform(self.training).shape[1],6)

    def test_post_collinearity_pca_matches_retained_pipeline_without_refitting(self):
        from sklearn.decomposition import PCA
        pipe = self.diagnostics["pipeline"]
        scaler = pipe.named_steps["columns"].named_transformers_["numeric"].named_steps["scale"]
        mean = scaler.mean_.copy()
        rows = scaler.n_samples_seen_
        summary, counts, used_pipe, actual = pca_diagnostic(self.training, fitted_pipeline=pipe)
        expected = PCA(svd_solver="full").fit(pipe.transform(self.training[list(NUMERIC_PREDICTORS)]))
        self.assertIs(used_pipe, pipe)
        self.assertEqual(len(summary), self.diagnostics["output_features"])
        np.testing.assert_allclose(actual.explained_variance_ratio_, expected.explained_variance_ratio_)
        np.testing.assert_array_equal(mean, scaler.mean_)
        self.assertEqual(rows, scaler.n_samples_seen_)

    def test_across_fold_mi_sample_sd_and_scope_validation(self):
        fixture = pd.DataFrame([{"outer_fold": fold, "variable": "width", "mi_nats": fold/10}
                                for fold in range(1,6)])
        row = summarize_associations(fixture).iloc[0]
        self.assertAlmostEqual(row.mean_mi_nats,.3)
        self.assertAlmostEqual(row.sd_mi_nats,np.std(np.arange(1,6)/10,ddof=1))
        self.assertAlmostEqual(row.min_mi_nats,.1); self.assertAlmostEqual(row.max_mi_nats,.5)
        self.assertAlmostEqual(row.fold1_mi_nats,.1)
        with self.assertRaises(ValueError): summarize_associations(fixture.iloc[:4])
        with self.assertRaises(ValueError): summarize_associations(pd.concat([fixture,fixture.iloc[:1]]))

    def test_structural_missingness_distinction(self):
        cols,values=missingness_categories(self.canonical)
        self.assertFalse((values[:,:6]!=0).any())
        self.assertTrue((values[:,cols.index("read_error")]==2).all())
        self.assertEqual(int((values[:,cols.index("patient_id_if_parseable")]==3).sum()),1580)

    def test_phase1_preservation_including_text_inside_figure_directory(self):
        result=verify_phase1_preservation(PROJECT)
        self.assertTrue(result["committed_contents_match"])
        self.assertTrue(result["other_phase1_image_figures_byte_identical"])
        self.assertIn("figures/manual_review/review_index.csv",result["working_file_sha256"])
        self.assertNotIn("figures/fig3_1_correlation_matrix.png",result["working_file_sha256"])

    def test_reduction_follows_vif_rule_and_training_scope(self):
        reducer=self.diagnostics["pipeline"].named_steps["rank_vif"]
        self.assertEqual(reducer.fit_rows_,len(self.training))
        if reducer.initial_rank_vif_.VIF.max()>10:
            self.assertEqual(reducer.removals_[0]["column"],"aspect_ratio")
        self.assertLessEqual(reducer.final_rank_vif_.VIF.max(),10)
        self.assertEqual(len(reducer.selected_columns_),self.diagnostics["output_features"])

    def test_association_reproducibility_on_training(self):
        other=analyze_training(self.training)
        pd.testing.assert_frame_equal(self.diagnostics["associations"],other["associations"])
        self.assertEqual(len(other["associations"]),6)

    def test_exact_eight_figures_execute(self):
        with tempfile.TemporaryDirectory() as folder:
            paths=generate_figures(self.canonical,self.training,self.diagnostics,folder)
            self.assertEqual({Path(p).name for p in paths},set(FIGURES))
            self.assertEqual(len(list(Path(folder).glob("*.png"))),8)
            for p in paths: self.assertGreater(Path(p).stat().st_size,1000)

    def test_full_raw_inventory_and_on_demand_read_preserve_every_file(self):
        root=PROJECT/"chest_xray"
        before=raw_snapshot(root)
        inventory=read_table(PROJECT/"data/processed/image_metadata_audit.csv")
        verify_raw_inventory(root,inventory,before)
        for mode in ["L","RGB"]:
            row=self.canonical[self.canonical.image_mode==mode].iloc[0]
            load_canonical_image(root,row)
        self.assertEqual(before,raw_snapshot(root))
        tampered=dict(before); tampered[inventory.iloc[0].filepath]="canary"
        with self.assertRaises(ValueError): verify_raw_inventory(root,inventory,tampered)


if __name__=="__main__": unittest.main()
