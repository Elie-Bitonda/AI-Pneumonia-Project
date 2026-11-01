"""I test on-demand pixels, bounded augmentation, and train-only metadata state."""

import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.base import clone

from src.phase2_pipeline import (preprocess_image, resize_geometry, pretrained_representation,
    load_canonical_image, AugmentationConfig, augment_training, build_metadata_pipeline, NUMERIC_PREDICTORS)


class ImageTests(unittest.TestCase):
    def test_shape_geometry_padding_range_determinism_source_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            for width,height in [(301,100),(100,301),(13,19),(224,224)]:
                path=Path(folder)/f"{width}_{height}.png"
                Image.new("L",(width,height),128).save(path)
                original=path.read_bytes()
                actual=preprocess_image(path)
                resized,pad=resize_geometry(width,height)
                self.assertEqual(actual.shape,(224,224)); self.assertEqual(actual.dtype,np.float32)
                self.assertEqual(resized[0]+pad[0]+pad[2],224)
                self.assertEqual(resized[1]+pad[1]+pad[3],224)
                self.assertIn(pad[2]-pad[0],(0,1)); self.assertIn(pad[3]-pad[1],(0,1))
                scale=min(224/width,224/height)
                self.assertLessEqual(abs(resized[0]-width*scale),.5)
                self.assertLessEqual(abs(resized[1]-height*scale),.5)
                np.testing.assert_array_equal(actual[pad[1]:pad[1]+resized[1],pad[0]:pad[0]+resized[0]],np.float32(128/255))
                self.assertTrue(0<=actual.min()<=actual.max()<=1)
                np.testing.assert_array_equal(actual,preprocess_image(path))
                self.assertEqual(path.read_bytes(),original)

    def test_equal_rgb_matches_L_and_different_color_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder); values=np.arange(600,dtype=np.uint8).reshape(20,30)
            Image.fromarray(values).save(p/"l.png")
            rgb=np.repeat(values[:,:,None],3,axis=2)
            Image.fromarray(rgb).save(p/"rgb.png")
            np.testing.assert_array_equal(preprocess_image(p/"l.png"),preprocess_image(p/"rgb.png"))
            rgb[0,0,1]+=1; Image.fromarray(rgb).save(p/"color.png")
            with self.assertRaises(ValueError): preprocess_image(p/"color.png")

    def test_canonical_hash_and_root_guards(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"x.png"; Image.new("L",(20,20)).save(p)
            row={"filepath":"x.png","sha256_hash":hashlib.sha256(p.read_bytes()).hexdigest()}
            self.assertEqual(load_canonical_image(folder,row).shape,(224,224))
            row["sha256_hash"]="wrong"
            with self.assertRaises(ValueError): load_canonical_image(folder,row)
            row["filepath"]="../outside.png"
            with self.assertRaises(ValueError): load_canonical_image(folder,row)
            with self.assertRaises(ValueError): preprocess_image(Path(folder)/"._x.png")

    def test_pretrained_replication_explicit_published_constants(self):
        a=np.full((224,224),.5,dtype=np.float32)
        replicated=pretrained_representation(a)
        self.assertEqual(replicated.shape,(3,224,224))
        np.testing.assert_array_equal(replicated[0],replicated[2])
        normalized=pretrained_representation(a,published_mean=(.1,.2,.3),published_std=(.5,.5,.5))
        np.testing.assert_allclose(normalized[:,0,0],[.8,.6,.4],rtol=1e-6)
        with self.assertRaises(ValueError): pretrained_representation(a,published_mean=(0,0,0))

    def test_augmentation_only_training_bounded_seeded_and_nonmutating(self):
        image=Image.fromarray(np.arange(224*224,dtype=np.uint8).reshape(224,224))
        before=image.tobytes(); config=AugmentationConfig()
        a=augment_training(image,config,42); b=augment_training(image,config,42)
        self.assertEqual(a.tobytes(),b.tobytes()); self.assertNotEqual(a.tobytes(),augment_training(image,config,43).tobytes())
        self.assertEqual(before,image.tobytes())
        for role in ["validation","evaluation"]:
            with self.assertRaises(ValueError): augment_training(image,config,42,role)
        with self.assertRaises(ValueError): AugmentationConfig(rotation_degrees=90)
        with self.assertRaises(ValueError): AugmentationConfig(translation_fraction=.1)


class MetadataPipelineTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(7)
        self.data=pd.DataFrame(rng.lognormal(size=(120,6)),columns=NUMERIC_PREDICTORS)
        self.labels=np.tile(["NORMAL","BACTERIAL_PNEUMONIA","VIRAL_PNEUMONIA"],40)

    def test_allowlist_and_target_identifiers_excluded(self):
        pipeline=build_metadata_pipeline()
        forbidden=["target_label","filepath","sha256_hash","filename","original_split","leakage_control_group","patient_id_if_parseable"]
        extended=self.data.assign(**{col:"canary" for col in forbidden})
        actual=pipeline.fit_transform(extended,self.labels)
        expected=build_metadata_pipeline().fit_transform(self.data,self.labels)
        np.testing.assert_allclose(actual,expected)
        for col in forbidden:
            with self.assertRaises(ValueError): build_metadata_pipeline([col])

    def test_train_only_scaler_and_transform_frozen(self):
        pipe=build_metadata_pipeline(reduce_collinearity=False).fit(self.data.iloc[:90])
        scaler=pipe.named_steps["columns"].named_transformers_["numeric"].named_steps["scale"]
        mean=scaler.mean_.copy()
        np.testing.assert_allclose(mean,self.data.iloc[:90].mean())
        self.assertEqual(scaler.n_samples_seen_,90)
        pipe.transform(self.data.iloc[90:]*10000)
        np.testing.assert_array_equal(mean,scaler.mean_)

    def test_clone_and_reproducible_pipeline(self):
        p=build_metadata_pipeline(); q=clone(p)
        np.testing.assert_allclose(p.fit_transform(self.data,self.labels),q.fit_transform(self.data,self.labels))
        self.assertEqual(len(p.get_feature_names_out()),p.transform(self.data).shape[1])
        with self.assertRaises(ValueError):
            p.fit(self.data.assign(width=np.nan),self.labels)


if __name__=="__main__": unittest.main()
