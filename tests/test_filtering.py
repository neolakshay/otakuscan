from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from src.models.reading_order import BoundingBox, DetectedRegion
from src.models.clustering import TextCluster
from src.models.filtering import (
    extract_cluster_features,
    FEATURE_NAMES,
    LearnedClusterClassifier,
    LearnedNarrationClassifier,
    FilterModelConfig,
)


class TestClusterFeatureExtraction(unittest.TestCase):
    def setUp(self) -> None:
        self.page_w = 800
        self.page_h = 1200

    def test_feature_extraction_normal_dialogue(self) -> None:
        bbox = BoundingBox(xmin=100.0, ymin=200.0, xmax=300.0, ymax=400.0)
        reg1 = DetectedRegion(polygon=[[100, 200], [300, 200], [300, 290], [100, 290]], bbox=bbox, text="HELLO WORLD", confidence=0.95)
        reg2 = DetectedRegion(polygon=[[100, 310], [300, 310], [300, 400], [100, 400]], bbox=bbox, text="HOW ARE YOU?", confidence=0.92)
        cluster = TextCluster(
            regions=[reg1, reg2],
            bbox=bbox,
            text="HELLO WORLD HOW ARE YOU?",
            confidence=0.935,
        )

        feats = extract_cluster_features(cluster, page_width=self.page_w, page_height=self.page_h)
        self.assertEqual(len(feats), len(FEATURE_NAMES))
        # Coordinates must be normalized
        self.assertAlmostEqual(feats[FEATURE_NAMES.index("norm_xmin")], 100.0 / 800.0)
        self.assertAlmostEqual(feats[FEATURE_NAMES.index("norm_ymin")], 200.0 / 1200.0)
        self.assertAlmostEqual(feats[FEATURE_NAMES.index("norm_xmax")], 300.0 / 800.0)
        self.assertAlmostEqual(feats[FEATURE_NAMES.index("norm_ymax")], 400.0 / 1200.0)
        self.assertEqual(feats[FEATURE_NAMES.index("num_lines")], 2.0)
        self.assertAlmostEqual(feats[FEATURE_NAMES.index("digit_ratio")], 0.0)
        self.assertGreater(feats[FEATURE_NAMES.index("punct_ratio")], 0.0)

    def test_feature_extraction_noise_page_number(self) -> None:
        bbox = BoundingBox(xmin=10.0, ymin=1150.0, xmax=40.0, ymax=1180.0)
        reg = DetectedRegion(polygon=[[10, 1150], [40, 1150], [40, 1180], [10, 1180]], bbox=bbox, text="84", confidence=0.88)
        cluster = TextCluster(regions=[reg], bbox=bbox, text="84", confidence=0.88)

        feats = extract_cluster_features(cluster, page_width=self.page_w, page_height=self.page_h)
        self.assertEqual(feats[FEATURE_NAMES.index("digit_ratio")], 1.0)
        self.assertEqual(feats[FEATURE_NAMES.index("num_lines")], 1.0)
        self.assertLess(feats[FEATURE_NAMES.index("margin_dist_min")], 0.05)

    def test_feature_extraction_kanji_screentone_noise(self) -> None:
        bbox = BoundingBox(xmin=500.0, ymin=500.0, xmax=550.0, ymax=550.0)
        reg = DetectedRegion(polygon=[[500, 500], [550, 500], [550, 550], [500, 550]], bbox=bbox, text="大级夫装", confidence=0.60)
        cluster = TextCluster(regions=[reg], bbox=bbox, text="大级夫装", confidence=0.60)

        feats = extract_cluster_features(cluster, page_width=self.page_w, page_height=self.page_h)
        self.assertEqual(feats[FEATURE_NAMES.index("non_ascii_ratio")], 1.0)
        self.assertEqual(feats[FEATURE_NAMES.index("alpha_ratio")], 0.0)

    def test_feature_extraction_empty_text_and_zero_dims(self) -> None:
        bbox = BoundingBox(xmin=0.0, ymin=0.0, xmax=0.0, ymax=0.0)
        cluster = TextCluster(regions=[], bbox=bbox, text="", confidence=0.0)
        # Should not raise ZeroDivisionError
        feats = extract_cluster_features(cluster, page_width=self.page_w, page_height=self.page_h)
        self.assertEqual(len(feats), len(FEATURE_NAMES))
        self.assertEqual(feats[FEATURE_NAMES.index("char_count")], 0.0)


class TestLearnedClusterClassifier(unittest.TestCase):
    def setUp(self) -> None:
        # Create linearly separable synthetic dataset
        np.random.seed(42)
        n = 100
        # Positive class: dialogue (high char_count, high punct, low digit, multi-line)
        X_pos = np.zeros((n, len(FEATURE_NAMES)))
        X_pos[:, FEATURE_NAMES.index("char_count")] = np.random.uniform(20, 100, n)
        X_pos[:, FEATURE_NAMES.index("digit_ratio")] = 0.0
        X_pos[:, FEATURE_NAMES.index("punct_ratio")] = np.random.uniform(0.05, 0.2, n)
        X_pos[:, FEATURE_NAMES.index("num_lines")] = np.random.choice([2, 3, 4], n)
        X_pos[:, FEATURE_NAMES.index("margin_dist_min")] = np.random.uniform(0.1, 0.4, n)

        # Negative class: noise (page numbers / kanji noise)
        X_neg = np.zeros((n, len(FEATURE_NAMES)))
        X_neg[:, FEATURE_NAMES.index("char_count")] = np.random.uniform(1, 5, n)
        X_neg[:, FEATURE_NAMES.index("digit_ratio")] = np.random.choice([0.0, 1.0], n)
        X_neg[:, FEATURE_NAMES.index("punct_ratio")] = 0.0
        X_neg[:, FEATURE_NAMES.index("num_lines")] = 1.0
        X_neg[:, FEATURE_NAMES.index("margin_dist_min")] = np.random.uniform(0.01, 0.05, n)

        self.X = np.vstack([X_pos, X_neg])
        self.y = np.hstack([np.ones(n), np.zeros(n)])

    def test_fit_and_predict(self) -> None:
        clf = LearnedClusterClassifier(l2_reg=0.01)
        clf.fit(self.X, self.y)
        self.assertTrue(clf.is_fitted)

        probs = clf.predict_proba(self.X)
        self.assertEqual(len(probs), len(self.y))
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0))

        preds = (probs >= 0.5).astype(float)
        acc = np.mean(preds == self.y)
        self.assertGreaterEqual(acc, 0.95)

    def test_save_and_load_json(self) -> None:
        clf = LearnedClusterClassifier(l2_reg=0.01)
        clf.fit(self.X, self.y)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            clf.save_to_json(tmp_path)
            loaded_clf = LearnedClusterClassifier.load_from_json(tmp_path)
            self.assertTrue(loaded_clf.is_fitted)

            orig_probs = clf.predict_proba(self.X)
            loaded_probs = loaded_clf.predict_proba(self.X)
            np.testing.assert_allclose(orig_probs, loaded_probs, atol=1e-5)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_fallback_when_unfitted(self) -> None:
        clf = LearnedClusterClassifier()
        # Predicting before fit should safely return 1.0 (keep candidate)
        dummy_feat = np.zeros(len(FEATURE_NAMES))
        prob = clf.predict_single(dummy_feat)
        self.assertEqual(prob, 1.0)


class TestLearnedNarrationClassifier(unittest.TestCase):
    def test_narration_classifier_fit_and_predict(self) -> None:
        np.random.seed(42)
        n = 50
        X_narr = np.zeros((n, len(FEATURE_NAMES)))
        X_narr[:, FEATURE_NAMES.index("aspect_ratio")] = np.random.uniform(3.0, 6.0, n)
        X_narr[:, FEATURE_NAMES.index("char_count")] = np.random.uniform(40, 120, n)
        X_narr[:, FEATURE_NAMES.index("margin_dist_min")] = np.random.uniform(0.01, 0.08, n)

        X_dial = np.zeros((n, len(FEATURE_NAMES)))
        X_dial[:, FEATURE_NAMES.index("aspect_ratio")] = np.random.uniform(0.8, 1.8, n)
        X_dial[:, FEATURE_NAMES.index("char_count")] = np.random.uniform(10, 50, n)
        X_dial[:, FEATURE_NAMES.index("margin_dist_min")] = np.random.uniform(0.15, 0.4, n)

        X = np.vstack([X_narr, X_dial])
        y = np.hstack([np.ones(n), np.zeros(n)])

        clf = LearnedNarrationClassifier(threshold=0.5)
        clf.fit(X, y)
        self.assertTrue(clf.is_fitted)

        probs = clf.predict_proba(X)
        self.assertGreaterEqual(np.mean((probs >= 0.5) == y), 0.90)


class MockOCREngine:
    def __init__(self, regions: list) -> None:
        self.regions = regions

    def detect_and_recognize(self, image: Any) -> list:
        return self.regions


class TestPredictorIntegration(unittest.TestCase):
    def test_predictor_filters_noise_and_assigns_narration(self) -> None:
        from PIL import Image
        from src.models.baseline_predictor import RapidOCRBaselinePredictor

        # Synthetic image
        img = Image.new("RGB", (800, 1200), color="white")

        # Mock OCR output: 1 dialogue, 1 noise, 1 narration
        dial_bbox = BoundingBox(xmin=200, ymin=300, xmax=400, ymax=450)
        noise_bbox = BoundingBox(xmin=10, ymin=1150, xmax=40, ymax=1180)
        narr_bbox = BoundingBox(xmin=50, ymin=50, xmax=750, ymax=120)

        dial_reg = DetectedRegion(polygon=[], bbox=dial_bbox, text="HELLO THERE! HOW ARE YOU?", confidence=0.95)
        noise_reg = DetectedRegion(polygon=[], bbox=noise_bbox, text="84", confidence=0.85)
        narr_reg = DetectedRegion(polygon=[], bbox=narr_bbox, text="MEANWHILE, ACROSS THE CITY...", confidence=0.92)

        mock_ocr = MockOCREngine([dial_reg, noise_reg, narr_reg])

        # Fitted filter classifier that filters noise
        filter_clf = LearnedClusterClassifier(decision_threshold=0.5)
        filter_clf.weights = np.zeros(len(FEATURE_NAMES))
        filter_clf.weights[FEATURE_NAMES.index("digit_ratio")] = -5.0
        filter_clf.weights[FEATURE_NAMES.index("char_count")] = 1.0
        filter_clf.bias = 0.0
        filter_clf.mean = np.zeros(len(FEATURE_NAMES))
        filter_clf.std = np.ones(len(FEATURE_NAMES))
        filter_clf.is_fitted = True

        # Fitted narration classifier
        narr_clf = LearnedNarrationClassifier(threshold=0.5)
        narr_clf.weights = np.zeros(len(FEATURE_NAMES))
        narr_clf.weights[FEATURE_NAMES.index("aspect_ratio")] = 5.0
        narr_clf.bias = -10.0
        narr_clf.mean = np.zeros(len(FEATURE_NAMES))
        narr_clf.std = np.ones(len(FEATURE_NAMES))
        narr_clf.is_fitted = True

        predictor = RapidOCRBaselinePredictor(
            ocr_engine=mock_ocr,
            enable_clustering=True,
            reading_order_mode="panel_aware",
            enable_dialogue_filter=True,
            dialogue_classifier=filter_clf,
            enable_narration_classifier=True,
            narration_classifier=narr_clf,
        )

        output = predictor.predict_images([img])
        self.assertEqual(len(output), 1)
        page_lines = output[0]

        # "84" noise should be filtered out!
        texts = [line["text"] for line in page_lines]
        self.assertNotIn("84", texts)
        self.assertIn("HELLO THERE! HOW ARE YOU?", texts)
        self.assertIn("MEANWHILE, ACROSS THE CITY...", texts)

        # Narration should have speaker "NARRATION"
        for line in page_lines:
            if "MEANWHILE" in line["text"]:
                self.assertEqual(line["speaker"], "NARRATION")
            elif "HELLO" in line["text"]:
                self.assertEqual(line["speaker"], "character1")


if __name__ == "__main__":
    unittest.main()

