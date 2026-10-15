from __future__ import annotations

import json
import unittest
from pathlib import Path
import numpy as np

from src.models.reading_order import BoundingBox, DetectedRegion
from src.models.clustering import TextCluster
from src.models.filtering import (
    FEATURE_NAMES,
    NARRATION_FEATURE_NAMES,
    extract_cluster_features,
    extract_narration_features,
    LearnedNarrationClassifier,
)
from src.models.baseline_predictor import RapidOCRBaselinePredictor
from PIL import Image


class TestNarrationClassifierRegressions(unittest.TestCase):
    def setUp(self) -> None:
        self.project_root = Path(__file__).resolve().parent.parent
        self.pw = 800.0
        self.ph = 1200.0
        self.default_weights_path = self.project_root / "models" / "narration_classifier_weights.json"

    def _make_cluster(self, text: str, bbox: BoundingBox) -> TextCluster:
        reg = DetectedRegion(polygon=[], bbox=bbox, text=text, confidence=0.95)
        return TextCluster(regions=[reg], bbox=bbox, text=text, confidence=0.95)

    def test_preservation_of_original_default_configuration(self) -> None:
        """Verifies that the original baseline classifier weights and configuration remain intact."""
        self.assertTrue(self.default_weights_path.is_file(), "Original narration classifier weights file missing!")
        clf = LearnedNarrationClassifier.load_from_json(self.default_weights_path)
        self.assertTrue(clf.is_fitted)
        self.assertEqual(len(clf.feature_names), 20)
        self.assertEqual(clf.feature_names, FEATURE_NAMES)
        self.assertAlmostEqual(clf.bias, -3.2302, places=3)
        self.assertEqual(clf.l2_reg, 0.01)

    def test_deterministic_classifier_output(self) -> None:
        """Verifies that the classifier yields bitwise deterministic probability outputs across runs."""
        clf = LearnedNarrationClassifier.load_from_json(self.default_weights_path)
        bbox = BoundingBox(xmin=100, ymin=100, xmax=700, ymax=180)
        cluster = self._make_cluster("IN MATTERS OF LOVE, THERE IS AN ABSOLUTE RULE.", bbox)
        feats = clf.extract_features(cluster, self.pw, self.ph)

        prob1 = clf.predict_single(feats)
        prob2 = clf.predict_single(feats)
        self.assertEqual(prob1, prob2)

    def test_dialogue_containing_past_tense_verbs(self) -> None:
        """Verifies dialogue with past-tense verbs is not blindly treated as narration."""
        clf = LearnedNarrationClassifier.load_from_json(self.default_weights_path)
        bbox = BoundingBox(xmin=200, ymin=400, xmax=450, ymax=600)
        dial_cluster = self._make_cluster("I was wondering if he had arrived yet.", bbox)
        feats = clf.extract_features(dial_cluster, self.pw, self.ph)
        prob = clf.predict_single(feats)
        # Should remain far below narration threshold
        self.assertLess(prob, 0.25)

    def test_dialogue_without_pronouns(self) -> None:
        """Verifies imperative dialogue lacking pronouns is not misclassified as narration."""
        clf = LearnedNarrationClassifier.load_from_json(self.default_weights_path)
        bbox = BoundingBox(xmin=150, ymin=300, xmax=350, ymax=500)
        cluster = self._make_cluster("STOP RIGHT THERE! DROP THE WEAPON!", bbox)
        feats = clf.extract_features(cluster, self.pw, self.ph)
        prob = clf.predict_single(feats)
        self.assertLess(prob, 0.25)

    def test_questions_and_short_replies(self) -> None:
        """Verifies questions and conversational replies are safely kept as dialogue."""
        clf = LearnedNarrationClassifier.load_from_json(self.default_weights_path)
        bbox = BoundingBox(xmin=250, ymin=250, xmax=400, ymax=350)

        for text in ["Are you ready?", "What happened?!", "Yes, absolutely.", "Huh?"]:
            cluster = self._make_cluster(text, bbox)
            feats = clf.extract_features(cluster, self.pw, self.ph)
            prob = clf.predict_single(feats)
            self.assertLess(prob, 0.25)

    def test_all_uppercase_dialogue(self) -> None:
        """Verifies shouty all-caps dialogue is not falsely flagged as a caption title."""
        clf = LearnedNarrationClassifier.load_from_json(self.default_weights_path)
        bbox = BoundingBox(xmin=300, ymin=300, xmax=550, ymax=500)
        cluster = self._make_cluster("GET OUT OF THE WAY RIGHT NOW!!", bbox)
        feats = clf.extract_features(cluster, self.pw, self.ph)
        prob = clf.predict_single(feats)
        self.assertLess(prob, 0.25)

    def test_short_reaction_bubbles(self) -> None:
        """Verifies short reaction bubbles are not classified as narration."""
        clf = LearnedNarrationClassifier.load_from_json(self.default_weights_path)
        bbox = BoundingBox(xmin=100, ymin=100, xmax=180, ymax=180)
        for text in ["AH!", "EH?", "...", "NO!"]:
            cluster = self._make_cluster(text, bbox)
            feats = clf.extract_features(cluster, self.pw, self.ph)
            prob = clf.predict_single(feats)
            self.assertLess(prob, 0.25)

    def test_genuine_narration_captions(self) -> None:
        """Verifies that genuine wide banner narration captions are supported."""
        bbox = BoundingBox(xmin=50, ymin=50, xmax=750, ymax=120)
        cluster = self._make_cluster("THE NEXT DAY, AT SHUCHI'IN PRIVATE ACADEMY...", bbox)
        feats_base = extract_cluster_features(cluster, self.pw, self.ph)
        feats_narr = extract_narration_features(cluster, self.pw, self.ph)
        self.assertEqual(len(feats_base), 20)
        self.assertEqual(len(feats_narr), 27)
        # Check narrative setting feature flag
        self.assertEqual(feats_narr[NARRATION_FEATURE_NAMES.index("has_narrative_setting")], 1.0)
        self.assertEqual(feats_narr[NARRATION_FEATURE_NAMES.index("is_wide_aspect_ratio")], 1.0)

    def test_narration_isolation_from_speaker_attribution(self) -> None:
        """Verifies narration lines receive 'NARRATION' and do not disturb conversational turns."""
        img = Image.new("RGB", (800, 1200), color="white")

        class MockEngine:
            def detect_and_recognize(self, _):
                # 1 narration banner at top, followed by 2 dialogue bubbles
                b_narr = BoundingBox(xmin=50, ymin=30, xmax=750, ymax=100)
                b_dial1 = BoundingBox(xmin=500, ymin=200, xmax=700, ymax=350)
                b_dial2 = BoundingBox(xmin=100, ymin=500, xmax=300, ymax=650)
                return [
                    DetectedRegion(polygon=[], bbox=b_narr, text="SUMMER VACATION ENDED.", confidence=0.99),
                    DetectedRegion(polygon=[], bbox=b_dial1, text="HOW ARE YOU FEELING TODAY?", confidence=0.95),
                    DetectedRegion(polygon=[], bbox=b_dial2, text="YES, I AM DOING GREAT!", confidence=0.95),
                ]

        clf_mock = LearnedNarrationClassifier(threshold=0.5)
        # Force narration for banner box
        clf_mock.weights = np.zeros(20)
        clf_mock.weights[FEATURE_NAMES.index("aspect_ratio")] = 10.0
        clf_mock.bias = -20.0
        clf_mock.mean = np.zeros(20)
        clf_mock.std = np.ones(20)
        clf_mock.is_fitted = True

        predictor = RapidOCRBaselinePredictor(
            ocr_engine=MockEngine(),
            default_speaker="character1",
            enable_clustering=True,
            reading_order_mode="panel_aware",
            enable_narration_classifier=True,
            narration_classifier=clf_mock,
            narration_threshold=0.5,
            speaker_attribution_mode="conversational_spatial",
        )

        preds = predictor.predict_images([img])
        self.assertEqual(len(preds), 1)
        items = preds[0]
        self.assertEqual(len(items), 3)
        self.assertEqual(items[0]["speaker"], "NARRATION")
        self.assertEqual(items[1]["speaker"], "character1")
        self.assertIn(items[2]["speaker"], ["character1", "character2"])


if __name__ == "__main__":
    unittest.main()
