import unittest
from pathlib import Path

from src.eval.scorer import (
    validate_submission_data,
    SubmissionValidationError,
    OtakuScorerBridge,
)

class TestOtakuScorerBridge(unittest.TestCase):
    def setUp(self):
        self.dataset_root = Path(__file__).resolve().parent.parent / "dataset"
        self.bridge = OtakuScorerBridge(dataset_root=self.dataset_root)

    def test_valid_submission_passes_validation(self):
        """A properly formatted submission dictionary should validate cleanly."""
        valid_data = [
            {
                "sequence_id": "seq_sample1",
                "pages": [
                    [{"speaker": "character1", "text": "HELLO WORLD"}],
                    [],
                    [{"speaker": "NARRATION", "text": "THE NEXT DAY..."}]
                ]
            }
        ]
        # Should not raise
        validate_submission_data(valid_data, expected_pages=3)

    def test_reject_wrong_page_count(self):
        """Submission with fewer or more than 3 pages must raise SubmissionValidationError."""
        invalid_data = [
            {
                "sequence_id": "seq_sample1",
                "pages": [[], []]  # only 2 pages
            }
        ]
        with self.assertRaises(SubmissionValidationError):
            validate_submission_data(invalid_data, expected_pages=3)

    def test_reject_empty_speaker_or_text(self):
        """Submission with whitespace-only speaker or text must raise error."""
        invalid_speaker = [
            {
                "sequence_id": "seq_sample1",
                "pages": [[{"speaker": "   ", "text": "HELLO"}], [], []]
            }
        ]
        with self.assertRaises(SubmissionValidationError):
            validate_submission_data(invalid_speaker)

        invalid_text = [
            {
                "sequence_id": "seq_sample1",
                "pages": [[{"speaker": "character1", "text": "   "}], [], []]
            }
        ]
        with self.assertRaises(SubmissionValidationError):
            validate_submission_data(invalid_text)

    def test_reject_extra_keys(self):
        """Items or rows with disallowed extra keys must fail validation."""
        extra_key_data = [
            {
                "sequence_id": "seq_sample1",
                "pages": [[], [], []],
                "extra": 123
            }
        ]
        with self.assertRaises(SubmissionValidationError):
            validate_submission_data(extra_key_data)

    def test_score_ground_truth_against_itself(self):
        """Official scorer bridge evaluates ground truth against itself to yield 1.0."""
        ref_path = self.dataset_root / "development" / "labels.jsonl"
        report = self.bridge.evaluate_jsonl(reference_path=ref_path, prediction_path=ref_path)
        macro = report["macro"]
        self.assertAlmostEqual(macro["text_order_score"], 1.0, places=4)
        self.assertAlmostEqual(macro["balanced_joint_f1"], 1.0, places=4)
        self.assertAlmostEqual(macro["matched_token_coverage"], 1.0, places=4)

    def test_score_empty_predictions_against_dev(self):
        """Empty predictions should yield 0.0 macro score for active pages."""
        ref_path = self.dataset_root / "development" / "labels.jsonl"
        # 80 empty sequences
        sequences = self.bridge.load_reference_jsonl(ref_path)
        empty_preds = {seq_id: [[], [], []] for seq_id in sequences}
        report = self.bridge.evaluate_dict(references=sequences, predictions=empty_preds)
        macro = report["macro"]
        self.assertAlmostEqual(macro["text_order_score"], 0.0, places=4)
        self.assertAlmostEqual(macro["balanced_joint_f1"], 0.0, places=4)

if __name__ == "__main__":
    unittest.main()
