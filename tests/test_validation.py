import unittest
import tempfile
from pathlib import Path

from src.data.loader import OtakuDatasetLoader
from src.pipeline.validation import (
    create_sequence_train_val_split,
    create_sequence_kfold_splits,
    ValidationPipeline,
)

class TestValidationPipeline(unittest.TestCase):
    def setUp(self):
        self.dataset_root = Path(__file__).resolve().parent.parent / "dataset"
        self.loader = OtakuDatasetLoader(dataset_root=self.dataset_root)
        self.dev_sequences = [s for s in self.loader.load_sequences() if s.split == "development"]

    def test_train_val_split_proportions_and_invariance(self):
        """Train/val split must keep every 3-page sequence intact with zero overlap."""
        val_ratio = 0.2
        split = create_sequence_train_val_split(
            sequences=self.dev_sequences,
            val_ratio=val_ratio,
            seed=42
        )
        self.assertIn("train", split)
        self.assertIn("val", split)
        train_ids = set(split["train"])
        val_ids = set(split["val"])

        # Check total sequences
        self.assertEqual(len(train_ids) + len(val_ids), 80)
        # Check mutual exclusivity (zero leakage across splits)
        self.assertEqual(len(train_ids.intersection(val_ids)), 0)
        # Check exact counts (64 train, 16 val for 80 sequences at 0.2 ratio)
        self.assertEqual(len(val_ids), 16)
        self.assertEqual(len(train_ids), 64)

    def test_kfold_splits_coverage(self):
        """K-fold splits must partition all 80 development sequences cleanly into K disjoint folds."""
        k = 5
        folds = create_sequence_kfold_splits(sequences=self.dev_sequences, n_splits=k, seed=42)
        self.assertEqual(len(folds), k)

        all_val_ids = set()
        for fold_idx, fold in enumerate(folds):
            train_set = set(fold["train"])
            val_set = set(fold["val"])
            self.assertEqual(len(train_set.intersection(val_set)), 0)
            self.assertEqual(len(train_set) + len(val_set), 80)
            self.assertEqual(len(val_set), 16)  # 80 / 5 = 16
            all_val_ids.update(val_set)

        # Union of all validation folds should exactly equal all 80 dev sequences
        self.assertEqual(len(all_val_ids), 80)

    def test_validation_pipeline_runs_and_scores(self):
        """Pipeline must evaluate a predictor on a validation fold and return metric scores."""
        pipeline = ValidationPipeline(dataset_root=self.dataset_root)
        val_split = create_sequence_train_val_split(self.dev_sequences, val_ratio=0.2, seed=42)
        
        # Test with empty baseline predictor: predict empty pages for every sequence
        def dummy_empty_predictor(sequence):
            return [[], [], []]

        eval_report = pipeline.evaluate_split(
            val_sequence_ids=val_split["val"],
            predictor=dummy_empty_predictor
        )
        self.assertIn("macro", eval_report)
        macro = eval_report["macro"]
        self.assertIn("balanced_joint_f1", macro)
        self.assertIn("text_order_score", macro)

    def test_split_export_writes_outside_dataset(self):
        """Exporting splits must only write outside dataset/ (e.g. to a specified target directory)."""
        pipeline = ValidationPipeline(dataset_root=self.dataset_root)
        val_split = create_sequence_train_val_split(self.dev_sequences, val_ratio=0.2, seed=42)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "splits.json"
            pipeline.export_split_definition(val_split, out_path)
            self.assertTrue(out_path.is_file())

if __name__ == "__main__":
    unittest.main()
