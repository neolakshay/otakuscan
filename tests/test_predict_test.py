import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image

from src.data.loader import SequenceRecord, OtakuDatasetLoader
from src.models.reading_order import BoundingBox, DetectedRegion

# Import functions from scripts.predict_test to test TDD cycle
from scripts.predict_test import (
    discover_test_sequences,
    load_adapted_predictor,
    validate_test_predictions,
    atomic_write_jsonl,
    generate_test_predictions,
)


class TestPredictTestSequenceDiscovery(unittest.TestCase):
    def test_discover_test_sequences_filters_correctly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            seq_file = tmp_path / "sequences.json"
            seq_data = [
                {"sequence_id": "seq_dev_1", "split": "development", "images": ["dev/1.png", "dev/2.png", "dev/3.png"]},
                {"sequence_id": "seq_test_1", "split": "test", "images": ["test/1.png", "test/2.png", "test/3.png"]},
                {"sequence_id": "seq_test_2", "split": "test", "images": ["test/4.png", "test/5.png", "test/6.png"]},
            ]
            seq_file.write_text(json.dumps(seq_data), encoding="utf-8")

            discovered = discover_test_sequences(dataset_root=tmp_path)
            self.assertEqual(len(discovered), 2)
            self.assertEqual([s.sequence_id for s in discovered], ["seq_test_1", "seq_test_2"])
            self.assertTrue(all(s.split == "test" for s in discovered))

    def test_discover_test_sequences_no_test_sequences_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            seq_file = tmp_path / "sequences.json"
            seq_data = [
                {"sequence_id": "seq_dev_1", "split": "development", "images": ["dev/1.png", "dev/2.png", "dev/3.png"]},
            ]
            seq_file.write_text(json.dumps(seq_data), encoding="utf-8")

            with self.assertRaises(ValueError) as ctx:
                discover_test_sequences(dataset_root=tmp_path)
            self.assertIn("No test sequences found", str(ctx.exception))

    def test_discover_test_sequences_missing_metadata_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                discover_test_sequences(dataset_root=Path(tmpdir))


class TestPredictTestPredictorLoading(unittest.TestCase):
    def test_load_adapted_predictor_missing_weights_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            with self.assertRaises(FileNotFoundError) as ctx:
                load_adapted_predictor(weights_dir=tmp_path)
            self.assertIn("Missing required model weights", str(ctx.exception))

    def test_load_adapted_predictor_real_weights_configuration(self):
        repo_root = Path(__file__).resolve().parent.parent
        weights_dir = repo_root / "models"
        if (weights_dir / "dialogue_filter_weights.json").is_file():
            predictor = load_adapted_predictor(weights_dir=weights_dir)
            self.assertTrue(predictor.enable_clustering)
            self.assertEqual(predictor.reading_order_mode, "panel_aware")
            self.assertTrue(predictor.enable_dialogue_filter)
            self.assertIsNotNone(predictor.dialogue_classifier)
            self.assertEqual(predictor.dialogue_classifier.decision_threshold, 0.35)
            self.assertTrue(predictor.enable_narration_classifier)
            self.assertIsNotNone(predictor.narration_classifier)
            self.assertEqual(predictor.narration_classifier.decision_threshold, 0.25)
            self.assertEqual(predictor.min_confidence, 0.0)


class TestPredictTestValidationLogic(unittest.TestCase):
    def setUp(self):
        self.test_sequences = [
            SequenceRecord(sequence_id="seq_test_1", split="test", image_paths=[Path("1"), Path("2"), Path("3")]),
            SequenceRecord(sequence_id="seq_test_2", split="test", image_paths=[Path("4"), Path("5"), Path("6")]),
        ]
        self.dev_ids = {"seq_dev_1"}

    def test_validate_test_predictions_valid(self):
        valid_predictions = [
            {
                "sequence_id": "seq_test_1",
                "pages": [
                    [{"speaker": "character1", "text": "Hello!"}],
                    [{"speaker": "NARRATION", "text": "Later that night..."}],
                    [],
                ],
            },
            {
                "sequence_id": "seq_test_2",
                "pages": [[], [], []],
            },
        ]
        # Should pass without raising
        validate_test_predictions(
            predictions=valid_predictions,
            expected_test_sequences=self.test_sequences,
            dev_sequence_ids=self.dev_ids,
        )

    def test_validate_test_predictions_missing_sequence(self):
        incomplete = [
            {"sequence_id": "seq_test_1", "pages": [[], [], []]},
        ]
        with self.assertRaises(ValueError) as ctx:
            validate_test_predictions(
                predictions=incomplete,
                expected_test_sequences=self.test_sequences,
                dev_sequence_ids=self.dev_ids,
            )
        self.assertIn("Missing test sequence", str(ctx.exception))

    def test_validate_test_predictions_includes_dev_sequence(self):
        contaminated = [
            {"sequence_id": "seq_test_1", "pages": [[], [], []]},
            {"sequence_id": "seq_test_2", "pages": [[], [], []]},
            {"sequence_id": "seq_dev_1", "pages": [[], [], []]},
        ]
        with self.assertRaises(ValueError) as ctx:
            validate_test_predictions(
                predictions=contaminated,
                expected_test_sequences=self.test_sequences,
                dev_sequence_ids=self.dev_ids,
            )
        self.assertIn("Development sequence", str(ctx.exception))

    def test_validate_test_predictions_duplicate_sequence(self):
        duplicated = [
            {"sequence_id": "seq_test_1", "pages": [[], [], []]},
            {"sequence_id": "seq_test_1", "pages": [[], [], []]},
        ]
        with self.assertRaises(ValueError) as ctx:
            validate_test_predictions(
                predictions=duplicated,
                expected_test_sequences=self.test_sequences,
                dev_sequence_ids=self.dev_ids,
            )
        self.assertIn("Duplicate", str(ctx.exception))

    def test_validate_test_predictions_wrong_page_count(self):
        wrong_pages = [
            {"sequence_id": "seq_test_1", "pages": [[], []]},
            {"sequence_id": "seq_test_2", "pages": [[], [], []]},
        ]
        with self.assertRaises(ValueError) as ctx:
            validate_test_predictions(
                predictions=wrong_pages,
                expected_test_sequences=self.test_sequences,
                dev_sequence_ids=self.dev_ids,
            )
        self.assertIn("exactly 3 page entries", str(ctx.exception))

    def test_validate_test_predictions_page_not_list(self):
        invalid_page = [
            {"sequence_id": "seq_test_1", "pages": ["not a list", [], []]},
            {"sequence_id": "seq_test_2", "pages": [[], [], []]},
        ]
        with self.assertRaises(ValueError) as ctx:
            validate_test_predictions(
                predictions=invalid_page,
                expected_test_sequences=self.test_sequences,
                dev_sequence_ids=self.dev_ids,
            )
        self.assertIn("must be a list", str(ctx.exception))

    def test_validate_test_predictions_invalid_speaker_or_text(self):
        empty_speaker = [
            {"sequence_id": "seq_test_1", "pages": [[{"speaker": "", "text": "Hi"}], [], []]},
            {"sequence_id": "seq_test_2", "pages": [[], [], []]},
        ]
        with self.assertRaises(ValueError) as ctx:
            validate_test_predictions(
                predictions=empty_speaker,
                expected_test_sequences=self.test_sequences,
                dev_sequence_ids=self.dev_ids,
            )
        self.assertIn("speaker", str(ctx.exception).lower())

        empty_text = [
            {"sequence_id": "seq_test_1", "pages": [[{"speaker": "character1", "text": "   "}], [], []]},
            {"sequence_id": "seq_test_2", "pages": [[], [], []]},
        ]
        with self.assertRaises(ValueError) as ctx:
            validate_test_predictions(
                predictions=empty_text,
                expected_test_sequences=self.test_sequences,
                dev_sequence_ids=self.dev_ids,
            )
        self.assertIn("text", str(ctx.exception).lower())

    def test_validate_test_predictions_invalid_narration_casing(self):
        bad_narration = [
            {"sequence_id": "seq_test_1", "pages": [[{"speaker": "narration", "text": "Hi"}], [], []]},
            {"sequence_id": "seq_test_2", "pages": [[], [], []]},
        ]
        with self.assertRaises(ValueError) as ctx:
            validate_test_predictions(
                predictions=bad_narration,
                expected_test_sequences=self.test_sequences,
                dev_sequence_ids=self.dev_ids,
            )
        self.assertIn("exact speaker label 'NARRATION'", str(ctx.exception))


class TestPredictTestAtomicOutput(unittest.TestCase):
    def test_atomic_write_inside_dataset_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "dataset"
            dataset_root.mkdir()
            output_in_dataset = dataset_root / "out.jsonl"
            with self.assertRaises(ValueError) as ctx:
                atomic_write_jsonl(
                    predictions=[{"sequence_id": "s1", "pages": [[], [], []]}],
                    output_path=output_in_dataset,
                    dataset_root=dataset_root,
                )
            self.assertIn("outside dataset", str(ctx.exception).lower())

    def test_atomic_write_success_and_parse_back(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "sub" / "output.jsonl"
            preds = [
                {"sequence_id": "s1", "pages": [[{"speaker": "character1", "text": "T1"}], [], []]},
                {"sequence_id": "s2", "pages": [[], [], []]},
            ]
            atomic_write_jsonl(predictions=preds, output_path=out_file)
            self.assertTrue(out_file.is_file())

            # Parse back
            parsed = []
            with open(out_file, "r", encoding="utf-8") as f:
                for line in f:
                    parsed.append(json.loads(line))
            self.assertEqual(parsed, preds)

    def test_atomic_write_preserves_existing_on_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "output.jsonl"
            out_file.write_text('{"original": true}\n', encoding="utf-8")

            # Try to write an un-serializable object
            with self.assertRaises(Exception):
                atomic_write_jsonl(
                    predictions=[{"unserializable": object()}],
                    output_path=out_file,
                )

            # Check original content is preserved
            self.assertEqual(out_file.read_text(encoding="utf-8"), '{"original": true}\n')


class TestPredictTestInferenceSynthetic(unittest.TestCase):
    def test_generate_test_predictions_synthetic_mock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dataset_dir = tmp_path / "dataset"
            dataset_dir.mkdir()
            test_img_dir = dataset_dir / "test" / "images" / "seq_test_syn"
            test_img_dir.mkdir(parents=True)

            img_paths = []
            for i in range(1, 4):
                img = Image.new("RGB", (100, 100), color=(255, 255, 255))
                p = test_img_dir / f"0{i}.png"
                img.save(p)
                img_paths.append(f"test/images/seq_test_syn/0{i}.png")

            seq_file = dataset_dir / "sequences.json"
            seq_file.write_text(json.dumps([
                {"sequence_id": "seq_test_syn", "split": "test", "images": img_paths}
            ]), encoding="utf-8")

            # Mock predictor
            mock_predictor = MagicMock()
            mock_predictor.predict_images.return_value = [
                [{"speaker": "character1", "text": "SYNTHETIC DIALOGUE"}],
                [],
                [],
            ]

            out_file = tmp_path / "predictions" / "test_preds.jsonl"
            result = generate_test_predictions(
                dataset_root=dataset_dir,
                output_path=out_file,
                predictor=mock_predictor,
            )

            self.assertTrue(out_file.is_file())
            self.assertEqual(result["sequence_count"], 1)
            self.assertEqual(result["total_pages"], 3)

            # Check output contents
            with open(out_file, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f]
            self.assertEqual(len(lines), 1)
            self.assertEqual(lines[0]["sequence_id"], "seq_test_syn")
            self.assertEqual(lines[0]["pages"][0][0]["text"], "SYNTHETIC DIALOGUE")

    def test_generate_test_predictions_fails_when_image_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dataset_dir = tmp_path / "dataset"
            dataset_dir.mkdir()

            seq_file = dataset_dir / "sequences.json"
            seq_file.write_text(json.dumps([
                {"sequence_id": "seq_missing_img", "split": "test", "images": ["test/missing1.png", "test/missing2.png", "test/missing3.png"]}
            ]), encoding="utf-8")

            mock_predictor = MagicMock()
            out_file = tmp_path / "predictions" / "test_preds.jsonl"

            with self.assertRaises(FileNotFoundError):
                generate_test_predictions(
                    dataset_root=dataset_dir,
                    output_path=out_file,
                    predictor=mock_predictor,
                )
            self.assertFalse(out_file.is_file())


class TestModelPackaging(unittest.TestCase):
    def test_models_archive_integrity_and_extractability(self):
        import tarfile
        repo_root = Path(__file__).resolve().parent.parent
        archive_path = repo_root / "models_archive.tar.gz"
        self.assertTrue(archive_path.is_file(), f"Archive not found at {archive_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            with tarfile.open(archive_path, "r:gz") as tar:
                try:
                    tar.extractall(tmp_path, filter="data")
                except TypeError:
                    tar.extractall(tmp_path)

            extracted_models_dir = tmp_path / "models"
            self.assertTrue(extracted_models_dir.is_dir())
            f_weights = extracted_models_dir / "dialogue_filter_weights.json"
            n_weights = extracted_models_dir / "narration_classifier_weights.json"
            self.assertTrue(f_weights.is_file())
            self.assertTrue(n_weights.is_file())

            # Verify that load_adapted_predictor accepts the extracted directory
            predictor = load_adapted_predictor(weights_dir=extracted_models_dir)
            self.assertIsNotNone(predictor.dialogue_classifier)
            self.assertIsNotNone(predictor.narration_classifier)


if __name__ == "__main__":
    unittest.main()

