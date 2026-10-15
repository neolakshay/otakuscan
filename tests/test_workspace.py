from __future__ import annotations

import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from src.data.workspace import (
    validate_dataset_directory,
    is_valid_dataset_directory,
    extract_dataset_archive,
)


class TestWorkspaceDatasetValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.project_root = Path(__file__).resolve().parent.parent

    def test_validate_existing_project_dataset(self) -> None:
        dataset_path = self.project_root / "dataset"
        resolved = validate_dataset_directory(dataset_path)
        self.assertEqual(resolved, dataset_path.resolve())
        self.assertTrue(is_valid_dataset_directory(dataset_path))

    def test_validate_missing_directory_raises_error(self) -> None:
        with self.assertRaises(FileNotFoundError) as ctx:
            validate_dataset_directory("/non/existent/path/dataset")
        self.assertIn("does not exist", str(ctx.exception))

    def test_validate_empty_directory_raises_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(FileNotFoundError) as ctx:
                validate_dataset_directory(tmp_dir)
            self.assertIn("sequences.json", str(ctx.exception))

    def test_validate_nested_dataset_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            nested = tmp_path / "dataset"
            nested.mkdir()
            (nested / "sequences.json").write_text("[]", encoding="utf-8")
            dev_dir = nested / "development"
            dev_dir.mkdir()
            (dev_dir / "labels.jsonl").write_text("", encoding="utf-8")

            resolved = validate_dataset_directory(tmp_path)
            self.assertEqual(resolved, nested.resolve())
            self.assertTrue(is_valid_dataset_directory(tmp_path))


class TestDatasetExtractionIdempotency(unittest.TestCase):
    def test_extract_and_reuse_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            archive_path = tmp_path / "otakuscan-dataset.tar.gz"
            extract_target = tmp_path / "extracted_data"

            # Create mock tar.gz
            content_dir = tmp_path / "mock_dataset"
            content_dir.mkdir()
            (content_dir / "sequences.json").write_text("[]", encoding="utf-8")
            dev_dir = content_dir / "development"
            dev_dir.mkdir()
            (dev_dir / "labels.jsonl").write_text("", encoding="utf-8")

            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(content_dir / "sequences.json", arcname="sequences.json")
                tar.add(dev_dir, arcname="development")

            # First extraction
            resolved_1 = extract_dataset_archive(archive_path, extract_target, force=False)
            self.assertTrue((resolved_1 / "sequences.json").is_file())

            # Place a canary file to prove idempotency (not deleted or overwritten)
            canary = resolved_1 / "canary.txt"
            canary.write_text("reused", encoding="utf-8")

            # Second call should reuse without re-extracting
            resolved_2 = extract_dataset_archive(archive_path, extract_target, force=False)
            self.assertEqual(resolved_1, resolved_2)
            self.assertTrue(canary.is_file(), "Extraction was not idempotent; canary file lost!")
            self.assertEqual(canary.read_text(), "reused")


if __name__ == "__main__":
    unittest.main()
