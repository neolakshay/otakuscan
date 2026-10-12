from __future__ import annotations

import json
import unittest
from pathlib import Path

from colab.vlm_manga_transcription_experiment import (
    parse_vlm_output,
    select_pilot_training_pages,
    compute_transcription_token_match,
)


class TestVLMOutputParsing(unittest.TestCase):
    def test_parse_valid_json_output(self) -> None:
        raw_output = json.dumps([
            {"speaker": "character", "text": "HELLO WORLD!"},
            {"speaker": "NARRATION", "text": "MEANWHILE IN TOKYO..."},
        ])
        parsed = parse_vlm_output(raw_output)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["speaker"], "character")
        self.assertEqual(parsed[0]["text"], "HELLO WORLD!")
        self.assertEqual(parsed[1]["speaker"], "NARRATION")
        self.assertEqual(parsed[1]["text"], "MEANWHILE IN TOKYO...")

    def test_parse_markdown_fenced_json(self) -> None:
        raw_output = """Here is the transcription:
```json
[
  {"speaker": "character1", "text": "WHAT ARE YOU DOING?"},
  {"speaker": "NARRATION", "text": "THE NEXT DAY..."}
]
```
Hope this helps!"""
        parsed = parse_vlm_output(raw_output)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["text"], "WHAT ARE YOU DOING?")
        self.assertEqual(parsed[1]["speaker"], "NARRATION")

    def test_parse_fallback_freeform_text(self) -> None:
        raw_output = """[NARRATION] Once upon a time in a distant land.
Mikage: What's wrong?
Pochita: Woof!"""
        parsed = parse_vlm_output(raw_output)
        self.assertEqual(len(parsed), 3)
        self.assertEqual(parsed[0]["speaker"], "NARRATION")
        self.assertIn("Once upon a time", parsed[0]["text"])
        self.assertEqual(parsed[1]["speaker"], "character")
        self.assertIn("What's wrong?", parsed[1]["text"])

    def test_parse_empty_or_whitespace_output(self) -> None:
        self.assertEqual(parse_vlm_output(""), [])
        self.assertEqual(parse_vlm_output("   \n\t  "), [])


class TestPilotSelectionAndLeakage(unittest.TestCase):
    def test_select_pilot_pages_from_train_split_only(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        splits_path = project_root / "splits" / "dev_splits.json"
        with open(splits_path, "r", encoding="utf-8") as f:
            splits_data = json.load(f)

        train_ids = set(splits_data["kfold_5"][0]["train"])
        val_ids = set(splits_data["kfold_5"][0]["val"])

        pilot_items = select_pilot_training_pages(project_root, n_pages=3)
        self.assertEqual(len(pilot_items), 3)

        for item in pilot_items:
            seq_id = item["sequence_id"]
            img_path = Path(item["image_path"])
            # Must strictly belong to training split
            self.assertIn(seq_id, train_ids)
            self.assertNotIn(seq_id, val_ids)
            # Image file must exist
            self.assertTrue(img_path.is_file(), f"Pilot image missing: {img_path}")


class TestTranscriptionMatching(unittest.TestCase):
    def test_compute_token_match(self) -> None:
        predicted = [
            {"speaker": "character", "text": "HELLO THERE HOW ARE YOU"},
            {"speaker": "NARRATION", "text": "MEANWHILE ACROSS TOWN"},
        ]
        ground_truth = [
            {"speaker": "character1", "text": "HELLO THERE! HOW ARE YOU?"},
            {"speaker": "NARRATION", "text": "MEANWHILE ACROSS TOWN."},
        ]

        metrics = compute_transcription_token_match(predicted, ground_truth)
        self.assertGreaterEqual(metrics["token_precision"], 0.90)
        self.assertGreaterEqual(metrics["token_recall"], 0.90)
        self.assertGreaterEqual(metrics["token_f1"], 0.90)


if __name__ == "__main__":
    unittest.main()
