from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.models.segmentation import WordSegmenter, DEFAULT_COMMON_WORDS


class TestWordSegmenter(unittest.TestCase):
    """Unit tests for WordSegmenter."""

    def setUp(self) -> None:
        self.vocab = {
            "the": 500,
            "new": 200,
            "so": 300,
            "cool": 150,
            "back": 120,
            "in": 400,
            "together": 50,
            "today": 60,
            "understand": 40,
            "another": 45,
            "normal": 80,
            "a": 600,
            "man": 100,
            "look": 150,
            "wait": 80,
            "what": 250,
            "really": 110,
        }
        self.segmenter = WordSegmenter(vocabulary=self.vocab, split_penalty=2.5)

    def test_missing_space_recovery(self) -> None:
        """Verifies multi-word concatenations are correctly split."""
        self.assertEqual(self.segmenter.segment_token("socool"), "so cool")
        self.assertEqual(self.segmenter.segment_token("thenew"), "the new")
        self.assertEqual(self.segmenter.segment_token("anormal"), "a normal")

    def test_casing_preservation(self) -> None:
        """Verifies original uppercase / capitalized styling is preserved."""
        self.assertEqual(self.segmenter.segment_token("SOCOOL"), "SO COOL")
        self.assertEqual(self.segmenter.segment_token("TheNew"), "The New")

    def test_valid_word_preservation(self) -> None:
        """Verifies valid vocabulary words are not split into erroneous fragments."""
        self.assertEqual(self.segmenter.segment_token("together"), "together")
        self.assertEqual(self.segmenter.segment_token("today"), "today")
        self.assertEqual(self.segmenter.segment_token("understand"), "understand")
        self.assertEqual(self.segmenter.segment_token("another"), "another")

    def test_contractions_preserved(self) -> None:
        """Verifies contractions with apostrophes are left completely intact."""
        self.assertEqual(self.segmenter.segment_token("don't"), "don't")
        self.assertEqual(self.segmenter.segment_token("it's"), "it's")
        self.assertEqual(self.segmenter.segment_token("I'm"), "I'm")
        self.assertEqual(self.segmenter.segment_token("can't"), "can't")

    def test_punctuation_spacing_in_text(self) -> None:
        """Verifies punctuation immediately followed by letters gets a clean space."""
        text = "HUH?DOES KOMABA SEEM DIFFERENT?"
        segmented = self.segmenter.segment_text(text)
        self.assertIn("HUH? DOES", segmented)

        text2 = "WAIT,LOOK AT THAT!"
        segmented2 = self.segmenter.segment_text(text2)
        self.assertIn("WAIT, LOOK", segmented2)

    def test_sound_effects_and_short_tokens(self) -> None:
        """Verifies short words and unknown sound effects remain untouched."""
        self.assertEqual(self.segmenter.segment_token("hi"), "hi")
        self.assertEqual(self.segmenter.segment_token("a"), "a")
        self.assertEqual(self.segmenter.segment_token(""), "")
        self.assertEqual(self.segmenter.segment_token("kyaa"), "kyaa")
        self.assertEqual(self.segmenter.segment_token("booom"), "booom")

    def test_serialization_roundtrip(self) -> None:
        """Verifies save_to_json and load_from_json produce identical behavior."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "vocab.json"
            self.segmenter.save_to_json(tmp_path)
            loaded = WordSegmenter.load_from_json(tmp_path)

            self.assertEqual(len(loaded.vocabulary), len(self.segmenter.vocabulary))
            self.assertEqual(loaded.segment_token("socool"), "so cool")
            self.assertEqual(loaded.segment_token("together"), "together")

    def test_leakage_assertion_on_fit(self) -> None:
        """Verifies fit_from_training_sequences raises ValueError if training overlaps forbidden IDs."""
        seg = WordSegmenter()
        with self.assertRaises(ValueError):
            seg.fit_from_training_sequences(
                golds_dict={},
                allowed_train_ids={"seq_train_1", "seq_val_1"},
                forbidden_val_test_ids={"seq_val_1", "seq_val_2"},
            )


if __name__ == "__main__":
    unittest.main()
