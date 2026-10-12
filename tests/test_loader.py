import unittest
from pathlib import Path
from PIL import Image

# Import to be implemented in src/data/loader.py
from src.data.loader import OtakuDatasetLoader, SequenceRecord, DialogueLine

class TestOtakuDatasetLoader(unittest.TestCase):
    def setUp(self):
        self.dataset_root = Path(__file__).resolve().parent.parent / "dataset"
        self.loader = OtakuDatasetLoader(dataset_root=self.dataset_root)

    def test_load_all_sequences(self):
        """Verify that loader reads exactly 95 sequences from sequences.json."""
        sequences = self.loader.load_sequences()
        self.assertEqual(len(sequences), 95)
        
        dev_seqs = [s for s in sequences if s.split == "development"]
        test_seqs = [s for s in sequences if s.split == "test"]
        self.assertEqual(len(dev_seqs), 80)
        self.assertEqual(len(test_seqs), 15)

    def test_sequence_integrity(self):
        """Every sequence must have exactly 3 page images that exist on disk."""
        sequences = self.loader.load_sequences()
        for seq in sequences:
            self.assertIsInstance(seq, SequenceRecord)
            self.assertTrue(seq.sequence_id.startswith("seq_"))
            self.assertEqual(len(seq.image_paths), 3)
            for img_p in seq.image_paths:
                self.assertTrue(img_p.exists(), f"Image path {img_p} does not exist")
                self.assertTrue(img_p.is_file())

    def test_load_development_labels(self):
        """Verify that labels.jsonl is cleanly loaded and mapped to sequences."""
        labels = self.loader.load_labels()
        self.assertEqual(len(labels), 80)
        
        for seq_id, pages in labels.items():
            self.assertTrue(seq_id.startswith("seq_"))
            self.assertEqual(len(pages), 3)
            for page in pages:
                for line in page:
                    self.assertIsInstance(line, DialogueLine)
                    self.assertIsInstance(line.speaker, str)
                    self.assertIsInstance(line.text, str)
                    self.assertTrue(len(line.text.strip()) > 0)

    def test_image_loading_and_rgb_normalization(self):
        """Images must be converted to 3-channel RGB mode even if raw file is P or L."""
        sequences = self.loader.load_sequences()
        sample_seq = sequences[0]
        images = self.loader.load_images(sample_seq, convert_rgb=True)
        self.assertEqual(len(images), 3)
        for img in images:
            self.assertIsInstance(img, Image.Image)
            self.assertEqual(img.mode, "RGB")

    def test_read_only_dataset_guarantee(self):
        """Loader must only perform read operations on dataset root."""
        self.assertEqual(self.loader.dataset_root, self.dataset_root)
        # Verify that dataset root exists and is unchanged
        self.assertTrue(self.dataset_root.is_dir())

if __name__ == "__main__":
    unittest.main()
