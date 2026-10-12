import unittest
from pathlib import Path
from PIL import Image, ImageDraw

from src.data.loader import SequenceRecord
from src.eval.scorer import validate_submission_data
from src.models.ocr import RapidOCREngine
from src.models.baseline_predictor import RapidOCRBaselinePredictor

class TestOCRPipeline(unittest.TestCase):
    def setUp(self):
        self.ocr_engine = RapidOCREngine()

    def test_blank_image_returns_empty_regions(self):
        """A completely blank image must return an empty list of detections."""
        blank_img = Image.new("RGB", (300, 300), color=(255, 255, 255))
        regions = self.ocr_engine.detect_and_recognize(blank_img)
        self.assertEqual(regions, [])

    def test_synthetic_image_detection(self):
        """Image with rendered text must produce DetectedRegion objects."""
        img = Image.new("RGB", (400, 200), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), "HELLO WORLD", fill=(0, 0, 0))
        regions = self.ocr_engine.detect_and_recognize(img)
        self.assertTrue(len(regions) > 0)
        first = regions[0]
        self.assertIn("HELLO", first.text.upper())
        self.assertTrue(first.confidence > 0.5)
        self.assertTrue(first.bbox.width > 0)
        self.assertTrue(first.bbox.height > 0)

    def test_baseline_predictor_handles_empty_and_valid_pages(self):
        """Predictor must format output as list of 3 pages and pass official validation."""
        predictor = RapidOCRBaselinePredictor(ocr_engine=self.ocr_engine, default_speaker="character1")
        
        # Create a dummy mock sequence with 3 blank images
        dummy_seq = SequenceRecord(
            sequence_id="seq_test_mock1",
            split="development",
            image_paths=[]  # Mocking load_page_images
        )
        
        blank1 = Image.new("RGB", (200, 200), color=(255, 255, 255))
        blank2 = Image.new("RGB", (200, 200), color=(255, 255, 255))
        with_text = Image.new("RGB", (300, 200), color=(255, 255, 255))
        draw = ImageDraw.Draw(with_text)
        draw.text((50, 50), "WHAT?!", fill=(0, 0, 0))

        pages = predictor.predict_images([blank1, with_text, blank2])
        self.assertEqual(len(pages), 3)
        self.assertEqual(pages[0], [])  # Page 1 empty
        self.assertEqual(pages[2], [])  # Page 3 empty
        self.assertTrue(len(pages[1]) > 0)  # Page 2 has detected text
        self.assertEqual(pages[1][0]["speaker"], "character1")
        
        # Test full submission formatting validation
        submission_row = [{"sequence_id": "seq_test_mock1", "pages": pages}]
        validate_submission_data(submission_row, expected_pages=3)

if __name__ == "__main__":
    unittest.main()
