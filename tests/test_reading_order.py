import unittest
from src.models.reading_order import BoundingBox, DetectedRegion, sort_reading_order

class TestReadingOrderSorter(unittest.TestCase):
    def test_vertical_precedence(self):
        """Top regions must precede bottom regions."""
        # Top-left box vs bottom-left box
        top_box = DetectedRegion(
            polygon=[[100, 100], [200, 100], [200, 150], [100, 150]],
            bbox=BoundingBox(100, 100, 200, 150),
            text="TOP",
            confidence=0.95
        )
        bottom_box = DetectedRegion(
            polygon=[[100, 400], [200, 400], [200, 450], [100, 450]],
            bbox=BoundingBox(100, 400, 200, 450),
            text="BOTTOM",
            confidence=0.95
        )
        # Input in reverse order
        sorted_regions = sort_reading_order([bottom_box, top_box])
        self.assertEqual([r.text for r in sorted_regions], ["TOP", "BOTTOM"])

    def test_manga_right_to_left_on_same_row(self):
        """Within the same vertical band, right regions must precede left regions (manga order)."""
        right_box = DetectedRegion(
            polygon=[[600, 100], [700, 100], [700, 150], [600, 150]],
            bbox=BoundingBox(600, 100, 700, 150),
            text="FIRST_RIGHT",
            confidence=0.95
        )
        left_box = DetectedRegion(
            polygon=[[200, 100], [300, 100], [300, 150], [200, 150]],
            bbox=BoundingBox(200, 100, 300, 150),
            text="SECOND_LEFT",
            confidence=0.95
        )
        sorted_regions = sort_reading_order([left_box, right_box])
        self.assertEqual([r.text for r in sorted_regions], ["FIRST_RIGHT", "SECOND_LEFT"])

    def test_multi_panel_layout(self):
        """Verify 4-quadrant layout: top-right -> top-left -> bottom-right -> bottom-left."""
        tr = DetectedRegion(
            polygon=[[600, 100], [700, 100], [700, 200], [600, 200]],
            bbox=BoundingBox(600, 100, 700, 200),
            text="1_TR",
            confidence=0.9
        )
        tl = DetectedRegion(
            polygon=[[100, 100], [200, 100], [200, 200], [100, 200]],
            bbox=BoundingBox(100, 100, 200, 200),
            text="2_TL",
            confidence=0.9
        )
        br = DetectedRegion(
            polygon=[[600, 500], [700, 500], [700, 600], [600, 600]],
            bbox=BoundingBox(600, 500, 700, 600),
            text="3_BR",
            confidence=0.9
        )
        bl = DetectedRegion(
            polygon=[[100, 500], [200, 500], [200, 600], [100, 600]],
            bbox=BoundingBox(100, 500, 200, 600),
            text="4_BL",
            confidence=0.9
        )
        # Input shuffled
        sorted_regions = sort_reading_order([bl, tr, br, tl])
        self.assertEqual([r.text for r in sorted_regions], ["1_TR", "2_TL", "3_BR", "4_BL"])

    def test_empty_and_single_input(self):
        """Empty list or single item returns without error."""
        self.assertEqual(sort_reading_order([]), [])
        single = DetectedRegion(
            polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
            bbox=BoundingBox(0, 0, 10, 10),
            text="ONLY",
            confidence=1.0
        )
        self.assertEqual(sort_reading_order([single]), [single])

if __name__ == "__main__":
    unittest.main()
