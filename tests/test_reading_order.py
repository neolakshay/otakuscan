import unittest
from src.models.reading_order import (
    BoundingBox,
    DetectedRegion,
    sort_reading_order,
    sort_panel_aware_reading_order,
)

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


class TestPanelAwareReadingOrder(unittest.TestCase):
    def test_vertically_stacked_panels_in_adjacent_columns(self):
        """When a right column has multiple stacked bubbles adjacent to a left column,
        the entire right column must be read top-to-bottom before the left column."""
        # Right column: Bubble 1 (top), Bubble 2 (bottom)
        r_top = DetectedRegion(
            polygon=[[500, 100], [650, 100], [650, 200], [500, 200]],
            bbox=BoundingBox(500, 100, 650, 200),
            text="R1_TOP",
            confidence=0.95
        )
        r_bottom = DetectedRegion(
            polygon=[[500, 300], [650, 300], [650, 400], [500, 400]],
            bbox=BoundingBox(500, 300, 650, 400),
            text="R2_BOTTOM",
            confidence=0.95
        )
        # Left column: Bubble sitting at y=180..280 (overlapping vertically between R1 and R2)
        l_mid = DetectedRegion(
            polygon=[[100, 180], [250, 180], [250, 280], [100, 280]],
            bbox=BoundingBox(100, 180, 250, 280),
            text="L_MID",
            confidence=0.95
        )

        # Baseline sorter incorrectly places L_MID before R2_BOTTOM due to row-banding.
        # Panel-aware sorter must read right column top-to-bottom first: R1 -> R2 -> L.
        sorted_regions = sort_panel_aware_reading_order([l_mid, r_bottom, r_top])
        self.assertEqual([r.text for r in sorted_regions], ["R1_TOP", "R2_BOTTOM", "L_MID"])

    def test_multi_column_three_panel_layout(self):
        """Right column -> Center column -> Left column."""
        col_right = DetectedRegion(
            polygon=[[600, 100], [700, 100], [700, 250], [600, 250]],
            bbox=BoundingBox(600, 100, 700, 250),
            text="COL_RIGHT",
            confidence=0.9
        )
        col_center = DetectedRegion(
            polygon=[[350, 120], [450, 120], [450, 260], [350, 260]],
            bbox=BoundingBox(350, 120, 450, 260),
            text="COL_CENTER",
            confidence=0.9
        )
        col_left = DetectedRegion(
            polygon=[[100, 110], [200, 110], [200, 270], [100, 270]],
            bbox=BoundingBox(100, 110, 200, 270),
            text="COL_LEFT",
            confidence=0.9
        )
        sorted_regions = sort_panel_aware_reading_order([col_left, col_right, col_center])
        self.assertEqual([r.text for r in sorted_regions], ["COL_RIGHT", "COL_CENTER", "COL_LEFT"])

    def test_irregular_layout_full_width_tiers_with_split_middle(self):
        """Top full-width banner -> Middle right panel (2 bubbles) -> Middle left panel -> Bottom full-width banner."""
        top_banner = DetectedRegion(
            polygon=[[200, 50], [600, 50], [600, 100], [200, 100]],
            bbox=BoundingBox(200, 50, 600, 100),
            text="TOP_BANNER",
            confidence=0.95
        )
        mid_r1 = DetectedRegion(
            polygon=[[500, 200], [650, 200], [650, 280], [500, 280]],
            bbox=BoundingBox(500, 200, 650, 280),
            text="MID_R1",
            confidence=0.95
        )
        mid_r2 = DetectedRegion(
            polygon=[[500, 340], [650, 340], [650, 420], [500, 420]],
            bbox=BoundingBox(500, 340, 650, 420),
            text="MID_R2",
            confidence=0.95
        )
        mid_l = DetectedRegion(
            polygon=[[100, 250], [250, 250], [250, 380], [100, 380]],
            bbox=BoundingBox(100, 250, 250, 380),
            text="MID_L",
            confidence=0.95
        )
        bottom_banner = DetectedRegion(
            polygon=[[150, 600], [650, 600], [650, 700], [150, 700]],
            bbox=BoundingBox(150, 600, 650, 700),
            text="BOTTOM_BANNER",
            confidence=0.95
        )

        input_items = [bottom_banner, mid_l, mid_r2, top_banner, mid_r1]
        sorted_regions = sort_panel_aware_reading_order(input_items)
        self.assertEqual(
            [r.text for r in sorted_regions],
            ["TOP_BANNER", "MID_R1", "MID_R2", "MID_L", "BOTTOM_BANNER"]
        )

    def test_right_to_left_within_single_panel(self):
        """Inside a single panel, side-by-side bubbles are sorted right to left."""
        b_right = DetectedRegion(
            polygon=[[300, 200], [350, 200], [350, 250], [300, 250]],
            bbox=BoundingBox(300, 200, 350, 250),
            text="P_RIGHT",
            confidence=0.9
        )
        b_left = DetectedRegion(
            polygon=[[150, 200], [200, 200], [200, 250], [150, 250]],
            bbox=BoundingBox(150, 200, 200, 250),
            text="P_LEFT",
            confidence=0.9
        )
        sorted_regions = sort_panel_aware_reading_order([b_left, b_right])
        self.assertEqual([r.text for r in sorted_regions], ["P_RIGHT", "P_LEFT"])

    def test_empty_and_single_input(self):
        """Empty list or single item returns without error."""
        self.assertEqual(sort_panel_aware_reading_order([]), [])
        single = DetectedRegion(
            polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
            bbox=BoundingBox(0, 0, 10, 10),
            text="ONLY",
            confidence=1.0
        )
        self.assertEqual(sort_panel_aware_reading_order([single]), [single])

if __name__ == "__main__":
    unittest.main()

