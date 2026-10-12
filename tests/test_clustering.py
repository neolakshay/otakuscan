import unittest
from src.models.reading_order import BoundingBox, DetectedRegion
from src.models.clustering import cluster_text_regions, TextCluster

class TestTextRegionClustering(unittest.TestCase):
    def test_multiline_speech_bubble_merged(self):
        """Lines stacked vertically within a bubble must merge into a single cluster in top-down order."""
        line1 = DetectedRegion(
            polygon=[[100, 100], [200, 100], [200, 120], [100, 120]],
            bbox=BoundingBox(100, 100, 200, 120),
            text="HELLO",
            confidence=0.9
        )
        line2 = DetectedRegion(
            polygon=[[102, 122], [198, 122], [198, 142], [102, 142]],
            bbox=BoundingBox(102, 122, 198, 142),
            text="WORLD",
            confidence=0.8
        )
        clusters = cluster_text_regions([line2, line1])
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].text, "HELLO WORLD")
        # Enclosing bounding box
        self.assertEqual(clusters[0].bbox.xmin, 100)
        self.assertEqual(clusters[0].bbox.ymin, 100)
        self.assertEqual(clusters[0].bbox.xmax, 200)
        self.assertEqual(clusters[0].bbox.ymax, 142)
        # Average confidence
        self.assertAlmostEqual(clusters[0].confidence, 0.85, places=4)

    def test_parallel_horizontal_bubbles_remain_separate(self):
        """Two separate bubbles sitting side-by-side horizontally must never merge."""
        # Right bubble (2 lines)
        r1 = DetectedRegion(
            polygon=[[600, 100], [700, 100], [700, 120], [600, 120]],
            bbox=BoundingBox(600, 100, 700, 120),
            text="RIGHT_1",
            confidence=0.9
        )
        r2 = DetectedRegion(
            polygon=[[600, 124], [700, 124], [700, 144], [600, 144]],
            bbox=BoundingBox(600, 124, 700, 144),
            text="RIGHT_2",
            confidence=0.9
        )
        # Left bubble (2 lines)
        l1 = DetectedRegion(
            polygon=[[200, 100], [300, 100], [300, 120], [200, 120]],
            bbox=BoundingBox(200, 100, 300, 120),
            text="LEFT_1",
            confidence=0.9
        )
        l2 = DetectedRegion(
            polygon=[[200, 124], [300, 124], [300, 144], [200, 144]],
            bbox=BoundingBox(200, 124, 300, 144),
            text="LEFT_2",
            confidence=0.9
        )

        clusters = cluster_text_regions([r1, l1, r2, l2])
        self.assertEqual(len(clusters), 2)
        cluster_texts = {c.text for c in clusters}
        self.assertIn("RIGHT_1 RIGHT_2", cluster_texts)
        self.assertIn("LEFT_1 LEFT_2", cluster_texts)

    def test_vertically_distant_regions_remain_separate(self):
        """Regions with vertical gap exceeding the line-height threshold must not merge."""
        top = DetectedRegion(
            polygon=[[100, 100], [200, 100], [200, 120], [100, 120]],
            bbox=BoundingBox(100, 100, 200, 120),
            text="TOP_SPEECH",
            confidence=0.9
        )
        # Vertical gap is 80px (4x line height of 20px)
        bottom = DetectedRegion(
            polygon=[[100, 200], [200, 200], [200, 220], [100, 220]],
            bbox=BoundingBox(100, 200, 200, 220),
            text="BOTTOM_SPEECH",
            confidence=0.9
        )
        clusters = cluster_text_regions([top, bottom])
        self.assertEqual(len(clusters), 2)

    def test_different_text_sizes_handled_conservatively(self):
        """Lines with drastically mismatched heights (e.g. SFX vs dialogue) must not merge."""
        dialogue = DetectedRegion(
            polygon=[[100, 100], [200, 100], [200, 120], [100, 120]],
            bbox=BoundingBox(100, 100, 200, 120),
            text="NORMAL DIALOGUE",
            confidence=0.9
        )
        # Giant sound effect right below dialogue (height 100px vs 20px -> ratio 0.2 < 0.4)
        giant_sfx = DetectedRegion(
            polygon=[[100, 122], [250, 122], [250, 222], [100, 222]],
            bbox=BoundingBox(100, 122, 250, 222),
            text="BOOOM",
            confidence=0.95
        )
        clusters = cluster_text_regions([dialogue, giant_sfx])
        self.assertEqual(len(clusters), 2)

    def test_punctuation_only_text_preserved(self):
        """Punctuation-only detections must not be dropped."""
        punct = DetectedRegion(
            polygon=[[100, 100], [130, 100], [130, 120], [100, 120]],
            bbox=BoundingBox(100, 100, 130, 120),
            text="...",
            confidence=0.88
        )
        clusters = cluster_text_regions([punct])
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].text, "...")

    def test_every_input_detection_represented(self):
        """Total text content across all clusters must contain every input text token."""
        inputs = [
            DetectedRegion(polygon=[[0, 0], [10, 0], [10, 10], [0, 10]], bbox=BoundingBox(0, 0, 10, 10), text=f"W_{i}", confidence=0.9)
            for i in range(10)
        ]
        clusters = cluster_text_regions(inputs)
        combined_output = " ".join(c.text for c in clusters)
        for i in range(10):
            self.assertIn(f"W_{i}", combined_output)

    def test_deterministic_output(self):
        """Running clustering multiple times with reversed inputs yields identical clusters."""
        r1 = DetectedRegion(polygon=[[100, 100], [200, 100], [200, 120], [100, 120]], bbox=BoundingBox(100, 100, 200, 120), text="A", confidence=0.9)
        r2 = DetectedRegion(polygon=[[100, 122], [200, 122], [200, 142], [100, 142]], bbox=BoundingBox(100, 122, 200, 142), text="B", confidence=0.9)
        r3 = DetectedRegion(polygon=[[100, 300], [200, 300], [200, 320], [100, 320]], bbox=BoundingBox(100, 300, 200, 320), text="C", confidence=0.9)

        clusters_forward = cluster_text_regions([r1, r2, r3])
        clusters_backward = cluster_text_regions([r3, r2, r1])

        self.assertEqual(len(clusters_forward), len(clusters_backward))
        self.assertEqual([c.text for c in clusters_forward], [c.text for c in clusters_backward])

    def test_prevent_horizontal_chaining_across_page(self):
        """Lines chained horizontally across a page must not merge into an absurdly wide cluster."""
        # 3 boxes side-by-side with small horizontal gaps
        box1 = DetectedRegion(polygon=[[100, 100], [150, 100], [150, 120], [100, 120]], bbox=BoundingBox(100, 100, 150, 120), text="B1", confidence=0.9)
        box2 = DetectedRegion(polygon=[[160, 100], [210, 100], [210, 120], [160, 120]], bbox=BoundingBox(160, 100, 210, 120), text="B2", confidence=0.9)
        box3 = DetectedRegion(polygon=[[220, 100], [270, 100], [270, 120], [220, 120]], bbox=BoundingBox(220, 100, 270, 120), text="B3", confidence=0.9)

        # Lines on the exact same row should NOT be merged vertically as lines of a single bubble
        clusters = cluster_text_regions([box1, box2, box3])
        self.assertEqual(len(clusters), 3)

if __name__ == "__main__":
    unittest.main()
