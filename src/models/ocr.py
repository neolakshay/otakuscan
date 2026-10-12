from __future__ import annotations

import numpy as np
from typing import List, Optional
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

from .reading_order import BoundingBox, DetectedRegion

class RapidOCREngine:
    """CPU-optimised text detection and recognition engine powered by RapidOCR (ONNX)."""

    def __init__(self, **kwargs) -> None:
        self.engine = RapidOCR(**kwargs)

    def detect_and_recognize(self, image: Image.Image) -> List[DetectedRegion]:
        """Runs text detection and recognition on a PIL Image.
        
        Returns a list of DetectedRegion objects with 4-point polygon coordinates,
        bounding boxes, transcribed text, and confidence scores.
        """
        if image.mode != "RGB":
            rgb_image = image.convert("RGB")
        else:
            rgb_image = image

        np_img = np.array(rgb_image)
        raw_result, _ = self.engine(np_img)

        if not raw_result:
            return []

        regions: List[DetectedRegion] = []
        for item in raw_result:
            polygon, text, score = item
            poly_points = [[float(pt[0]), float(pt[1])] for pt in polygon]
            xs = [pt[0] for pt in poly_points]
            ys = [pt[1] for pt in poly_points]
            
            bbox = BoundingBox(
                xmin=float(min(xs)),
                ymin=float(min(ys)),
                xmax=float(max(xs)),
                ymax=float(max(ys)),
            )

            regions.append(DetectedRegion(
                polygon=poly_points,
                bbox=bbox,
                text=str(text),
                confidence=float(score),
            ))

        return regions
