from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class TailDetectionResult:
    """Diagnostic outcome of speech-bubble tail detection on a text cluster."""
    status: str  # 'tail_detected', 'no_tail', 'narration_or_box', 'burst_bubble', 'background_bleed', 'no_bubble'
    has_tail: bool
    confidence: float
    tip_coords: Optional[Tuple[float, float]] = None  # (x, y) in page coordinates
    centroid_coords: Optional[Tuple[float, float]] = None  # (x, y) in page coordinates
    direction_vector: Optional[Tuple[float, float]] = None  # normalized (dx, dy)
    contour_area: float = 0.0
    bubble_bbox: Optional[Tuple[float, float, float, float]] = None  # (xmin, ymin, xmax, ymax)
    failure_reason: Optional[str] = None


class BubbleTailDetector:
    """Conservative computer-vision detector for speech-bubble contours and tail geometry.
    
    Principles:
    1. Unsupervised / classical CV: Uses edge morphology, connected components, and contour geometry.
    2. Zero Label Leakage: Does not use OCR transcripts, speaker labels, or gold references.
    3. Conservative Null Hypothesis: If a tail is ambiguous, noisy, or absent, returns has_tail=False
       with confidence=0.0 rather than inventing a spurious directional signal.
    """

    def __init__(
        self,
        canny_low: int = 40,
        canny_high: int = 130,
        max_pad: int = 80,
        min_bubble_area_ratio: float = 1.05,
        max_bubble_area_ratio: float = 6.0,
        acute_angle_threshold_deg: float = 95.0,
        max_acute_vertices_for_tail: int = 3,
    ) -> None:
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.max_pad = max_pad
        self.min_bubble_area_ratio = min_bubble_area_ratio
        self.max_bubble_area_ratio = max_bubble_area_ratio
        self.acute_angle_threshold_deg = acute_angle_threshold_deg
        self.max_acute_vertices_for_tail = max_acute_vertices_for_tail

    def detect_tail(
        self,
        image_gray: np.ndarray,
        text_bbox: Tuple[float, float, float, float],
    ) -> TailDetectionResult:
        """Analyzes local image region around text_bbox to detect bubble contour and tail.
        
        Args:
            image_gray: Grayscale page image.
            text_bbox: (xmin, ymin, xmax, ymax) in page coordinates.
            
        Returns:
            TailDetectionResult detailing contour and tail status.
        """
        H, W = image_gray.shape
        t_xmin, t_ymin, t_xmax, t_ymax = text_bbox
        tw = max(1.0, t_xmax - t_xmin)
        th = max(1.0, t_ymax - t_ymin)
        text_area = tw * th

        # Dynamic ROI padding
        pad_x = min(self.max_pad, max(30, int(tw * 0.6)))
        pad_y = min(self.max_pad, max(30, int(th * 0.6)))

        rx1 = max(0, int(t_xmin - pad_x))
        ry1 = max(0, int(t_ymin - pad_y))
        rx2 = min(W, int(t_xmax + pad_x))
        ry2 = min(H, int(t_ymax + pad_y))

        crop = image_gray[ry1:ry2, rx1:rx2].copy()
        ch, cw = crop.shape

        # Text box in crop coordinates
        ctx1 = int(t_xmin - rx1)
        cty1 = int(t_ymin - ry1)
        ctx2 = int(t_xmax - rx1)
        cty2 = int(t_ymax - ry1)

        tcx = (ctx1 + ctx2) // 2
        tcy = (cty1 + cty2) // 2

        # Check for inverted narration box (white text on dark background)
        text_patch = crop[max(0, cty1):min(ch, cty2), max(0, ctx1):min(cw, ctx2)]
        if text_patch.size > 0 and np.mean(text_patch) < 80:
            return TailDetectionResult(
                status="narration_or_box",
                has_tail=False,
                confidence=0.0,
                failure_reason="Inverted text box (white text on dark background)",
            )

        # Inpaint/whiteout text box interior to prevent text glyphs from fragmenting bubble
        clean_crop = crop.copy()
        clean_crop[max(0, cty1):min(ch, cty2), max(0, ctx1):min(cw, ctx2)] = 255

        # Canny edge detection
        edges = cv2.Canny(clean_crop, self.canny_low, self.canny_high)

        # Morphological dilation to bridge small 1-2 pixel ink gaps in bubble border
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_edges = cv2.dilate(edges, kernel)
        free_space = cv2.bitwise_not(dilated_edges)

        # Flood fill from text center
        flood_mask = np.zeros((ch + 2, cw + 2), np.uint8)
        filled = free_space.copy()
        if tcy >= ch or tcx >= cw or tcy < 0 or tcx < 0:
            return TailDetectionResult(
                status="no_bubble",
                has_tail=False,
                confidence=0.0,
                failure_reason="Text center outside crop bounds",
            )

        # If seed point is accidentally on an edge pixel, find closest white pixel
        if filled[tcy, tcx] == 0:
            found_seed = False
            for dy in range(-5, 6):
                for dx in range(-5, 6):
                    ny, nx = tcy + dy, tcx + dx
                    if 0 <= ny < ch and 0 <= nx < cw and filled[ny, nx] > 0:
                        tcx, tcy = nx, ny
                        found_seed = True
                        break
                if found_seed:
                    break
            if not found_seed:
                return TailDetectionResult(
                    status="no_bubble",
                    has_tail=False,
                    confidence=0.0,
                    failure_reason="Seed point entirely blocked by ink edges",
                )

        cv2.floodFill(filled, flood_mask, (tcx, tcy), 128)
        bubble_mask = (filled == 128).astype(np.uint8) * 255

        cnts, _ = cv2.findContours(bubble_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return TailDetectionResult(
                status="no_bubble",
                has_tail=False,
                confidence=0.0,
                failure_reason="No connected bubble region found",
            )

        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        cbx, cby, cbw, cbh = cv2.boundingRect(cnt)

        # Check for uncontained background bleed (touches multiple crop edges)
        touches_left = (cbx <= 2)
        touches_right = (cbx + cbw >= cw - 2)
        touches_top = (cby <= 2)
        touches_bottom = (cby + cbh >= ch - 2)
        touch_count = sum([touches_left, touches_right, touches_top, touches_bottom])

        if touch_count >= 3:
            return TailDetectionResult(
                status="background_bleed",
                has_tail=False,
                confidence=0.0,
                contour_area=area,
                failure_reason=f"Bubble unclosed: bleeds into {touch_count} crop edges",
            )

        # Check area plausibility
        if area < text_area * self.min_bubble_area_ratio:
            return TailDetectionResult(
                status="no_bubble",
                has_tail=False,
                confidence=0.0,
                contour_area=area,
                failure_reason=f"Detected area ({area:.0f}) smaller than text box ({text_area:.0f})",
            )

        # Centroid of bubble
        M = cv2.moments(cnt)
        if M["m00"] <= 0:
            return TailDetectionResult(
                status="no_bubble",
                has_tail=False,
                confidence=0.0,
                failure_reason="Zero moment for bubble contour",
            )

        bcx = M["m10"] / M["m00"]
        bcy = M["m01"] / M["m00"]

        # Polygonal approximation to analyze vertices
        peri = cv2.arcLength(cnt, True)
        poly = cv2.approxPolyDP(cnt, 0.015 * peri, True)
        pts = poly.reshape(-1, 2)
        n_pts = len(pts)

        if n_pts < 4:
            return TailDetectionResult(
                status="no_tail",
                has_tail=False,
                confidence=0.0,
                contour_area=area,
                failure_reason="Contour too degenerate (< 4 vertices)",
            )

        # Detect acute vertices
        acute_candidates: List[Tuple[np.ndarray, float, float]] = []  # (pt, angle_deg, dist_from_center)
        for i in range(n_pts):
            p_prev = pts[(i - 1) % n_pts]
            p_curr = pts[i]
            p_next = pts[(i + 1) % n_pts]

            v1 = (p_prev - p_curr).astype(float)
            v2 = (p_next - p_curr).astype(float)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 > 5 and norm2 > 5:
                cos_a = np.dot(v1, v2) / (norm1 * norm2)
                ang_deg = math.degrees(math.acos(np.clip(cos_a, -1.0, 1.0)))
                dist_c = np.linalg.norm(p_curr - np.array([bcx, bcy]))

                # Check if this vertex is simply a flat crop boundary corner
                is_crop_corner = (
                    (p_curr[0] <= 3 or p_curr[0] >= cw - 3) and
                    (p_curr[1] <= 3 or p_curr[1] >= ch - 3)
                )

                if ang_deg < self.acute_angle_threshold_deg and not is_crop_corner:
                    acute_candidates.append((p_curr, ang_deg, dist_c))

        page_bcx = rx1 + bcx
        page_bcy = ry1 + bcy
        bubble_page_bbox = (float(rx1 + cbx), float(ry1 + cby), float(rx1 + cbx + cbw), float(ry1 + cby + cbh))

        # Check for scream/burst bubble with many acute spikes
        if len(acute_candidates) > self.max_acute_vertices_for_tail:
            return TailDetectionResult(
                status="burst_bubble",
                has_tail=False,
                confidence=0.0,
                centroid_coords=(page_bcx, page_bcy),
                contour_area=area,
                bubble_bbox=bubble_page_bbox,
                failure_reason=f"Scream/burst bubble: {len(acute_candidates)} acute spikes detected",
            )

        # Check for smooth bubble with no acute tips
        if len(acute_candidates) == 0:
            return TailDetectionResult(
                status="no_tail",
                has_tail=False,
                confidence=0.0,
                centroid_coords=(page_bcx, page_bcy),
                contour_area=area,
                bubble_bbox=bubble_page_bbox,
                failure_reason="Smooth oval/box bubble: no acute protrusions found",
            )

        # Candidate selection: pick most prominent acute tip (farthest from center with sharpest angle)
        # Score = dist_c / (ang_deg + 10.0)
        best_pt, best_ang, best_dist = max(
            acute_candidates,
            key=lambda item: item[2] / (item[1] + 10.0)
        )

        tip_x = rx1 + float(best_pt[0])
        tip_y = ry1 + float(best_pt[1])

        # Compute direction vector from bubble centroid to tail tip
        dx = tip_x - page_bcx
        dy = tip_y - page_bcy
        length = math.hypot(dx, dy)
        if length < 5.0:
            return TailDetectionResult(
                status="unknown_or_low_confidence",
                has_tail=False,
                confidence=0.0,
                centroid_coords=(page_bcx, page_bcy),
                contour_area=area,
                bubble_bbox=bubble_page_bbox,
                failure_reason="Tail tip too close to bubble center",
            )

        unit_dx = dx / length
        unit_dy = dy / length

        # Confidence heuristic: based on sharpness and protrusion distance
        sharpness_score = max(0.0, min(1.0, (100.0 - best_ang) / 50.0))
        protrusion_score = max(0.0, min(1.0, length / (max(tw, th) * 0.8)))
        confidence = round(0.5 * sharpness_score + 0.5 * protrusion_score, 3)

        return TailDetectionResult(
            status="tail_detected",
            has_tail=True,
            confidence=confidence,
            tip_coords=(tip_x, tip_y),
            centroid_coords=(page_bcx, page_bcy),
            direction_vector=(round(unit_dx, 4), round(unit_dy, 4)),
            contour_area=area,
            bubble_bbox=bubble_page_bbox,
        )
