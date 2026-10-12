from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

@dataclass(frozen=True)
class BoundingBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def center_x(self) -> float:
        return (self.xmin + self.xmax) / 2.0

    @property
    def center_y(self) -> float:
        return (self.ymin + self.ymax) / 2.0

@dataclass(frozen=True)
class DetectedRegion:
    polygon: List[List[float]]
    bbox: BoundingBox
    text: str
    confidence: float

def sort_reading_order(
    regions: List[DetectedRegion],
    vertical_overlap_ratio: float = 0.4
) -> List[DetectedRegion]:
    """Sort detected text regions in standard manga reading order: Top-to-Bottom, Right-to-Left.
    
    Algorithm:
    1. Sort candidates primarily by top vertical coordinate (ymin).
    2. Cluster into horizontal bands/rows where regions share substantial vertical overlap.
    3. Sort rows top-to-bottom.
    4. Within each row, sort regions right-to-left (center_x descending).
    5. Return flattened sorted sequence.
    """
    if len(regions) <= 1:
        return list(regions)

    # Initial sort by ymin
    sorted_by_y = sorted(regions, key=lambda r: (r.bbox.ymin, -r.bbox.center_x))

    # Cluster into vertical bands
    rows: List[List[DetectedRegion]] = []
    row_bounds: List[Tuple[float, float]] = []  # (ymin, ymax) of each row

    for region in sorted_by_y:
        r_ymin = region.bbox.ymin
        r_ymax = region.bbox.ymax
        r_height = max(1.0, region.bbox.height)

        assigned_row_idx = None
        for idx, (b_ymin, b_ymax) in enumerate(row_bounds):
            b_height = max(1.0, b_ymax - b_ymin)
            # Calculate vertical intersection
            overlap_min = max(r_ymin, b_ymin)
            overlap_max = min(r_ymax, b_ymax)
            overlap = max(0.0, overlap_max - overlap_min)
            
            # If vertical overlap is significant relative to either height
            if overlap / min(r_height, b_height) >= vertical_overlap_ratio:
                assigned_row_idx = idx
                break

        if assigned_row_idx is not None:
            rows[assigned_row_idx].append(region)
            # Expand row bounds
            b_ymin, b_ymax = row_bounds[assigned_row_idx]
            row_bounds[assigned_row_idx] = (min(b_ymin, r_ymin), max(b_ymax, r_ymax))
        else:
            rows.append([region])
            row_bounds.append((r_ymin, r_ymax))

    # Sort each row right-to-left (manga order: center_x descending)
    ordered_regions: List[DetectedRegion] = []
    # Sort rows by their average ymin/center_y
    sorted_rows = sorted(zip(rows, row_bounds), key=lambda item: item[1][0])
    for row_items, _ in sorted_rows:
        # Right to left: highest center_x first
        row_sorted = sorted(row_items, key=lambda r: -r.bbox.center_x)
        ordered_regions.extend(row_sorted)

    return ordered_regions
