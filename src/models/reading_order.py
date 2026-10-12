from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Any

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

def sort_panel_aware_reading_order(
    regions: List[Any],
    min_gutter_x: float = 25.0,
    min_gutter_y: float = 40.0,
) -> List[Any]:
    """Sort text regions using recursive spatial decomposition (Hierarchical XY-Cut).
    
    Algorithm:
    1. Base Case: 0 or 1 item returns as-is.
    2. Horizontal Cuts (Tiers):
       Find whitespace gaps across the y-axis separating upper and lower bounding box sets.
       If horizontal cuts exist, partition into tiers ordered strictly Top-to-Bottom.
       Recursively sort each tier.
    3. Vertical Cuts (Columns):
       If no horizontal cut divides the current region, find whitespace gaps across the x-axis.
       If vertical cuts exist, partition into columns ordered Right-to-Left (manga convention).
       Recursively sort each column (ensuring vertically stacked bubbles inside a right panel
       are completely traversed before advancing to the left panel).
    4. Fallback:
       If neither dimension provides clean gutter separation, sort using the robust
       horizontal row-band + right-to-left within row sorter (`sort_reading_order`).
    """
    if len(regions) <= 1:
        return list(regions)

    # 1. Sweep vertically for horizontal cuts (tiers across the current block)
    sorted_by_y = sorted(regions, key=lambda r: (r.bbox.ymin, r.bbox.xmin))
    h_cuts = []
    max_y = sorted_by_y[0].bbox.ymax
    for item in sorted_by_y[1:]:
        if item.bbox.ymin >= max_y + min_gutter_y:
            h_cuts.append((max_y, item.bbox.ymin))
            max_y = item.bbox.ymax
        else:
            max_y = max(max_y, item.bbox.ymax)

    if h_cuts:
        tiers: List[List[Any]] = []
        cur_tier: List[Any] = []
        cur_max_y = sorted_by_y[0].bbox.ymax
        for item in sorted_by_y:
            if cur_tier and item.bbox.ymin >= cur_max_y + min_gutter_y:
                tiers.append(cur_tier)
                cur_tier = [item]
                cur_max_y = item.bbox.ymax
            else:
                cur_tier.append(item)
                cur_max_y = max(cur_max_y, item.bbox.ymax)
        if cur_tier:
            tiers.append(cur_tier)

        if len(tiers) > 1:
            sorted_tiers: List[Any] = []
            for t in tiers:
                sorted_tiers.extend(sort_panel_aware_reading_order(
                    t,
                    min_gutter_x=min_gutter_x,
                    min_gutter_y=min_gutter_y,
                ))
            return sorted_tiers

    # 2. Sweep horizontally for vertical cuts (columns within the current block)
    sorted_by_x = sorted(regions, key=lambda r: (r.bbox.xmin, r.bbox.ymin))
    v_cuts = []
    max_x = sorted_by_x[0].bbox.xmax
    for item in sorted_by_x[1:]:
        if item.bbox.xmin >= max_x + min_gutter_x:
            v_cuts.append((max_x, item.bbox.xmin))
            max_x = item.bbox.xmax
        else:
            max_x = max(max_x, item.bbox.xmax)

    if v_cuts:
        cols: List[List[Any]] = []
        cur_col: List[Any] = []
        cur_max_x = sorted_by_x[0].bbox.xmax
        for item in sorted_by_x:
            if cur_col and item.bbox.xmin >= cur_max_x + min_gutter_x:
                cols.append(cur_col)
                cur_col = [item]
                cur_max_x = item.bbox.xmax
            else:
                cur_col.append(item)
                cur_max_x = max(cur_max_x, item.bbox.xmax)
        if cur_col:
            cols.append(cur_col)

        if len(cols) > 1:
            sorted_cols: List[Any] = []
            # Manga convention: rightmost column first (descending x)
            for c in reversed(cols):
                sorted_cols.extend(sort_panel_aware_reading_order(
                    c,
                    min_gutter_x=min_gutter_x,
                    min_gutter_y=min_gutter_y,
                ))
            return sorted_cols

    # 3. Fallback: local row-band sorter
    return sort_reading_order(regions)

