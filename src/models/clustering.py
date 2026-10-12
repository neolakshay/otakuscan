from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .reading_order import BoundingBox, DetectedRegion

@dataclass(frozen=True)
class TextCluster:
    """A coherent text region (speech bubble or caption box) aggregating constituent text lines."""
    regions: List[DetectedRegion]
    bbox: BoundingBox
    text: str
    confidence: float

    @property
    def polygon(self) -> List[List[float]]:
        """Axis-aligned 4-point polygon matching the enclosing bounding box."""
        return [
            [self.bbox.xmin, self.bbox.ymin],
            [self.bbox.xmax, self.bbox.ymin],
            [self.bbox.xmax, self.bbox.ymax],
            [self.bbox.xmin, self.bbox.ymax],
        ]

def should_merge_lines(
    r1: DetectedRegion,
    r2: DetectedRegion,
    max_vertical_gap_ratio: float = 0.8,
    min_horizontal_overlap_ratio: float = 0.25,
    max_center_diff_ratio: float = 0.45,
    min_height_ratio: float = 0.40,
) -> bool:
    """Determines whether two detected lines belong to the same multi-line speech bubble.
    
    Conditions:
    1. Scale Consistency: line heights must be within a consistent ratio (e.g. >= 0.40).
    2. Vertical Stacking: one line must be vertically adjacent to the other, not on the same row.
    3. Horizontal Alignment: lines must share substantial horizontal overlap or aligned centers.
    """
    b1, b2 = r1.bbox, r2.bbox
    h1, h2 = b1.height, b2.height
    w1, w2 = b1.width, b2.width

    min_h = min(h1, h2)
    max_h = max(h1, h2)
    if min_h <= 0 or max_h <= 0 or (min_h / max_h) < min_height_ratio:
        return False

    # Determine upper and lower line
    if b1.ymin <= b2.ymin:
        upper, lower = b1, b2
    else:
        upper, lower = b2, b1

    # Check vertical relationship
    vertical_gap = lower.ymin - upper.ymax
    vertical_overlap = max(0.0, upper.ymax - lower.ymin)

    # If lines heavily overlap vertically (e.g. > 45% of height), they are side-by-side on the same row
    if (vertical_overlap / min_h) > 0.45:
        return False

    # If the gap between upper bottom and lower top exceeds the line-height threshold, do not merge
    if vertical_gap > (max_vertical_gap_ratio * min_h):
        return False

    # Check horizontal alignment
    x_overlap = max(0.0, min(upper.xmax, lower.xmax) - max(upper.xmin, lower.xmin))
    min_w = min(w1, w2)
    max_w = max(w1, w2)

    has_overlap = min_w > 0 and (x_overlap / min_w) >= min_horizontal_overlap_ratio
    center_diff = abs(upper.center_x - lower.center_x)
    has_aligned_centers = max_w > 0 and (center_diff / max_w) <= max_center_diff_ratio

    return has_overlap or has_aligned_centers

def cluster_text_regions(
    regions: List[DetectedRegion],
    max_vertical_gap_ratio: float = 0.8,
    min_horizontal_overlap_ratio: float = 0.25,
    max_center_diff_ratio: float = 0.45,
    min_height_ratio: float = 0.40,
) -> List[TextCluster]:
    """Deterministically groups line-level detections into coherent speech bubble clusters.
    
    - Never discards detections: every input region is represented in a cluster.
    - Preserves punctuation-only and short text.
    - Sorts lines inside each cluster strictly Top-to-Bottom.
    - Joins line text with a single space.
    """
    if not regions:
        return []

    if len(regions) == 1:
        r = regions[0]
        return [TextCluster(
            regions=[r],
            bbox=r.bbox,
            text=r.text.strip(),
            confidence=r.confidence,
        )]

    # Deterministic initial ordering by ymin, then xmin
    sorted_regions = sorted(regions, key=lambda r: (r.bbox.ymin, r.bbox.xmin))
    n = len(sorted_regions)

    # Build adjacency graph
    adj: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if should_merge_lines(
                sorted_regions[i],
                sorted_regions[j],
                max_vertical_gap_ratio=max_vertical_gap_ratio,
                min_horizontal_overlap_ratio=min_horizontal_overlap_ratio,
                max_center_diff_ratio=max_center_diff_ratio,
                min_height_ratio=min_height_ratio,
            ):
                adj[i].append(j)
                adj[j].append(i)

    # Find connected components using BFS
    visited = [False] * n
    clusters: List[TextCluster] = []

    for i in range(n):
        if visited[i]:
            continue

        component_indices: List[int] = []
        queue = [i]
        visited[i] = True

        while queue:
            curr = queue.pop(0)
            component_indices.append(curr)
            for neighbor in adj[curr]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

        comp_regions = [sorted_regions[idx] for idx in component_indices]
        # Sort constituent lines strictly Top-to-Bottom (ymin, then xmin)
        comp_regions.sort(key=lambda r: (r.bbox.ymin, r.bbox.xmin))

        # Enclosing bounding box
        xmin = min(r.bbox.xmin for r in comp_regions)
        ymin = min(r.bbox.ymin for r in comp_regions)
        xmax = max(r.bbox.xmax for r in comp_regions)
        ymax = max(r.bbox.ymax for r in comp_regions)

        # Concatenate text with single space
        joined_text = " ".join(r.text.strip() for r in comp_regions if r.text.strip())
        avg_conf = sum(r.confidence for r in comp_regions) / len(comp_regions)

        clusters.append(TextCluster(
            regions=comp_regions,
            bbox=BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax),
            text=joined_text,
            confidence=avg_conf,
        ))

    # Deterministic output sorting by (ymin, xmin)
    clusters.sort(key=lambda c: (c.bbox.ymin, c.bbox.xmin))
    return clusters
