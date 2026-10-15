from __future__ import annotations

import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.loader import OtakuDatasetLoader
from src.models.ocr import RapidOCREngine
from src.models.clustering import cluster_text_regions, TextCluster
from src.experiments.bubble_tail_detector import BubbleTailDetector, TailDetectionResult


# Strictly permitted Fold 0 development sequences for the POC
REPRESENTATIVE_DEV_SEQUENCES = [
    "seq_032cbb4dc8686963",  # Clean duologue, 2 speakers + narration, smooth bubbles
    "seq_ac5793f8168efdd1",  # Multi-speaker action, 5 speakers, dark screentone, narration
    "seq_72fdfff4b4353a42",  # Comedy duologue, 3 speakers, inverted narration box
    "seq_864aaca43a3580bb",  # Crowded dialogue, 8 speakers, overlapping bubbles
]


def run_bubble_tail_poc(
    dataset_root: Path | str = project_root / "dataset",
    output_dir: Path | str = project_root / "logs" / "bubble_tail_poc",
) -> Dict[str, Any]:
    dataset_root = Path(dataset_root).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Verify that sequences are strictly within Fold 0 training split
    splits_file = project_root / "splits" / "dev_splits.json"
    with open(splits_file, "r", encoding="utf-8") as f:
        splits = json.load(f)
    train_ids = set(splits["kfold_5"][0]["train"])
    val_ids = set(splits["kfold_5"][0]["val"])

    for sid in REPRESENTATIVE_DEV_SEQUENCES:
        assert sid in train_ids, f"Leakage violation! Sequence {sid} is not in Fold 0 training split!"
        assert sid not in val_ids, f"Leakage violation! Sequence {sid} is in Fold 0 validation split!"

    print(f"Verified {len(REPRESENTATIVE_DEV_SEQUENCES)} development sequences strictly in Fold 0 training split.")

    loader = OtakuDatasetLoader(dataset_root)
    all_seqs = {s.sequence_id: s for s in loader.load_sequences()}
    ocr = RapidOCREngine()
    detector = BubbleTailDetector()

    total_candidates_examined = 0
    status_counts: Dict[str, int] = {
        "tail_detected": 0,
        "no_tail": 0,
        "narration_or_box": 0,
        "burst_bubble": 0,
        "background_bleed": 0,
        "no_bubble": 0,
        "unknown_or_low_confidence": 0,
    }

    per_sequence_results: List[Dict[str, Any]] = []

    tracemalloc.start()
    total_start_time = time.perf_counter()

    for seq_id in REPRESENTATIVE_DEV_SEQUENCES:
        seq_record = all_seqs[seq_id]
        images = loader.load_images(seq_record, convert_rgb=True)
        print(f"\nProcessing {seq_id} (3 pages)...")

        seq_clusters_count = 0
        seq_tails_count = 0

        for p_idx, img_pil in enumerate(images):
            img_np = np.array(img_pil)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            H, W = gray.shape

            # Run OCR and text clustering
            regs = ocr.detect_and_recognize(img_pil)
            clusters: List[TextCluster] = cluster_text_regions(regs)

            page_vis = img_bgr.copy()

            for c_idx, c in enumerate(clusters):
                total_candidates_examined += 1
                seq_clusters_count += 1

                t_bbox = (float(c.bbox.xmin), float(c.bbox.ymin), float(c.bbox.xmax), float(c.bbox.ymax))
                res: TailDetectionResult = detector.detect_tail(gray, t_bbox)

                status_counts[res.status] = status_counts.get(res.status, 0) + 1

                # Visual annotations on page_vis
                tx1, ty1, tx2, ty2 = int(t_bbox[0]), int(t_bbox[1]), int(t_bbox[2]), int(t_bbox[3])
                cv2.rectangle(page_vis, (tx1, ty1), (tx2, ty2), (255, 0, 0), 1)

                if res.bubble_bbox:
                    bx1, by1, bx2, by2 = [int(v) for v in res.bubble_bbox]
                    cv2.rectangle(page_vis, (bx1, by1), (bx2, by2), (0, 200, 0), 1)

                if res.has_tail and res.tip_coords and res.centroid_coords:
                    seq_tails_count += 1
                    cx, cy = int(res.centroid_coords[0]), int(res.centroid_coords[1])
                    tip_x, tip_y = int(res.tip_coords[0]), int(res.tip_coords[1])

                    # Red dot on centroid
                    cv2.circle(page_vis, (cx, cy), 3, (0, 0, 255), -1)
                    # Yellow dot on tail tip
                    cv2.circle(page_vis, (tip_x, tip_y), 4, (0, 255, 255), -1)
                    # Orange arrow pointing in direction of tail
                    cv2.arrowedLine(page_vis, (cx, cy), (tip_x, tip_y), (0, 140, 255), 2, tipLength=0.25)
                    # Label confidence
                    cv2.putText(
                        page_vis,
                        f"Tail ({res.confidence:.2f})",
                        (tip_x + 5, tip_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 140, 255),
                        1,
                    )
                else:
                    # Draw subtle status tag
                    tag = "NO_TAIL" if res.status == "no_tail" else res.status.upper()
                    cv2.putText(
                        page_vis,
                        tag,
                        (tx1, max(12, ty1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (150, 150, 150),
                        1,
                    )

            # Save annotated full page
            page_out_path = out_dir / f"annotated_{seq_id}_p{p_idx+1}.png"
            cv2.imwrite(str(page_out_path), page_vis)

        per_sequence_results.append({
            "sequence_id": seq_id,
            "total_clusters": seq_clusters_count,
            "tails_detected": seq_tails_count,
        })

    total_elapsed = time.perf_counter() - total_start_time
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    summary_report = {
        "benchmark_environment": "Ubuntu 24.04 VM, 6 vCPUs, 8 GB RAM (CPU-only)",
        "sample_selection": {
            "source_split": "Fold 0 Training (splits/dev_splits.json)",
            "leakage_status": "Strict zero leakage (zero validation or test images)",
            "sequences": REPRESENTATIVE_DEV_SEQUENCES,
            "total_pages": len(REPRESENTATIVE_DEV_SEQUENCES) * 3,
        },
        "measurements": {
            "total_candidates_examined": total_candidates_examined,
            "status_distribution": status_counts,
            "plausible_bubble_regions_detected": (
                total_candidates_examined - status_counts.get("no_bubble", 0) - status_counts.get("background_bleed", 0)
            ),
            "plausible_bubble_rate": round(
                (total_candidates_examined - status_counts.get("no_bubble", 0) - status_counts.get("background_bleed", 0))
                / max(1, total_candidates_examined),
                3,
            ),
            "tail_detection_rate": round(status_counts.get("tail_detected", 0) / max(1, total_candidates_examined), 3),
            "elapsed_seconds": round(total_elapsed, 3),
            "seconds_per_page": round(total_elapsed / 12.0, 3),
            "peak_memory_mb": round(peak_mem / (1024 * 1024), 2),
        },
        "per_sequence_breakdown": per_sequence_results,
    }

    report_path = out_dir / "tail_poc_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    print(f"\nSaved proof-of-concept summary report to: {report_path}")
    return summary_report


def main() -> None:
    report = run_bubble_tail_poc()
    meas = report["measurements"]
    print("\n" + "=" * 70)
    print("PHASE 4 PROOF OF CONCEPT: SPEECH-BUBBLE TAIL GEOMETRY AUDIT")
    print("=" * 70)
    print(f"Total Candidate Clusters Examined: {meas['total_candidates_examined']}")
    print(f"Plausible Bubble Regions Detected: {meas['plausible_bubble_regions_detected']} ({meas['plausible_bubble_rate']*100:.1f}%)")
    print(f"Tails Detected: {meas['status_distribution']['tail_detected']} ({meas['tail_detection_rate']*100:.1f}%)")
    print("\nStatus Breakdown:")
    for status, count in meas["status_distribution"].items():
        pct = count / max(1, meas["total_candidates_examined"]) * 100
        print(f"  {status:<26}: {count:>3} ({pct:>5.1f}%)")
    print("\nCompute Footprint:")
    print(f"  Total Runtime (12 pages): {meas['elapsed_seconds']}s ({meas['seconds_per_page']}s/page)")
    print(f"  Peak Memory: {meas['peak_memory_mb']} MB")
    print("=" * 70)


if __name__ == "__main__":
    main()
