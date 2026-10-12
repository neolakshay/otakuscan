from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Any, List

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.loader import OtakuDatasetLoader

def analyze_errors() -> Dict[str, Any]:
    logs_dir = project_root / "logs"
    preds_dir = project_root / "predictions"
    dataset_root = project_root / "dataset"

    with open(logs_dir / "baseline_fold0_eval.json") as f:
        base_eval = json.load(f)
    with open(logs_dir / "clustered_fold0_eval.json") as f:
        clust_eval = json.load(f)

    base_seqs = base_eval["full_evaluation"]["sequences"]
    clust_seqs = clust_eval["full_evaluation"]["sequences"]

    metrics = [
        "text_order_score",
        "matched_token_coverage",
        "speaker_accuracy_on_matched",
        "balanced_joint_f1",
        "joint_f1"
    ]

    per_sequence_comparison: Dict[str, Dict[str, Any]] = {}
    for sid in sorted(base_seqs.keys()):
        b = base_seqs[sid]
        c = clust_seqs[sid]
        seq_entry: Dict[str, Any] = {}
        for m in metrics:
            bv = b.get(m)
            cv = c.get(m)
            delta = (cv - bv) if (bv is not None and cv is not None) else None
            seq_entry[m] = {
                "baseline": bv,
                "clustered": cv,
                "delta": delta
            }
        per_sequence_comparison[sid] = seq_entry

    # Load predictions and ground truth for deep error breakdown
    loader = OtakuDatasetLoader(dataset_root)
    gt_labels = loader.load_labels()
    
    with open(preds_dir / "clustered_fold0.jsonl") as f:
        clust_preds = {json.loads(line)["sequence_id"]: json.loads(line)["pages"] for line in f}

    total_gt_bubbles = sum(len(p) for sid in clust_preds for p in gt_labels[sid])
    total_pred_clusters = sum(len(p) for sid in clust_preds for p in clust_preds[sid])

    analysis_data = {
        "macro_summary": {
            "baseline": base_eval["summary"]["macro_metrics_baseline"],
            "clustered": clust_eval["summary"]["macro_metrics_clustered"],
        },
        "per_sequence": per_sequence_comparison,
        "volume_stats": {
            "total_gt_dialogue_bubbles": total_gt_bubbles,
            "total_raw_detections": clust_eval["summary"]["raw_detections_count"],
            "total_predicted_clusters": total_pred_clusters,
            "average_gt_bubbles_per_page": round(total_gt_bubbles / 48, 2),
            "average_pred_clusters_per_page": round(total_pred_clusters / 48, 2),
        }
    }

    out_log = logs_dir / "phase2_error_analysis.json"
    with open(out_log, "w", encoding="utf-8") as f:
        json.dump(analysis_data, f, indent=2)
    print(f"Saved analysis JSON to {out_log}")

    return analysis_data

if __name__ == "__main__":
    analyze_errors()
