from __future__ import annotations

import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Dict, Any, List

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.loader import OtakuDatasetLoader
from src.pipeline.validation import ValidationPipeline
from src.models.baseline_predictor import RapidOCRBaselinePredictor
from src.eval.scorer import validate_submission_data

def run_panel_order_evaluation() -> Dict[str, Any]:
    dataset_root = project_root / "dataset"
    splits_file = project_root / "splits" / "dev_splits.json"
    
    predictions_dir = project_root / "predictions"
    logs_dir = project_root / "logs"
    predictions_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    with open(splits_file, "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    # Fold 0 validation sequences
    fold0_val_ids = splits_data["kfold_5"][0]["val"]
    print(f"Loaded Fold 0 with {len(fold0_val_ids)} validation sequences.")

    loader = OtakuDatasetLoader(dataset_root)
    pipeline = ValidationPipeline(dataset_root)

    # Enable text-region clustering AND panel-aware reading order
    predictor = RapidOCRBaselinePredictor(
        default_speaker="character1",
        min_confidence=0.0,
        enable_clustering=True,
        reading_order_mode="panel_aware",
    )

    # Memory and latency tracking
    tracemalloc.start()
    start_time = time.perf_counter()

    all_sequences = {s.sequence_id: s for s in loader.load_sequences()}
    prediction_rows: List[Dict[str, Any]] = []
    
    raw_detections_count = 0
    clustered_regions_count = 0
    empty_pages_count = 0
    total_pages = 0
    runtime_errors: List[str] = []

    for seq_id in fold0_val_ids:
        seq_record = all_sequences[seq_id]
        try:
            images = loader.load_images(seq_record, convert_rgb=True)
            pages: List[List[Dict[str, str]]] = []
            
            for img in images:
                total_pages += 1
                raw_regions = predictor.ocr_engine.detect_and_recognize(img)
                raw_detections_count += len(raw_regions)
                
                if not raw_regions:
                    empty_pages_count += 1
                    pages.append([])
                    continue
                
                # Apply clustering
                clusters = predictor.clustering_fn(raw_regions)
                clustered_regions_count += len(clusters)
                
                # Sort clusters by panel-aware reading order
                ordered = predictor.reading_order_sorter(clusters)
                page_items = [
                    {"speaker": predictor.default_speaker, "text": c.text.strip()}
                    for c in ordered
                    if c.text.strip()
                ]
                pages.append(page_items)

            prediction_rows.append({
                "sequence_id": seq_id,
                "pages": pages,
            })
        except Exception as e:
            runtime_errors.append(f"{seq_id}: {str(e)}")

    elapsed_time = time.perf_counter() - start_time
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Validate output format
    validate_submission_data(prediction_rows, expected_pages=3)

    # Save to predictions/panel_order_fold0.jsonl (keeping previous predictions immutable)
    pred_path = predictions_dir / "panel_order_fold0.jsonl"
    with open(pred_path, "w", encoding="utf-8") as f:
        for row in prediction_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Saved {len(prediction_rows)} predictions to {pred_path}")

    # Evaluate using official scoring bridge
    eval_report = pipeline.evaluate_split(
        val_sequence_ids=fold0_val_ids,
        predictor=lambda seq: next(r["pages"] for r in prediction_rows if r["sequence_id"] == seq.sequence_id),
    )

    # Load preserved baseline and clustered metrics for comparison
    baseline_log_path = logs_dir / "baseline_fold0_eval.json"
    with open(baseline_log_path, "r", encoding="utf-8") as f:
        baseline_log = json.load(f)
    baseline_macro = baseline_log["summary"]["macro_metrics_baseline"]

    clustered_log_path = logs_dir / "clustered_fold0_eval.json"
    with open(clustered_log_path, "r", encoding="utf-8") as f:
        clustered_log = json.load(f)
    clustered_macro = clustered_log["summary"]["macro_metrics_clustered"]

    summary_results = {
        "fold": 0,
        "sequence_count": len(fold0_val_ids),
        "total_pages": total_pages,
        "elapsed_seconds": round(elapsed_time, 3),
        "seconds_per_page": round(elapsed_time / total_pages, 3),
        "peak_memory_mb": round(peak_mem / (1024 * 1024), 2),
        "raw_detections_count": raw_detections_count,
        "clustered_regions_count": clustered_regions_count,
        "compression_ratio": round(raw_detections_count / max(1, clustered_regions_count), 2),
        "average_clusters_per_page": round(clustered_regions_count / total_pages, 2),
        "empty_pages_predicted": empty_pages_count,
        "runtime_errors": runtime_errors,
        "macro_metrics_panel_order": eval_report["macro"],
        "macro_metrics_clustered": clustered_macro,
        "macro_metrics_baseline": baseline_macro,
    }

    log_path = logs_dir / "panel_order_fold0_eval.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary_results,
            "full_evaluation": eval_report,
        }, f, indent=2)
    print(f"Saved evaluation report to {log_path}")

    return summary_results

if __name__ == "__main__":
    results = run_panel_order_evaluation()
    print("\n=== PANEL-AWARE READING ORDER EVALUATION RESULTS: FOLD 0 ===")
    print(f"Runtime: {results['elapsed_seconds']}s ({results['seconds_per_page']}s/page)")
    print(f"Peak Memory: {results['peak_memory_mb']} MB")
    print(f"Raw Detections: {results['raw_detections_count']} -> Clustered Bubbles: {results['clustered_regions_count']}")
    print(f"Average Clusters/Page: {results['average_clusters_per_page']}")
    print(f"Errors: {len(results['runtime_errors'])}")
    print("\n--- MACRO METRIC COMPARISON ---")
    print(f"{'Metric':<30} {'Clustered':<16} {'Panel-Aware':<16} {'Delta':<12}")
    print("-" * 74)
    for k in ("balanced_joint_f1", "text_order_score", "matched_token_coverage", "speaker_accuracy_on_matched", "joint_f1"):
        clust_val = results["macro_metrics_clustered"].get(k, 0.0)
        panel_val = results["macro_metrics_panel_order"].get(k, 0.0)
        delta = panel_val - clust_val
        print(f"{k:<30} {clust_val:<16.4f} {panel_val:<16.4f} {delta:+12.4f}")
