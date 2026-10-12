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

def run_fold0_evaluation() -> Dict[str, Any]:
    project_root = Path(__file__).resolve().parent.parent
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
    predictor = RapidOCRBaselinePredictor(default_speaker="character1", min_confidence=0.0)

    # Start memory and time tracking
    tracemalloc.start()
    start_time = time.perf_counter()

    all_sequences = {s.sequence_id: s for s in loader.load_sequences()}
    prediction_rows: List[Dict[str, Any]] = []
    
    total_detected_regions = 0
    empty_pages_count = 0
    total_pages = 0
    runtime_errors: List[str] = []

    for seq_id in fold0_val_ids:
        seq_record = all_sequences[seq_id]
        try:
            pages = predictor.predict_sequence(seq_record, loader)
            for page in pages:
                total_pages += 1
                if len(page) == 0:
                    empty_pages_count += 1
                else:
                    total_detected_regions += len(page)

            prediction_rows.append({
                "sequence_id": seq_id,
                "pages": pages,
            })
        except Exception as e:
            runtime_errors.append(f"{seq_id}: {str(e)}")

    elapsed_time = time.perf_counter() - start_time
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Validate predictions formatting
    validate_submission_data(prediction_rows, expected_pages=3)

    # Save predictions outside dataset/
    pred_path = predictions_dir / "baseline_fold0.jsonl"
    with open(pred_path, "w", encoding="utf-8") as f:
        for row in prediction_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Saved {len(prediction_rows)} predictions to {pred_path}")

    # Evaluate using official scoring bridge
    eval_report = pipeline.evaluate_split(
        val_sequence_ids=fold0_val_ids,
        predictor=lambda seq: next(r["pages"] for r in prediction_rows if r["sequence_id"] == seq.sequence_id),
    )

    # Also evaluate empty prediction baseline on the same fold for direct comparison
    empty_report = pipeline.evaluate_split(
        val_sequence_ids=fold0_val_ids,
        predictor=lambda seq: [[], [], []],
    )

    summary_results = {
        "fold": 0,
        "sequence_count": len(fold0_val_ids),
        "total_pages": total_pages,
        "elapsed_seconds": round(elapsed_time, 3),
        "seconds_per_sequence": round(elapsed_time / len(fold0_val_ids), 3),
        "seconds_per_page": round(elapsed_time / total_pages, 3),
        "peak_memory_mb": round(peak_mem / (1024 * 1024), 2),
        "total_detected_regions": total_detected_regions,
        "average_regions_per_page": round(total_detected_regions / total_pages, 2),
        "empty_pages_predicted": empty_pages_count,
        "runtime_errors": runtime_errors,
        "macro_metrics_baseline": eval_report["macro"],
        "macro_metrics_empty": empty_report["macro"],
    }

    log_path = logs_dir / "baseline_fold0_eval.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary_results,
            "full_evaluation": eval_report,
            "empty_evaluation": empty_report,
        }, f, indent=2)
    print(f"Saved evaluation report to {log_path}")

    return summary_results

if __name__ == "__main__":
    results = run_fold0_evaluation()
    print("\n=== EVALUATION RESULTS: FOLD 0 ===")
    print(f"Runtime: {results['elapsed_seconds']}s ({results['seconds_per_page']}s/page)")
    print(f"Peak Traced Memory: {results['peak_memory_mb']} MB")
    print(f"Detected Regions: {results['total_detected_regions']} (avg {results['average_regions_per_page']} / page)")
    print(f"Empty Pages Predicted: {results['empty_pages_predicted']} / {results['total_pages']}")
    print(f"Errors: {len(results['runtime_errors'])}")
    print("\n--- MACRO METRIC COMPARISON ---")
    print(f"{'Metric':<30} {'Empty Baseline':<18} {'RapidOCR Baseline':<18}")
    print("-" * 66)
    for k in ("balanced_joint_f1", "text_order_score", "matched_token_coverage", "speaker_accuracy_on_matched", "joint_f1"):
        empty_val = results["macro_metrics_empty"].get(k)
        base_val = results["macro_metrics_baseline"].get(k)
        empty_str = f"{empty_val:.4f}" if empty_val is not None else "None"
        base_str = f"{base_val:.4f}" if base_val is not None else "None"
        print(f"{k:<30} {empty_str:<18} {base_str:<18}")
