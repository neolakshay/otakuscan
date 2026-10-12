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


def run_adapted_evaluation() -> Dict[str, Any]:
    dataset_root = project_root / "dataset"
    splits_file = project_root / "splits" / "dev_splits.json"
    
    predictions_dir = project_root / "predictions"
    logs_dir = project_root / "logs"
    models_dir = project_root / "models"
    predictions_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    with open(splits_file, "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    # Fold 0 validation sequences
    fold0_val_ids = splits_data["kfold_5"][0]["val"]
    print(f"Loaded Fold 0 with {len(fold0_val_ids)} validation sequences.")

    loader = OtakuDatasetLoader(dataset_root)
    pipeline = ValidationPipeline(dataset_root)

    filter_weights = models_dir / "dialogue_filter_weights.json"
    narr_weights = models_dir / "narration_classifier_weights.json"

    assert filter_weights.is_file(), f"Filter weights not found: {filter_weights}"
    assert narr_weights.is_file(), f"Narration weights not found: {narr_weights}"

    # Adapted predictor with learned dialogue filter and narration classifier
    predictor = RapidOCRBaselinePredictor(
        default_speaker="character1",
        min_confidence=0.0,
        enable_clustering=True,
        reading_order_mode="panel_aware",
        enable_dialogue_filter=True,
        dialogue_classifier_path=filter_weights,
        filter_threshold=0.35,
        enable_narration_classifier=True,
        narration_classifier_path=narr_weights,
        narration_threshold=0.25,
    )

    # Memory and latency tracking
    tracemalloc.start()
    start_time = time.perf_counter()

    all_sequences = {s.sequence_id: s for s in loader.load_sequences()}
    prediction_rows: List[Dict[str, Any]] = []
    
    raw_detections_count = 0
    retained_regions_count = 0
    narration_predicted_count = 0
    empty_pages_count = 0
    total_pages = 0
    runtime_errors: List[str] = []

    for seq_id in fold0_val_ids:
        seq_record = all_sequences[seq_id]
        total_pages += len(seq_record.image_paths)

        try:
            images = loader.load_images(seq_record, convert_rgb=True)
            for img in images:
                raw_regs = predictor.ocr_engine.detect_and_recognize(img)
                raw_detections_count += len(raw_regs)

            pred_pages = predictor.predict_images(images)

            for p in pred_pages:
                retained_regions_count += len(p)
                for item in p:
                    if item.get("speaker") == "NARRATION":
                        narration_predicted_count += 1
                if not p:
                    empty_pages_count += 1

            prediction_rows.append({
                "sequence_id": seq_id,
                "pages": pred_pages,
            })
        except Exception as e:
            runtime_errors.append(f"{seq_id}: {str(e)}")
            prediction_rows.append({
                "sequence_id": seq_id,
                "pages": [[] for _ in seq_record.image_paths],
            })

    elapsed_time = time.perf_counter() - start_time
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Validate output schema
    validate_submission_data(prediction_rows, expected_pages=3)

    # Write predictions artifact (immutably outside dataset/)
    pred_path = predictions_dir / "adapted_fold0.jsonl"
    with open(pred_path, "w", encoding="utf-8") as f:
        for row in prediction_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Saved adapted predictions to {pred_path}")

    # Official evaluation via pipeline bridge
    eval_report = pipeline.evaluate_split(
        val_sequence_ids=fold0_val_ids,
        predictor=lambda seq: next(r["pages"] for r in prediction_rows if r["sequence_id"] == seq.sequence_id),
    )

    # Load previous baseline reports for comparative audit
    baseline_log_path = logs_dir / "baseline_fold0_eval.json"
    with open(baseline_log_path, "r", encoding="utf-8") as f:
        baseline_log = json.load(f)
    baseline_macro = baseline_log["summary"]["macro_metrics_baseline"]

    clustered_log_path = logs_dir / "clustered_fold0_eval.json"
    with open(clustered_log_path, "r", encoding="utf-8") as f:
        clustered_log = json.load(f)
    clustered_macro = clustered_log["summary"]["macro_metrics_clustered"]

    panel_order_log_path = logs_dir / "panel_order_fold0_eval.json"
    with open(panel_order_log_path, "r", encoding="utf-8") as f:
        panel_order_log = json.load(f)
    panel_order_macro = panel_order_log["summary"]["macro_metrics_panel_order"]

    summary_results = {
        "fold": 0,
        "sequence_count": len(fold0_val_ids),
        "total_pages": total_pages,
        "elapsed_seconds": round(elapsed_time, 3),
        "seconds_per_page": round(elapsed_time / total_pages, 3),
        "peak_memory_mb": round(peak_mem / (1024 * 1024), 2),
        "raw_detections_count": raw_detections_count,
        "retained_regions_count": retained_regions_count,
        "narration_predicted_count": narration_predicted_count,
        "empty_pages_predicted": empty_pages_count,
        "runtime_errors": runtime_errors,
        "macro_metrics_adapted": eval_report["macro"],
        "macro_metrics_panel_order": panel_order_macro,
        "macro_metrics_clustered": clustered_macro,
        "macro_metrics_baseline": baseline_macro,
    }

    log_path = logs_dir / "adapted_fold0_eval.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary_results,
            "full_evaluation": eval_report,
        }, f, indent=2)
    print(f"Saved evaluation report to {log_path}")

    return summary_results


if __name__ == "__main__":
    results = run_adapted_evaluation()
    print("\n=== PHASE 3 ADAPTED PIPELINE EVALUATION: FOLD 0 ===")
    print(f"Runtime: {results['elapsed_seconds']}s ({results['seconds_per_page']}s/page)")
    print(f"Peak Memory: {results['peak_memory_mb']} MB")
    print(f"Raw Detections: {results['raw_detections_count']} -> Retained Predictions: {results['retained_regions_count']}")
    print(f"Narration Lines Predicted: {results['narration_predicted_count']}")
    print(f"Errors: {len(results['runtime_errors'])}")
    print("\n--- MACRO METRIC COMPARISON ---")
    print(f"{'Metric':<30} {'Panel-Aware':<14} {'Adapted':<14} {'Delta':<12}")
    print("-" * 72)
    for k in ("balanced_joint_f1", "text_order_score", "matched_token_coverage", "speaker_accuracy_on_matched", "joint_f1"):
        panel_val = results["macro_metrics_panel_order"].get(k, 0.0)
        adapted_val = results["macro_metrics_adapted"].get(k, 0.0)
        delta = adapted_val - panel_val
        print(f"{k:<30} {panel_val:<14.4f} {adapted_val:<14.4f} {delta:+12.4f}")
