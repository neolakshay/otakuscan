from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.loader import OtakuDatasetLoader
from src.eval.scorer import OtakuScorerBridge, validate_submission_data
from src.models.baseline_predictor import RapidOCRBaselinePredictor
from src.models.ocr import RapidOCREngine
from src.pipeline.validation import ValidationPipeline


def run_pipeline_comparison(
    dataset_root: Path | str = project_root / "dataset",
    splits_file: Path | str = project_root / "splits" / "dev_splits.json",
    models_dir: Path | str = project_root / "models",
    logs_dir: Path | str = project_root / "logs",
) -> Dict[str, Any]:
    dataset_root = Path(dataset_root).resolve()
    splits_file = Path(splits_file).resolve()
    models_dir = Path(models_dir).resolve()
    logs_dir = Path(logs_dir).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    with open(splits_file, "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    fold0_val_ids = splits_data["kfold_5"][0]["val"]
    print(f"Loaded Fold 0 with {len(fold0_val_ids)} validation sequences (48 pages).")

    loader = OtakuDatasetLoader(dataset_root)
    all_sequences = {s.sequence_id: s for s in loader.load_sequences()}
    val_sequences = [all_sequences[sid] for sid in fold0_val_ids]

    filter_weights = models_dir / "dialogue_filter_weights.json"
    narr_weights = models_dir / "narration_classifier_weights.json"
    vocab_weights = models_dir / "word_segmenter_vocab.json"

    assert filter_weights.is_file(), f"Filter weights not found: {filter_weights}"
    assert narr_weights.is_file(), f"Narration weights not found: {narr_weights}"
    assert vocab_weights.is_file(), f"Word segmenter vocab not found: {vocab_weights}"

    # Load ground truth labels for Fold 0 validation
    labels_file = dataset_root / "development" / "labels.jsonl"
    golds: Dict[str, List[List[Dict[str, str]]]] = {}
    with open(labels_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["sequence_id"] in fold0_val_ids:
                golds[rec["sequence_id"]] = rec["pages"]

    # Shared OCR engine to eliminate OCR re-computation latency across configs
    ocr_engine = RapidOCREngine()

    # Preload images for deterministic runtime benchmark
    print("Pre-loading validation images into memory...")
    val_images = {}
    for seq in val_sequences:
        val_images[seq.sequence_id] = loader.load_images(seq, convert_rgb=True)

    # Precompute OCR detections once across the 48 validation pages
    print("Running OCR extraction on 48 validation pages (one-time pre-pass)...")
    tracemalloc.start()
    ocr_start = time.perf_counter()
    ocr_cache = {}
    for seq_id, imgs in val_images.items():
        ocr_cache[seq_id] = [ocr_engine.detect_and_recognize(img) for img in imgs]
    ocr_elapsed = time.perf_counter() - ocr_start
    _, ocr_peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"OCR extraction completed in {ocr_elapsed:.2f}s ({ocr_elapsed/48.0:.3f}s/page).")

    configs = [
        {
            "name": "1. Adapted Pipeline (Baseline)",
            "short_name": "adapted_baseline",
            "enable_word_segmenter": False,
            "speaker_attribution_mode": "single",
        },
        {
            "name": "2. Adapted + Word Segmentation",
            "short_name": "adapted_word_seg",
            "enable_word_segmenter": True,
            "speaker_attribution_mode": "single",
        },
        {
            "name": "3. Word Seg + Strategy B (Spatial L/R)",
            "short_name": "strat_b_spatial",
            "enable_word_segmenter": True,
            "speaker_attribution_mode": "spatial_clustering",
        },
        {
            "name": "4. Word Seg + Strategy D (Turn Alternating)",
            "short_name": "strat_d_alternating",
            "enable_word_segmenter": True,
            "speaker_attribution_mode": "turn_alternating",
        },
        {
            "name": "5. Word Seg + Strategy F (Conversational/Spatial)",
            "short_name": "strat_f_conversational",
            "enable_word_segmenter": True,
            "speaker_attribution_mode": "conversational_spatial",
        },
    ]

    all_results = {}

    scorer_bridge = OtakuScorerBridge(dataset_root)

    for cfg in configs:
        print(f"\n--- Running: {cfg['name']} ---")
        predictor = RapidOCRBaselinePredictor(
            ocr_engine=ocr_engine,
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
            enable_word_segmenter=cfg["enable_word_segmenter"],
            word_segmenter_path=vocab_weights,
            speaker_attribution_mode=cfg["speaker_attribution_mode"],
        )

        tracemalloc.start()
        start_time = time.perf_counter()

        pred_rows = {}
        for seq_id, imgs in val_images.items():
            pred_pages = predictor.predict_images(imgs, cached_regions=ocr_cache[seq_id])
            pred_rows[seq_id] = pred_pages

        postprocess_elapsed = time.perf_counter() - start_time
        total_elapsed = ocr_elapsed + postprocess_elapsed
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        eval_report = scorer_bridge.evaluate_dict(golds, pred_rows)
        macro = eval_report["macro"]



        all_results[cfg["short_name"]] = {
            "name": cfg["name"],
            "total_elapsed_seconds": round(total_elapsed, 3),
            "postprocess_elapsed_seconds": round(postprocess_elapsed, 4),
            "seconds_per_page": round(total_elapsed / 48.0, 3),
            "peak_memory_mb": round((peak_mem + ocr_peak_mem) / (1024 * 1024), 2),
            "macro_metrics": macro,
            "per_sequence": eval_report["sequences"],
        }

        print(f"  Total: {total_elapsed:.2f}s ({total_elapsed/48.0:.3f}s/page) [Post: {postprocess_elapsed:.3f}s], Peak RAM: {(peak_mem + ocr_peak_mem)/(1024*1024):.1f} MB")
        for k in ("balanced_joint_f1", "text_order_score", "matched_token_coverage", "speaker_accuracy_on_matched", "joint_f1"):
            print(f"  {k:<30}: {macro[k]:.4f}")


    # Save to logs
    output_log = logs_dir / "pipeline_comparisons_fold0.json"
    with open(output_log, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved full comparisons log to: {output_log}")

    return all_results


def main() -> None:
    results = run_pipeline_comparison()
    print("\n" + "=" * 96)
    print("PHASE 5: FINAL COMPLETE PIPELINE COMPARISON ON FOLD 0 (16 sequences, 48 pages)")
    print("=" * 96)
    headers = ["Metric", "Config 1", "Config 2", "Config 3", "Config 4", "Config 5 (Best)"]
    print(f"{headers[0]:<30} {headers[1]:<12} {headers[2]:<12} {headers[3]:<12} {headers[4]:<12} {headers[5]:<14}")
    print("-" * 96)

    metric_keys = [
        ("balanced_joint_f1", "balanced_joint_f1"),
        ("text_order_score", "text_order_score"),
        ("matched_token_coverage", "matched_token_coverage"),
        ("speaker_accuracy_on_matched", "speaker_acc_matched"),
        ("joint_f1", "joint_f1"),
    ]

    cfg_keys = [
        "adapted_baseline",
        "adapted_word_seg",
        "strat_b_spatial",
        "strat_d_alternating",
        "strat_f_conversational",
    ]

    for m_key, m_label in metric_keys:
        vals = [results[ck]["macro_metrics"][m_key] for ck in cfg_keys]
        print(f"{m_key:<30} {vals[0]:<12.4f} {vals[1]:<12.4f} {vals[2]:<12.4f} {vals[3]:<12.4f} {vals[4]:<14.4f}")

    print("-" * 96)
    runtimes = [results[ck]["total_elapsed_seconds"] for ck in cfg_keys]
    print(f"{'Runtime (s)':<30} {runtimes[0]:<12.2f} {runtimes[1]:<12.2f} {runtimes[2]:<12.2f} {runtimes[3]:<12.2f} {runtimes[4]:<14.2f}")
    mems = [results[ck]["peak_memory_mb"] for ck in cfg_keys]
    print(f"{'Peak Memory (MB)':<30} {mems[0]:<12.2f} {mems[1]:<12.2f} {mems[2]:<12.2f} {mems[3]:<12.2f} {mems[4]:<14.2f}")
    print("=" * 96)


if __name__ == "__main__":
    main()
