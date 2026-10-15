#!/usr/bin/env python3
"""Evaluates narration classifier variants on Fold 0 validation with official scoring and diagnostics."""
from __future__ import annotations

import json
import pickle
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.loader import OtakuDatasetLoader
from src.models.ocr import RapidOCREngine
from src.models.baseline_predictor import RapidOCRBaselinePredictor
from src.eval.scorer import OtakuScorerBridge
from dataset.score import (
    TOKEN,
    norm,
    tokens,
    align_tokens,
    score_sequence,
)


def evaluate_narration_variants(
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

    val_ids = splits_data["kfold_5"][0]["val"]
    print(f"Loaded Fold 0 with {len(val_ids)} validation sequences (48 pages).")

    loader = OtakuDatasetLoader(dataset_root)
    all_seqs = {s.sequence_id: s for s in loader.load_sequences()}
    val_seqs = [all_seqs[sid] for sid in val_ids]

    # Load ground truth from dataset/development/labels.jsonl
    golds = {}
    labels_file = dataset_root / "development" / "labels.jsonl"
    with open(labels_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec["sequence_id"] in val_ids:
                    golds[rec["sequence_id"]] = rec["pages"]

    ocr_engine = RapidOCREngine()

    # Pre-load validation images
    print("Pre-loading validation images into memory...")
    val_images = {}
    for seq_rec in val_seqs:
        val_images[seq_rec.sequence_id] = loader.load_images(seq_rec, convert_rgb=True)

    # One-time OCR pass
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

    filter_weights = models_dir / "dialogue_filter_weights.json"
    vocab_weights = models_dir / "word_segmenter_vocab.json"

    variants = [
        {
            "name": "Variant A: Original Baseline Classifier (20 feats, Th=0.25)",
            "short_name": "variant_a",
            "weights_path": models_dir / "narration_classifier_weights_baseline.json",
            "threshold": 0.25,
        },
        {
            "name": "Variant B: Class-Weighted Classifier (20 feats, Th=0.90)",
            "short_name": "variant_b",
            "weights_path": models_dir / "narration_classifier_weights_variant_b.json",
            "threshold": 0.90,
        },
        {
            "name": "Variant C: Class-Weighted Classifier (27 feats: Lexical+Geometric, Th=0.90)",
            "short_name": "variant_c",
            "weights_path": models_dir / "narration_classifier_weights_variant_c.json",
            "threshold": 0.90,
        },
    ]

    scorer_bridge = OtakuScorerBridge(dataset_root)
    all_results = {}

    for var in variants:
        print(f"\n================================================================================")
        print(f"RUNNING: {var['name']}")
        print(f"Weights: {var['weights_path'].name}, Decision Threshold: {var['threshold']}")
        print(f"================================================================================")

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
            narration_classifier_path=var["weights_path"],
            narration_threshold=var["threshold"],
            enable_word_segmenter=True,
            word_segmenter_path=vocab_weights,
            speaker_attribution_mode="conversational_spatial",
        )

        tracemalloc.start()
        start_time = time.perf_counter()

        pred_rows = {}
        for seq_id, imgs in val_images.items():
            pred_pages = predictor.predict_images(imgs, cached_regions=ocr_cache[seq_id])
            pred_rows[seq_id] = pred_pages

        post_elapsed = time.perf_counter() - start_time
        tot_elapsed = ocr_elapsed + post_elapsed
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        eval_report = scorer_bridge.evaluate_dict(golds, pred_rows)
        macro = eval_report["macro"]

        # Compute detailed token-level narration diagnostics
        gold_narr_pred_narr = 0  # TP
        gold_dial_pred_narr = 0  # FP
        gold_narr_pred_dial = 0  # FN (missed narration)
        gold_dial_pred_dial = 0  # TN
        total_gold_narr_tokens = 0
        total_gold_dial_tokens = 0
        total_pred_narr_tokens = 0

        seq_diagnostics = {}

        for seq_id in val_ids:
            g_pages = golds[seq_id]
            p_pages = pred_rows[seq_id]

            s_gold_narr = 0
            s_pred_narr = 0
            s_tp_narr = 0
            s_fp_narr = 0
            s_fn_narr = 0

            for p_idx, (g_page, p_page) in enumerate(zip(g_pages, p_pages)):
                g_toks = tokens(g_page)
                p_toks = tokens(p_page)

                for _, spk in g_toks:
                    if spk == "NARRATION":
                        total_gold_narr_tokens += 1
                        s_gold_narr += 1
                    elif spk is not None:
                        total_gold_dial_tokens += 1

                for _, spk in p_toks:
                    if spk == "NARRATION":
                        total_pred_narr_tokens += 1
                        s_pred_narr += 1

                pairs = align_tokens(g_toks, p_toks)
                for g_item, p_item, w in pairs:
                    g_spk = g_item[1]
                    p_spk = p_item[1]
                    if g_spk == "NARRATION" and p_spk == "NARRATION":
                        gold_narr_pred_narr += 1
                        s_tp_narr += 1
                    elif g_spk != "NARRATION" and p_spk == "NARRATION":
                        gold_dial_pred_narr += 1
                        s_fp_narr += 1
                    elif g_spk == "NARRATION" and p_spk != "NARRATION":
                        gold_narr_pred_dial += 1
                        s_fn_narr += 1
                    elif g_spk != "NARRATION" and p_spk != "NARRATION":
                        gold_dial_pred_dial += 1

            seq_diagnostics[seq_id] = {
                "gold_narr_tokens": s_gold_narr,
                "pred_narr_tokens": s_pred_narr,
                "tp_narr": s_tp_narr,
                "fp_narr": s_fp_narr,
                "fn_narr": s_fn_narr,
                "balanced_joint_f1": eval_report["sequences"][seq_id]["balanced_joint_f1"],
                "joint_f1": eval_report["sequences"][seq_id]["joint_f1"],
                "speaker_accuracy_on_matched": eval_report["sequences"][seq_id]["speaker_accuracy_on_matched"],
            }

        n_prec = gold_narr_pred_narr / max(gold_narr_pred_narr + gold_dial_pred_narr, 1)
        n_rec = gold_narr_pred_narr / max(gold_narr_pred_narr + gold_narr_pred_dial, 1)
        n_f1 = 2 * n_prec * n_rec / max(n_prec + n_rec, 1e-8)

        print(f"  Official Macro Metrics:")
        for k in ("balanced_joint_f1", "text_order_score", "matched_token_coverage", "speaker_accuracy_on_matched", "joint_f1"):
            print(f"    {k:<30}: {macro[k]:.4f}")

        print(f"\n  Narration Diagnostics (Matched Tokens):")
        print(f"    TP (Gold Narr -> Pred Narr)    : {gold_narr_pred_narr}")
        print(f"    FP (Gold Dial -> Pred Narr)    : {gold_dial_pred_narr} (False Narration)")
        print(f"    FN (Gold Narr -> Pred Dial)    : {gold_narr_pred_dial} (Missed Narration)")
        print(f"    TN (Gold Dial -> Pred Dial)    : {gold_dial_pred_dial}")
        print(f"    Narration Precision            : {n_prec:.4f} ({n_prec*100:.1f}%)")
        print(f"    Narration Recall               : {n_rec:.4f} ({n_rec*100:.1f}%)")
        print(f"    Narration F1                   : {n_f1:.4f}")

        all_results[var["short_name"]] = {
            "name": var["name"],
            "threshold": var["threshold"],
            "runtime_seconds": round(tot_elapsed, 3),
            "postprocess_seconds": round(post_elapsed, 4),
            "peak_memory_mb": round((peak_mem + ocr_peak_mem) / (1024 * 1024), 2),
            "macro_metrics": macro,
            "narration_diagnostics": {
                "tp": gold_narr_pred_narr,
                "fp": gold_dial_pred_narr,
                "fn": gold_narr_pred_dial,
                "tn": gold_dial_pred_dial,
                "precision": round(n_prec, 4),
                "recall": round(n_rec, 4),
                "f1": round(n_f1, 4),
                "total_gold_narr_tokens": total_gold_narr_tokens,
                "total_pred_narr_tokens": total_pred_narr_tokens,
            },
            "per_sequence": seq_diagnostics,
        }

    # Comparative analysis against Variant A (Baseline)
    base_metrics = all_results["variant_a"]["macro_metrics"]
    base_per_seq = all_results["variant_a"]["per_sequence"]

    print("\n" + "=" * 80)
    print("COMPARATIVE EVALUATION SUMMARY ACROSS VARIANTS ON FOLD 0 (16 sequences, 48 pages)")
    print("=" * 80)
    print(f"{'Metric':<32} {'Variant A (Base)':<18} {'Variant B':<18} {'Variant C':<18}")
    print("-" * 80)
    for m in ("balanced_joint_f1", "text_order_score", "matched_token_coverage", "speaker_accuracy_on_matched", "joint_f1"):
        val_a = all_results["variant_a"]["macro_metrics"][m]
        val_b = all_results["variant_b"]["macro_metrics"][m]
        val_c = all_results["variant_c"]["macro_metrics"][m]
        print(f"{m:<32} {val_a:<18.4f} {val_b:<18.4f} {val_c:<18.4f}")

    print("-" * 80)
    print(f"{'Narration Precision':<32} {all_results['variant_a']['narration_diagnostics']['precision']:<18.4f} {all_results['variant_b']['narration_diagnostics']['precision']:<18.4f} {all_results['variant_c']['narration_diagnostics']['precision']:<18.4f}")
    print(f"{'Narration Recall':<32} {all_results['variant_a']['narration_diagnostics']['recall']:<18.4f} {all_results['variant_b']['narration_diagnostics']['recall']:<18.4f} {all_results['variant_c']['narration_diagnostics']['recall']:<18.4f}")
    print(f"{'Narration F1':<32} {all_results['variant_a']['narration_diagnostics']['f1']:<18.4f} {all_results['variant_b']['narration_diagnostics']['f1']:<18.4f} {all_results['variant_c']['narration_diagnostics']['f1']:<18.4f}")
    print(f"{'Recovered Narration (TP)':<32} {all_results['variant_a']['narration_diagnostics']['tp']:<18d} {all_results['variant_b']['narration_diagnostics']['tp']:<18d} {all_results['variant_c']['narration_diagnostics']['tp']:<18d}")
    print(f"{'False Narration (FP)':<32} {all_results['variant_a']['narration_diagnostics']['fp']:<18d} {all_results['variant_b']['narration_diagnostics']['fp']:<18d} {all_results['variant_c']['narration_diagnostics']['fp']:<18d}")
    print(f"{'Missed Narration (FN)':<32} {all_results['variant_a']['narration_diagnostics']['fn']:<18d} {all_results['variant_b']['narration_diagnostics']['fn']:<18d} {all_results['variant_c']['narration_diagnostics']['fn']:<18d}")
    print("=" * 80)

    # Per-sequence comparison for Variant C vs Baseline
    c_improved = 0
    c_tied = 0
    c_regressed = 0
    print("\nPer-Sequence Changes (Variant C vs Baseline):")
    for sid in val_ids:
        b_f1_base = base_per_seq[sid]["balanced_joint_f1"]
        b_f1_c = all_results["variant_c"]["per_sequence"][sid]["balanced_joint_f1"]
        diff = b_f1_c - b_f1_base
        status = "TIED"
        if diff > 0.0001:
            status = "IMPROVED"
            c_improved += 1
        elif diff < -0.0001:
            status = "REGRESSED"
            c_regressed += 1
        else:
            c_tied += 1
        print(f"  {sid}: Base={b_f1_base:.4f} -> VarC={b_f1_c:.4f} (diff={diff:+.4f}) [{status}]")

    print(f"\nSequence Outcomes: Improved={c_improved}, Tied={c_tied}, Regressed={c_regressed}")

    out_file = logs_dir / "narration_evaluation_fold0.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved evaluation results to {out_file}")

    return all_results


if __name__ == "__main__":
    evaluate_narration_variants()
