#!/usr/bin/env python3
"""Trains and compares narration classifier variants A, B, and C strictly on Fold 0 training data."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.loader import OtakuDatasetLoader
from src.models.ocr import RapidOCREngine
from src.models.clustering import cluster_text_regions
from src.models.filtering import (
    FEATURE_NAMES,
    NARRATION_FEATURE_NAMES,
    extract_cluster_features,
    extract_narration_features,
    LearnedNarrationClassifier,
)
from scripts.train_dialogue_filter import match_cluster_to_ground_truth


def train_variants(
    dataset_root: Path | str = project_root / "dataset",
    splits_file: Path | str = project_root / "splits" / "dev_splits.json",
    cache_path: Optional[Path | str] = None,
    output_dir: Path | str = project_root / "models",
) -> Dict[str, Any]:
    dataset_root = Path(dataset_root).resolve()
    splits_file = Path(splits_file).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(splits_file, "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    # Strictly Fold 0 training sequences (64 sequences)
    train_ids = splits_data["kfold_5"][0]["train"]
    val_ids = set(splits_data["kfold_5"][0]["val"])

    # Strict leakage prevention assertion
    assert not any(tid in val_ids for tid in train_ids), "LEAKAGE ERROR: Validation sequence in training split!"

    print(f"=== Narration Classifier Variants Training ===")
    print(f"Training on Fold 0: {len(train_ids)} sequences. Validation sequences strictly excluded.")

    loader = OtakuDatasetLoader(dataset_root)
    all_seqs = {s.sequence_id: s for s in loader.load_sequences()}
    all_labels = loader.load_labels(split="development")

    # Load OCR detections from cache or run engine
    ocr_detections: Dict[str, list] = {}
    if cache_path is not None and Path(cache_path).is_file():
        print(f"Loading cached training OCR detections from {cache_path}...")
        with open(cache_path, "rb") as f:
            ocr_detections = pickle.load(f)
    else:
        # Fallback to shared scratch cache if available
        fallback_cache = Path("/home/neo/.gemini/antigravity-ide/brain/d2f41da8-3938-47b3-a22c-63f0984ed8b1/scratch/fold0_train_raw_detections.pkl")
        if fallback_cache.is_file():
            print(f"Loading cached training OCR detections from {fallback_cache}...")
            with open(fallback_cache, "rb") as f:
                ocr_detections = pickle.load(f)
        else:
            print("Running RapidOCR engine on training sequences...")
            engine = RapidOCREngine()
            for i, seq_id in enumerate(train_ids, 1):
                seq_rec = all_seqs[seq_id]
                images = loader.load_images(seq_rec, convert_rgb=True)
                seq_pages = []
                for img in images:
                    seq_pages.append(engine.detect_and_recognize(img))
                ocr_detections[seq_id] = seq_pages

    X_base: List[List[float]] = []
    X_ext: List[List[float]] = []
    y_narr: List[float] = []

    for seq_id in train_ids:
        seq_rec = all_seqs[seq_id]
        gt_pages = all_labels[seq_id]
        pages_detections = ocr_detections[seq_id]

        for p_idx, page_regions in enumerate(pages_detections):
            img_path = seq_rec.image_paths[p_idx]
            with Image.open(img_path) as im:
                pw, ph = im.size

            clusters = cluster_text_regions(page_regions)
            gt_lines = gt_pages[p_idx]

            for c in clusters:
                is_dial, is_narr, sim, _ = match_cluster_to_ground_truth(c.text, gt_lines)
                if is_dial:
                    base_feats = extract_cluster_features(c, page_width=pw, page_height=ph)
                    ext_feats = extract_narration_features(c, page_width=pw, page_height=ph)
                    X_base.append(base_feats)
                    X_ext.append(ext_feats)
                    y_narr.append(1.0 if is_narr else 0.0)

    X_base_arr = np.array(X_base, dtype=float)
    X_ext_arr = np.array(X_ext, dtype=float)
    y_narr_arr = np.array(y_narr, dtype=float)

    n_pos = int(np.sum(y_narr_arr == 1.0))
    n_neg = int(np.sum(y_narr_arr == 0.0))
    print(f"\nExtracted {len(y_narr_arr)} dialogue clusters from 64 training sequences:")
    print(f"  Narration positive: {n_pos} ({n_pos / len(y_narr_arr) * 100:.2f}%)")
    print(f"  Dialogue positive:  {n_neg} ({n_neg / len(y_narr_arr) * 100:.2f}%)")

    results = {}

    # 1. Variant A: Original Unweighted 20-Feature Model
    print("\n--- Fitting Variant A (Original Unweighted Logistic Regression, 20 features) ---")
    clf_a = LearnedNarrationClassifier(l2_reg=0.01, threshold=0.25)
    clf_a.fit(X_base_arr, y_narr_arr, class_weight=None)
    probs_a = clf_a.predict_proba(X_base_arr)
    preds_a = (probs_a >= clf_a.decision_threshold).astype(float)
    tp_a = int(np.sum((preds_a == 1) & (y_narr_arr == 1)))
    fp_a = int(np.sum((preds_a == 1) & (y_narr_arr == 0)))
    fn_a = int(np.sum((preds_a == 0) & (y_narr_arr == 1)))
    prec_a = tp_a / max(tp_a + fp_a, 1)
    rec_a = tp_a / max(tp_a + fn_a, 1)
    f1_a = 2 * prec_a * rec_a / max(prec_a + rec_a, 1e-8)
    print(f"Variant A: Intercept = {clf_a.bias:.4f}")
    print(f"  At Th={clf_a.decision_threshold:.2f}: TP={tp_a}/{n_pos} (Rec={rec_a:.3f}), FP={fp_a} (Prec={prec_a:.3f}), F1={f1_a:.3f}")
    path_a = output_dir / "narration_classifier_weights_baseline.json"
    clf_a.save_to_json(path_a)
    print(f"Saved Variant A weights to {path_a}")
    results["variant_a"] = {"intercept": clf_a.bias, "tp": tp_a, "fp": fp_a, "precision": prec_a, "recall": rec_a, "f1": f1_a, "path": str(path_a)}

    # 2. Variant B: Class-Weighted 20-Feature Model
    print("\n--- Fitting Variant B (Class-Weighted Balanced, 20 features) ---")
    clf_b = LearnedNarrationClassifier(l2_reg=0.01, threshold=0.90)
    clf_b.fit(X_base_arr, y_narr_arr, class_weight="balanced")
    probs_b = clf_b.predict_proba(X_base_arr)
    preds_b = (probs_b >= clf_b.decision_threshold).astype(float)
    tp_b = int(np.sum((preds_b == 1) & (y_narr_arr == 1)))
    fp_b = int(np.sum((preds_b == 1) & (y_narr_arr == 0)))
    fn_b = int(np.sum((preds_b == 0) & (y_narr_arr == 1)))
    prec_b = tp_b / max(tp_b + fp_b, 1)
    rec_b = tp_b / max(tp_b + fn_b, 1)
    f1_b = 2 * prec_b * rec_b / max(prec_b + rec_b, 1e-8)
    print(f"Variant B: Intercept = {clf_b.bias:.4f}")
    print(f"  At Th={clf_b.decision_threshold:.2f}: TP={tp_b}/{n_pos} (Rec={rec_b:.3f}), FP={fp_b} (Prec={prec_b:.3f}), F1={f1_b:.3f}")
    path_b = output_dir / "narration_classifier_weights_variant_b.json"
    clf_b.save_to_json(path_b)
    print(f"Saved Variant B weights to {path_b}")
    results["variant_b"] = {"intercept": clf_b.bias, "tp": tp_b, "fp": fp_b, "precision": prec_b, "recall": rec_b, "f1": f1_b, "path": str(path_b)}

    # 3. Variant C: Class-Weighted 27-Feature Model (Lexical + Geometric)
    print("\n--- Fitting Variant C (Class-Weighted Balanced, 27 features: Lexical + Geometric) ---")
    clf_c = LearnedNarrationClassifier(
        l2_reg=0.01,
        threshold=0.90,
        feature_names=NARRATION_FEATURE_NAMES,
    )
    clf_c.fit(X_ext_arr, y_narr_arr, class_weight="balanced")
    probs_c = clf_c.predict_proba(X_ext_arr)
    preds_c = (probs_c >= clf_c.decision_threshold).astype(float)
    tp_c = int(np.sum((preds_c == 1) & (y_narr_arr == 1)))
    fp_c = int(np.sum((preds_c == 1) & (y_narr_arr == 0)))
    fn_c = int(np.sum((preds_c == 0) & (y_narr_arr == 1)))
    prec_c = tp_c / max(tp_c + fp_c, 1)
    rec_c = tp_c / max(tp_c + fn_c, 1)
    f1_c = 2 * prec_c * rec_c / max(prec_c + rec_c, 1e-8)
    print(f"Variant C: Intercept = {clf_c.bias:.4f}")
    print(f"  At Th={clf_c.decision_threshold:.2f}: TP={tp_c}/{n_pos} (Rec={rec_c:.3f}), FP={fp_c} (Prec={prec_c:.3f}), F1={f1_c:.3f}")
    path_c = output_dir / "narration_classifier_weights_variant_c.json"
    clf_c.save_to_json(path_c)
    print(f"Saved Variant C weights to {path_c}")
    results["variant_c"] = {"intercept": clf_c.bias, "tp": tp_c, "fp": fp_c, "precision": prec_c, "recall": rec_c, "f1": f1_c, "path": str(path_c)}

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train narration classifier variants strictly on Fold 0 training data.")
    parser.add_argument("--dataset-root", type=str, default=str(project_root / "dataset"))
    parser.add_argument("--splits-file", type=str, default=str(project_root / "splits" / "dev_splits.json"))
    parser.add_argument("--cache-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(project_root / "models"))
    args = parser.parse_args()

    train_variants(
        dataset_root=args.dataset_root,
        splits_file=args.splits_file,
        cache_path=args.cache_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
