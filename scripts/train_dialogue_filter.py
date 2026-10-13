from __future__ import annotations

import argparse
import difflib
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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
    extract_cluster_features,
    LearnedClusterClassifier,
    LearnedNarrationClassifier,
)


def match_cluster_to_ground_truth(
    cluster_text: str,
    gt_lines: list,
    min_sim: float = 0.35,
) -> Tuple[bool, bool, float, str]:
    """Matches a cluster against ground truth dialogue lines on the page.
    
    Returns:
        (is_dialogue, is_narration, best_similarity, matched_gt_text)
    """
    c_clean = cluster_text.lower().strip()
    if not c_clean:
        return False, False, 0.0, ""

    best_sim = 0.0
    best_gt = None
    is_substr = False

    for gt in gt_lines:
        gt_clean = gt.text.lower().strip()
        sim = difflib.SequenceMatcher(None, c_clean, gt_clean).ratio()
        if c_clean in gt_clean or (len(c_clean) > 3 and gt_clean in c_clean):
            is_substr = True
            sim = max(sim, min(len(c_clean), len(gt_clean)) / max(len(c_clean), len(gt_clean)))
        if sim > best_sim:
            best_sim = sim
            best_gt = gt

    is_dialogue = (best_sim >= min_sim) or is_substr
    is_narration = False
    gt_text = ""

    if is_dialogue and best_gt is not None:
        gt_text = best_gt.text
        if best_gt.speaker == "NARRATION":
            is_narration = True

    return is_dialogue, is_narration, best_sim, gt_text


def train_models(
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

    # Assert strict leakage prevention
    assert not any(tid in val_ids for tid in train_ids), "LEAKAGE DETECTED: Val sequence found in train split!"

    print(f"Training on Fold 0: {len(train_ids)} sequences. Validation sequences strictly excluded.")

    loader = OtakuDatasetLoader(dataset_root)
    all_seqs = {s.sequence_id: s for s in loader.load_sequences()}
    all_labels = loader.load_labels(split="development")

    start_time = time.perf_counter()

    # Load OCR detections from cache or run engine
    ocr_detections: Dict[str, list] = {}
    if cache_path is not None and Path(cache_path).is_file():
        print(f"Loading cached training OCR detections from {cache_path}...")
        with open(cache_path, "rb") as f:
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
            if i % 10 == 0:
                print(f"Processed {i}/{len(train_ids)} training sequences...")

    # Extract features and targets
    X_filter: List[List[float]] = []
    y_filter: List[float] = []

    X_narr: List[List[float]] = []
    y_narr: List[float] = []

    total_clusters = 0

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
                total_clusters += 1
                feats = extract_cluster_features(c, page_width=pw, page_height=ph)
                is_dial, is_narr, sim, _ = match_cluster_to_ground_truth(c.text, gt_lines)

                X_filter.append(feats)
                y_filter.append(1.0 if is_dial else 0.0)

                # Narration dataset: only trained on dialogue clusters
                if is_dial:
                    X_narr.append(feats)
                    y_narr.append(1.0 if is_narr else 0.0)

    X_filt_arr = np.array(X_filter, dtype=float)
    y_filt_arr = np.array(y_filter, dtype=float)

    X_narr_arr = np.array(X_narr, dtype=float)
    y_narr_arr = np.array(y_narr, dtype=float)

    print(f"\nExtracted {len(X_filt_arr)} training clusters:")
    print(f"  Dialogue (Positive): {int(np.sum(y_filt_arr == 1.0))}")
    print(f"  Noise (Negative):    {int(np.sum(y_filt_arr == 0.0))}")
    print(f"  Narration candidates: {int(np.sum(y_narr_arr == 1.0))} out of {len(y_narr_arr)}")

    # 1. Fit Dialogue vs Noise Classifier
    print("\nFitting Dialogue vs Noise Classifier (L-BFGS-B L2 Logistic Regression)...")
    clf_filter = LearnedClusterClassifier(l2_reg=0.01, decision_threshold=0.35)
    clf_filter.fit(X_filt_arr, y_filt_arr)

    filt_probs = clf_filter.predict_proba(X_filt_arr)
    filt_preds = (filt_probs >= clf_filter.decision_threshold).astype(float)
    filt_acc = float(np.mean(filt_preds == y_filt_arr))
    tp = int(np.sum((filt_preds == 1) & (y_filt_arr == 1)))
    fp = int(np.sum((filt_preds == 1) & (y_filt_arr == 0)))
    tn = int(np.sum((filt_preds == 0) & (y_filt_arr == 0)))
    fn = int(np.sum((filt_preds == 0) & (y_filt_arr == 1)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    print(f"Dialogue Filter Training Results:")
    print(f"  Accuracy:  {filt_acc:.4f}")
    print(f"  Precision: {precision:.4f} (TP={tp}, FP={fp})")
    print(f"  Recall:    {recall:.4f} (TN={tn}, FN={fn})")
    print(f"  F1:        {f1:.4f}")

    filter_weights_path = output_dir / "dialogue_filter_weights.json"
    clf_filter.save_to_json(filter_weights_path)
    print(f"Saved dialogue filter weights to {filter_weights_path}")

    # 2. Fit Narration Classifier
    print("\nFitting Narration Classifier (L-BFGS-B L2 Logistic Regression)...")
    clf_narr = LearnedNarrationClassifier(l2_reg=0.01, threshold=0.45)
    clf_narr.fit(X_narr_arr, y_narr_arr)

    narr_probs = clf_narr.predict_proba(X_narr_arr)
    narr_preds = (narr_probs >= clf_narr.decision_threshold).astype(float)
    narr_acc = float(np.mean(narr_preds == y_narr_arr))
    n_tp = int(np.sum((narr_preds == 1) & (y_narr_arr == 1)))
    n_fp = int(np.sum((narr_preds == 1) & (y_narr_arr == 0)))
    n_tn = int(np.sum((narr_preds == 0) & (y_narr_arr == 0)))
    n_fn = int(np.sum((narr_preds == 0) & (y_narr_arr == 1)))
    n_prec = n_tp / max(n_tp + n_fp, 1)
    n_rec = n_tp / max(n_tp + n_fn, 1)
    n_f1 = 2 * n_prec * n_rec / max(n_prec + n_rec, 1e-8)

    print(f"Narration Classifier Training Results:")
    print(f"  Accuracy:  {narr_acc:.4f}")
    print(f"  Precision: {n_prec:.4f} (TP={n_tp}, FP={n_fp})")
    print(f"  Recall:    {n_rec:.4f} (TN={n_tn}, FN={n_fn})")
    print(f"  F1:        {n_f1:.4f}")

    narr_weights_path = output_dir / "narration_classifier_weights.json"
    clf_narr.save_to_json(narr_weights_path)
    print(f"Saved narration classifier weights to {narr_weights_path}")

    elapsed = time.perf_counter() - start_time
    print(f"\nTraining completed in {elapsed:.2f} seconds.")

    return {
        "train_sequences": len(train_ids),
        "total_clusters": total_clusters,
        "elapsed_seconds": round(elapsed, 3),
        "dialogue_filter": {
            "accuracy": round(filt_acc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "weights_path": str(filter_weights_path),
        },
        "narration_classifier": {
            "accuracy": round(narr_acc, 4),
            "precision": round(n_prec, 4),
            "recall": round(n_rec, 4),
            "f1": round(n_f1, 4),
            "weights_path": str(narr_weights_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train dialogue filter and narration classifier using permitted development data."
    )
    parser.add_argument(
        "--dataset-root",
        "--dataset_root",
        type=Path,
        default=project_root / "dataset",
        help="Path to dataset root directory (default: <project_root>/dataset)",
    )
    parser.add_argument(
        "--splits-file",
        "--splits_file",
        type=Path,
        default=project_root / "splits" / "dev_splits.json",
        help="Path to splits definition JSON (default: <project_root>/splits/dev_splits.json)",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        "--models-dir",
        type=Path,
        default=project_root / "models",
        help="Output directory to save trained weights JSON files (default: <project_root>/models)",
    )
    parser.add_argument(
        "--cache-path",
        "--cache_path",
        type=Path,
        default=None,
        help="Optional path to cached training OCR detections pickle file",
    )

    args = parser.parse_args()

    cache_arg = args.cache_path
    if cache_arg is None:
        # Check known scratch cache path if available locally
        fallback_cache = Path("/home/neo/.gemini/antigravity-ide/brain/d2f41da8-3938-47b3-a22c-63f0984ed8b1/scratch/fold0_train_raw_detections.pkl")
        if fallback_cache.is_file():
            cache_arg = fallback_cache

    train_models(
        dataset_root=args.dataset_root,
        splits_file=args.splits_file,
        cache_path=cache_arg,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

