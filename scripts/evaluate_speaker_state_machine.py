from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.loader import OtakuDatasetLoader
from src.eval.scorer import OtakuScorerBridge
from src.models.baseline_predictor import RapidOCRBaselinePredictor
from src.models.ocr import RapidOCREngine
from src.models.state_machine import ConversationalStateMachine


def run_evaluation(
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
    speaker_profile: Dict[str, Dict[str, Any]] = {}
    with open(labels_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            sid = rec["sequence_id"]
            if sid in fold0_val_ids:
                golds[sid] = rec["pages"]
                dialogue_speakers = set()
                for p in rec["pages"]:
                    for item in p:
                        spk = item["speaker"]
                        if spk != "NARRATION" and spk is not None:
                            dialogue_speakers.add(spk)
                speaker_profile[sid] = {
                    "num_speakers": len(dialogue_speakers),
                    "is_monologue": len(dialogue_speakers) <= 1,
                    "is_three_plus": len(dialogue_speakers) >= 3,
                }

    ocr_engine = RapidOCREngine()

    print("Pre-loading validation images into memory...")
    val_images = {}
    for seq in val_sequences:
        val_images[seq.sequence_id] = loader.load_images(seq, convert_rgb=True)

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
            "name": "Strategy F Baseline (Conversational/Spatial)",
            "short_name": "strategy_f",
            "speaker_attribution_mode": "conversational_spatial",
            "state_machine": None,
        },
        {
            "name": "Experimental: Conversational State Machine (CSM)",
            "short_name": "exp_csm",
            "speaker_attribution_mode": "conversational_state_machine",
            "state_machine": ConversationalStateMachine(
                dx_jump=0.45,
                dy_jump=0.15,
                proximity_dx=0.20,
                proximity_dy=0.12,
                enable_third_speaker=False,
            ),
        },
        {
            "name": "Experimental: CSM + Gated Third Speaker",
            "short_name": "exp_csm_third",
            "speaker_attribution_mode": "conversational_state_machine",
            "state_machine": ConversationalStateMachine(
                dx_jump=0.45,
                dy_jump=0.15,
                proximity_dx=0.20,
                proximity_dy=0.12,
                enable_third_speaker=True,
                third_speaker_min_seq_lines=15,
                third_speaker_min_page_lines=6,
                third_speaker_min_dist=0.45,
            ),
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
            enable_word_segmenter=True,
            word_segmenter_path=vocab_weights,
            speaker_attribution_mode=cfg["speaker_attribution_mode"],
            state_machine=cfg["state_machine"],
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

    # Compute Comparative Analysis (Strategy F vs Exp CSM)
    base_seqs = all_results["strategy_f"]["per_sequence"]
    exp_seqs = all_results["exp_csm"]["per_sequence"]

    diff_table = []
    improved = 0
    tied = 0
    regressed = 0

    monologue_base_f1 = []
    monologue_exp_f1 = []
    three_plus_base_f1 = []
    three_plus_exp_f1 = []

    for sid in fold0_val_ids:
        b_f1 = base_seqs[sid]["joint_f1"]
        e_f1 = exp_seqs[sid]["joint_f1"]
        diff = e_f1 - b_f1
        if abs(diff) < 1e-5:
            tied += 1
            status = "TIED"
        elif diff > 0:
            improved += 1
            status = "IMPROVED"
        else:
            regressed += 1
            status = "REGRESSED"

        prof = speaker_profile[sid]
        if prof["is_monologue"]:
            monologue_base_f1.append(b_f1)
            monologue_exp_f1.append(e_f1)
        if prof["is_three_plus"]:
            three_plus_base_f1.append(b_f1)
            three_plus_exp_f1.append(e_f1)

        diff_table.append({
            "sequence_id": sid,
            "speakers": prof["num_speakers"],
            "base_joint_f1": round(b_f1, 4),
            "exp_joint_f1": round(e_f1, 4),
            "diff_joint_f1": round(diff, 4),
            "base_bal_f1": round(base_seqs[sid]["balanced_joint_f1"], 4),
            "exp_bal_f1": round(exp_seqs[sid]["balanced_joint_f1"], 4),
            "diff_bal_f1": round(exp_seqs[sid]["balanced_joint_f1"] - base_seqs[sid]["balanced_joint_f1"], 4),
            "status": status,
        })

    all_results["comparison"] = {
        "improved_count": improved,
        "tied_count": tied,
        "regressed_count": regressed,
        "diff_table": diff_table,
        "monologue_sequences": {
            "count": len(monologue_base_f1),
            "base_mean_joint_f1": round(sum(monologue_base_f1)/len(monologue_base_f1), 4) if monologue_base_f1 else 0.0,
            "exp_mean_joint_f1": round(sum(monologue_exp_f1)/len(monologue_exp_f1), 4) if monologue_exp_f1 else 0.0,
        },
        "three_plus_sequences": {
            "count": len(three_plus_base_f1),
            "base_mean_joint_f1": round(sum(three_plus_base_f1)/len(three_plus_base_f1), 4) if three_plus_base_f1 else 0.0,
            "exp_mean_joint_f1": round(sum(three_plus_exp_f1)/len(three_plus_exp_f1), 4) if three_plus_exp_f1 else 0.0,
        },
    }

    # Save to logs
    output_log = logs_dir / "speaker_state_machine_eval.json"
    with open(output_log, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved full evaluation log to: {output_log}")

    return all_results


def main():
    results = run_evaluation()
    print("\n" + "=" * 90)
    print("PHASE 4: EXPERIMENT 2 CONTROLLED EVALUATION (FOLD 0 VALIDATION, 16 SEQUENCES)")
    print("=" * 90)
    header = f"{'Metric':<30} {'Strategy F (Base)':<20} {'CSM (Exp 2)':<20} {'CSM + Third Spk':<20}"
    print(header)
    print("-" * 90)
    metrics = [
        "balanced_joint_f1",
        "text_order_score",
        "matched_token_coverage",
        "speaker_accuracy_on_matched",
        "joint_f1",
    ]
    for m in metrics:
        v_base = results["strategy_f"]["macro_metrics"][m]
        v_csm = results["exp_csm"]["macro_metrics"][m]
        v_third = results["exp_csm_third"]["macro_metrics"][m]
        print(f"{m:<30} {v_base:<20.4f} {v_csm:<20.4f} {v_third:<20.4f}")
    print("-" * 90)
    r_base = results["strategy_f"]["total_elapsed_seconds"]
    r_csm = results["exp_csm"]["total_elapsed_seconds"]
    r_third = results["exp_csm_third"]["total_elapsed_seconds"]
    print(f"{'Runtime (s)':<30} {r_base:<20.2f} {r_csm:<20.2f} {r_third:<20.2f}")
    m_base = results["strategy_f"]["peak_memory_mb"]
    m_csm = results["exp_csm"]["peak_memory_mb"]
    m_third = results["exp_csm_third"]["peak_memory_mb"]
    print(f"{'Peak Memory (MB)':<30} {m_base:<20.2f} {m_csm:<20.2f} {m_third:<20.2f}")
    print("=" * 90)

    cmp = results["comparison"]
    print(f"\nSequences Improved : {cmp['improved_count']} / 16")
    print(f"Sequences Tied     : {cmp['tied_count']} / 16")
    print(f"Sequences Regressed: {cmp['regressed_count']} / 16")

    print("\nPer-Sequence Breakdown (Strategy F vs CSM):")
    print(f"{'Sequence ID':<24} {'Spk':<5} {'Base J-F1':<10} {'Exp J-F1':<10} {'Delta J-F1':<12} {'Base Bal':<10} {'Exp Bal':<10} {'Status':<10}")
    print("-" * 95)
    for r in cmp["diff_table"]:
        print(f"{r['sequence_id']:<24} {r['speakers']:<5} {r['base_joint_f1']:<10.4f} {r['exp_joint_f1']:<10.4f} {r['diff_joint_f1']:<+12.4f} {r['base_bal_f1']:<10.4f} {r['exp_bal_f1']:<10.4f} {r['status']:<10}")

    print("\nSubset Analysis:")
    mono = cmp["monologue_sequences"]
    print(f"  Monologues (N={mono['count']}): Base J-F1 = {mono['base_mean_joint_f1']:.4f}, Exp J-F1 = {mono['exp_mean_joint_f1']:.4f}")
    three = cmp["three_plus_sequences"]
    print(f"  3+ Speakers (N={three['count']}): Base J-F1 = {three['base_mean_joint_f1']:.4f}, Exp J-F1 = {three['exp_mean_joint_f1']:.4f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
