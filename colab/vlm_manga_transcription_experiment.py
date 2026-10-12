"""Google Colab VLM Pilot Experiment: Vision-Language Model Adaptation for Manga
OtakuScan Phase 3: ML Adaptation with GPU Compute

This script runs on Google Colab (with T4/V100/A100/L4 GPU) to evaluate Open-Weight VLMs
(e.g., Qwen/Qwen2-VL-2B-Instruct) on difficult manga extraction cases against RapidOCR.

Key Objectives:
1. Benchmark VLM transcription accuracy against RapidOCR on controlled development pages.
2. Measure GPU vs CPU latency and memory requirements.
3. Compare token extraction quality against ground-truth development dialogue.
4. Formulate teacher-distillation findings for lightweight local inference.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

colab_root = Path(__file__).resolve().parent.parent
if str(colab_root) not in sys.path:
    sys.path.insert(0, str(colab_root))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.data.loader import OtakuDatasetLoader
from src.models.ocr import RapidOCREngine
from src.models.clustering import cluster_text_regions
from src.models.reading_order import sort_panel_aware_reading_order


MANGA_PROMPT_TEMPLATE = """You are an expert manga transcription assistant.
Analyze this manga page image and transcribe all character speech and narrative caption text.
Rules:
1. Transcribe dialogue in manga reading order: top-to-bottom, right-to-left within each panel column.
2. Distinguish character dialogue from narrative captions: mark caption boxes with speaker 'NARRATION'.
3. Exclude background sound effects (SFX, onomatopoeia like BOOM, CLACK, SNAP).
4. Output format: A JSON list of objects with keys 'speaker' (either 'character' or 'NARRATION') and 'text'.
Example:
[
  {"speaker": "character", "text": "HELLO WORLD!"},
  {"speaker": "NARRATION", "text": "MEANWHILE IN TOKYO..."}
]
Output ONLY the JSON list without conversational preamble.
"""


def parse_vlm_output(raw_output: str) -> List[Dict[str, str]]:
    """Robustly extracts structured dialogue items from raw VLM generations.
    
    Handles:
    1. Direct JSON array string
    2. Markdown fenced code blocks (```json ... ```)
    3. Free-form text lines with speaker prefixes (fallback)
    """
    text = (raw_output or "").strip()
    if not text:
        return []

    # 1. Attempt to find JSON array in markdown code block or brackets
    json_match = re.search(r"```(?:json)?\s*(\[\s*\{.*?\}\s*\])\s*```", text, re.DOTALL)
    if not json_match:
        json_match = re.search(r"(\[\s*\{.*?\}\s*\])", text, re.DOTALL)

    if json_match:
        json_str = json_match.group(1).strip()
        try:
            data = json.loads(json_str)
            if isinstance(data, list):
                valid_items: List[Dict[str, str]] = []
                for item in data:
                    if isinstance(item, dict) and "text" in item:
                        txt = str(item.get("text", "")).strip()
                        if txt:
                            spk = str(item.get("speaker", "character")).strip()
                            if "narr" in spk.lower():
                                spk = "NARRATION"
                            else:
                                spk = "character"
                            valid_items.append({"speaker": spk, "text": txt})
                if valid_items:
                    return valid_items
        except json.JSONDecodeError:
            pass

    # 2. Line-by-line fallback parsing
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    extracted: List[Dict[str, str]] = []

    for line in lines:
        if line.startswith("```") or line in ("[", "]", "{", "}"):
            continue
        # Skip conversational meta-text
        if any(line.lower().startswith(p) for p in ("here is", "sure", "output:", "transcription:", "dialogue:")):
            continue

        speaker = "character"
        cleaned_line = line

        # Check for [NARRATION] or Narrator prefix
        if re.search(r"^\[?narrat(?:ion|or)\]?[:\s-]*", line, re.IGNORECASE):
            speaker = "NARRATION"
            cleaned_line = re.sub(r"^\[?narrat(?:ion|or)\]?[:\s-]*", "", line, flags=re.IGNORECASE).strip()
        elif re.search(r"^\[?character\d*\]?[:\s-]*", line, re.IGNORECASE):
            cleaned_line = re.sub(r"^\[?character\d*\]?[:\s-]*", "", line, flags=re.IGNORECASE).strip()
        elif ":" in line:
            parts = line.split(":", 1)
            speaker_prefix = parts[0].strip().lower()
            if "narr" in speaker_prefix:
                speaker = "NARRATION"
            cleaned_line = parts[1].strip()

        # Clean quotes
        cleaned_line = cleaned_line.strip("\"' \t-•*")
        if cleaned_line:
            extracted.append({"speaker": speaker, "text": cleaned_line})

    return extracted


def select_pilot_training_pages(
    project_root: Path | str,
    n_pages: int = 3,
) -> List[Dict[str, Any]]:
    """Selects a small, controlled set of pilot development pages strictly from Fold 0 train split.
    
    Guarantees:
    - Never touches Fold 0 validation sequences.
    - Never touches unlabelled test sequences.
    """
    root = Path(project_root).resolve()
    splits_path = root / "splits" / "dev_splits.json"
    with open(splits_path, "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    train_ids = splits_data["kfold_5"][0]["train"]
    val_ids = set(splits_data["kfold_5"][0]["val"])

    loader = OtakuDatasetLoader(root / "dataset")
    seq_records = {s.sequence_id: s for s in loader.load_sequences()}
    all_labels = loader.load_labels(split="development")

    pilot_pages: List[Dict[str, Any]] = []

    for seq_id in train_ids:
        assert seq_id not in val_ids, f"CRITICAL: Leakage detected on {seq_id}"
        seq_rec = seq_records[seq_id]
        gt_pages = all_labels[seq_id]

        for p_idx, img_path in enumerate(seq_rec.image_paths):
            gt_lines = gt_pages[p_idx]
            # Choose pages with non-empty dialogue
            if gt_lines and len(gt_lines) >= 3:
                pilot_pages.append({
                    "sequence_id": seq_id,
                    "page_index": p_idx,
                    "image_path": str(img_path),
                    "ground_truth": [
                        {"speaker": line.speaker, "text": line.text}
                        for line in gt_lines
                    ],
                })
            if len(pilot_pages) >= n_pages:
                return pilot_pages

    return pilot_pages


def compute_transcription_token_match(
    predicted: List[Dict[str, str]],
    ground_truth: List[Dict[str, str]],
) -> Dict[str, float]:
    """Computes token-level precision, recall, and F1 against ground truth dialogue."""
    pred_tokens = [t.lower().strip() for item in predicted for t in item["text"].split()]
    gt_tokens = [t.lower().strip() for item in ground_truth for t in item["text"].split()]

    if not pred_tokens and not gt_tokens:
        return {"token_precision": 1.0, "token_recall": 1.0, "token_f1": 1.0}
    if not pred_tokens or not gt_tokens:
        return {"token_precision": 0.0, "token_recall": 0.0, "token_f1": 0.0}

    # Soft token overlap matching using difflib
    matched_gt = set()
    matched_pred = 0

    for p_tok in pred_tokens:
        found_match = False
        for g_idx, g_tok in enumerate(gt_tokens):
            if g_idx in matched_gt:
                continue
            sim = difflib.SequenceMatcher(None, p_tok, g_tok).ratio()
            if sim >= 0.75:
                matched_gt.add(g_idx)
                found_match = True
                break
        if found_match:
            matched_pred += 1

    precision = matched_pred / max(len(pred_tokens), 1)
    recall = len(matched_gt) / max(len(gt_tokens), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "token_precision": round(precision, 4),
        "token_recall": round(recall, 4),
        "token_f1": round(f1, 4),
        "pred_token_count": len(pred_tokens),
        "gt_token_count": len(gt_tokens),
        "matched_token_count": len(matched_gt),
    }


def run_rapidocr_on_page(image_path: Path | str) -> Dict[str, Any]:
    """Runs RapidOCR + panel-aware clustering baseline on a page and measures latency."""
    t0 = time.perf_counter()
    engine = RapidOCREngine()
    with Image.open(image_path) as im:
        rgb_im = im.convert("RGB")
        raw_regions = engine.detect_and_recognize(rgb_im)
        clusters = cluster_text_regions(raw_regions)
        ordered = sort_panel_aware_reading_order(clusters)

    latency = time.perf_counter() - t0
    lines = [{"speaker": "character", "text": c.text.strip()} for c in ordered if c.text.strip()]

    return {
        "engine": "RapidOCR + Panel-Aware Clustering",
        "latency_seconds": round(latency, 3),
        "lines": lines,
        "line_count": len(lines),
    }


def run_vlm_on_page(
    image_path: Path | str,
    model: Any,
    processor: Any,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Runs Qwen2-VL inference on a single manga page and measures GPU metrics."""
    t0 = time.perf_counter()
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": MANGA_PROMPT_TEMPLATE},
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Use qwen_vl_utils if available, otherwise direct PIL image passing
    try:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    except (ImportError, Exception):
        inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")

    inputs = inputs.to(device)

    if device == "cuda" and HAS_TORCH and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512)

    latency = time.perf_counter() - t0

    generated_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
    ]
    raw_output = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]

    peak_vram_mb = 0.0
    if device == "cuda" and HAS_TORCH and torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    parsed_lines = parse_vlm_output(raw_output)

    return {
        "engine": "Qwen2-VL-2B-Instruct",
        "latency_seconds": round(latency, 3),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "raw_output": raw_output,
        "lines": parsed_lines,
        "line_count": len(parsed_lines),
    }


def run_pilot_experiment(
    project_root: Path | str = colab_root,
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
    n_pages: int = 3,
    output_json: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Runs a controlled pilot comparing RapidOCR vs Qwen2-VL on permitted development pages."""
    root = Path(project_root).resolve()
    print("=" * 70)
    print("OTAKUSCAN PHASE 3B: CONTROLLED VLM PILOT EXPERIMENT")
    print(f"Candidate Model: {model_id}")
    print("=" * 70)

    device = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "None (CPU)"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if device == "cuda" else 0.0

    print(f"Target Device:  {device} ({gpu_name})")
    print(f"Available VRAM: {vram_gb:.2f} GB")

    pilot_pages = select_pilot_training_pages(root, n_pages=n_pages)
    print(f"\nSelected {len(pilot_pages)} pilot pages strictly from Fold 0 training split:")
    for i, p in enumerate(pilot_pages):
        print(f"  [{i+1}] Seq: {p['sequence_id']} | Page: {p['page_index']} | File: {Path(p['image_path']).name}")

    results_per_page: List[Dict[str, Any]] = []

    # Check if transformers and torch are ready for VLM execution
    vlm_model = None
    vlm_processor = None
    vlm_load_time = 0.0
    vlm_status = "unloaded"

    if device == "cuda":
        try:
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
            print(f"\nLoading {model_id} onto GPU ({device})...")
            t0 = time.perf_counter()
            vlm_processor = AutoProcessor.from_pretrained(model_id)
            vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            vlm_load_time = time.perf_counter() - t0
            vlm_status = "ready"
            print(f"Model loaded successfully in {vlm_load_time:.2f} seconds.")
        except Exception as e:
            vlm_status = f"failed_to_load: {str(e)}"
            print(f"Could not load VLM on this runtime: {e}")
    else:
        vlm_status = "skipped_cpu_runtime (CPU inference >30s/page prohibited)"
        print("\nNote: Running on CPU. VLM GPU inference skipped to prevent severe CPU stalls.")

    for idx, page_info in enumerate(pilot_pages):
        img_path = page_info["image_path"]
        gt_lines = page_info["ground_truth"]
        page_name = Path(img_path).name
        print(f"\n--- Evaluating Pilot Page {idx+1}/{len(pilot_pages)}: {page_name} ---")

        # 1. Run RapidOCR
        ocr_res = run_rapidocr_on_page(img_path)
        ocr_match = compute_transcription_token_match(ocr_res["lines"], gt_lines)
        print(f"RapidOCR: Latency: {ocr_res['latency_seconds']}s | Lines: {ocr_res['line_count']} | Token F1: {ocr_match['token_f1']}")

        # 2. Run VLM if available
        vlm_res: Dict[str, Any] = {
            "status": vlm_status,
            "latency_seconds": None,
            "peak_vram_mb": None,
            "lines": [],
            "token_match": None,
        }
        if vlm_model is not None and vlm_processor is not None:
            try:
                vlm_out = run_vlm_on_page(img_path, vlm_model, vlm_processor, device=device)
                vlm_match = compute_transcription_token_match(vlm_out["lines"], gt_lines)
                vlm_res = {
                    "status": "success",
                    "latency_seconds": vlm_out["latency_seconds"],
                    "peak_vram_mb": vlm_out["peak_vram_mb"],
                    "lines": vlm_out["lines"],
                    "token_match": vlm_match,
                    "raw_output": vlm_out["raw_output"],
                }
                print(f"Qwen2-VL: Latency: {vlm_res['latency_seconds']}s | Peak VRAM: {vlm_res['peak_vram_mb']}MB | Lines: {len(vlm_out['lines'])} | Token F1: {vlm_match['token_f1']}")
            except Exception as e:
                vlm_res["status"] = f"error: {str(e)}"
                print(f"Qwen2-VL error on {page_name}: {e}")

        results_per_page.append({
            "image_path": img_path,
            "page_name": page_name,
            "sequence_id": page_info["sequence_id"],
            "page_index": page_info["page_index"],
            "ground_truth_count": len(gt_lines),
            "rapidocr": {
                "latency_seconds": ocr_res["latency_seconds"],
                "lines": ocr_res["lines"],
                "token_match": ocr_match,
            },
            "vlm": vlm_res,
        })

    summary_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": model_id,
        "device": device,
        "gpu_name": gpu_name,
        "vram_gb": round(vram_gb, 2),
        "vlm_status": vlm_status,
        "vlm_load_time_seconds": round(vlm_load_time, 2),
        "pages_evaluated": len(pilot_pages),
        "per_page_results": results_per_page,
    }

    out_path = Path(output_json) if output_json else root / "logs" / "colab_vlm_pilot_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
    print(f"\nSaved VLM pilot experiment results to {out_path}")

    return summary_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Colab VLM Manga Transcription Pilot")
    parser.add_argument("--pages", type=int, default=3, help="Number of pilot pages to evaluate")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--output", type=str, default="", help="Path to output JSON")
    args = parser.parse_args()

    out_file = Path(args.output) if args.output else None
    run_pilot_experiment(n_pages=args.pages, model_id=args.model, output_json=out_file)
