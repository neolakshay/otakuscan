from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tracemalloc
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.loader import OtakuDatasetLoader, SequenceRecord
from src.eval.scorer import validate_submission_data, SubmissionValidationError
from src.models.baseline_predictor import RapidOCRBaselinePredictor
from src.models.ocr import RapidOCREngine


def discover_test_sequences(
    dataset_root: Path | str,
    loader: Optional[OtakuDatasetLoader] = None,
) -> List[SequenceRecord]:
    """Discovers test sequences from dataset metadata.
    
    Raises:
        FileNotFoundError: If dataset metadata does not exist.
        ValueError: If no test sequences are found in metadata.
    """
    root_path = Path(dataset_root).resolve()
    if loader is None:
        loader = OtakuDatasetLoader(root_path)

    all_sequences = loader.load_sequences()
    test_sequences = [s for s in all_sequences if s.split == "test"]

    if not test_sequences:
        raise ValueError(f"No test sequences found in dataset metadata at {root_path / 'sequences.json'}")

    return test_sequences


def load_adapted_predictor(
    weights_dir: Path | str,
    ocr_engine: Optional[RapidOCREngine] = None,
) -> RapidOCRBaselinePredictor:
    """Loads the adapted RapidOCR predictor with learned dialogue filter and narration classifier.
    
    Reproduces the exact adapted development configuration from scripts/evaluate_adapted_fold0.py.
    
    Raises:
        FileNotFoundError: If required weights are missing.
        RuntimeError: If weights fail to load into classifiers.
    """
    w_dir = Path(weights_dir).resolve()
    filter_weights = w_dir / "dialogue_filter_weights.json"
    narr_weights = w_dir / "narration_classifier_weights.json"

    missing: List[str] = []
    if not filter_weights.is_file():
        missing.append(str(filter_weights))
    if not narr_weights.is_file():
        missing.append(str(narr_weights))

    if missing:
        raise FileNotFoundError(
            f"Missing required model weights file(s): {', '.join(missing)}. "
            f"Both dialogue_filter_weights.json and narration_classifier_weights.json must be present in {w_dir}. "
            f"Refusing to silently fall back to an unadapted model."
        )

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
    )

    if predictor.dialogue_classifier is None:
        raise RuntimeError(f"Dialogue classifier failed to load from {filter_weights}")
    if predictor.narration_classifier is None:
        raise RuntimeError(f"Narration classifier failed to load from {narr_weights}")

    return predictor


def validate_test_predictions(
    predictions: List[Dict[str, Any]],
    expected_test_sequences: List[SequenceRecord],
    dev_sequence_ids: Optional[Set[str]] = None,
) -> None:
    """Performs strict validation on the generated test predictions before output.
    
    Verifies:
    - Every expected test sequence is present exactly once.
    - No development sequence is included.
    - Every sequence has exactly three page entries.
    - Every page is a list.
    - Every prediction contains valid, non-empty speaker and text strings.
    - Narration uses the exact speaker label 'NARRATION'.
    - No control characters in text.
    - Passes official validate_submission_data schema check.
    
    Raises:
        ValueError: If any validation rule fails.
    """
    if not isinstance(predictions, list):
        raise ValueError(f"Predictions must be a list of sequence records, got {type(predictions)}")

    expected_ids = [s.sequence_id for s in expected_test_sequences]
    expected_ids_set = set(expected_ids)

    seen_ids: Set[str] = set()
    for row_idx, row in enumerate(predictions, 1):
        if not isinstance(row, dict):
            raise ValueError(f"Row {row_idx}: expected dictionary, got {type(row)}")
        if set(row.keys()) != {"sequence_id", "pages"}:
            raise ValueError(f"Row {row_idx}: expected exactly keys 'sequence_id' and 'pages', got {set(row.keys())}")

        seq_id = row["sequence_id"]
        if not isinstance(seq_id, str) or not seq_id.strip() or seq_id != seq_id.strip():
            raise ValueError(f"Row {row_idx}: sequence_id must be a nonempty, trimmed string")
        if seq_id in seen_ids:
            raise ValueError(f"Row {row_idx}: Duplicate sequence_id '{seq_id}' in predictions")
        seen_ids.add(seq_id)

        # Check against dev sequence IDs
        if dev_sequence_ids and seq_id in dev_sequence_ids:
            raise ValueError(f"Row {row_idx}: Development sequence '{seq_id}' included in test predictions!")

        # Check if this sequence belongs to expected test sequences
        if seq_id not in expected_ids_set:
            raise ValueError(f"Row {row_idx}: Unexpected sequence_id '{seq_id}' not in expected test sequences")

        pages = row["pages"]
        if not isinstance(pages, list) or len(pages) != 3:
            raise ValueError(
                f"Row {row_idx} ({seq_id}): every sequence must have exactly 3 page entries, got {len(pages) if isinstance(pages, list) else type(pages)}"
            )

        for p_idx, page in enumerate(pages, 1):
            if not isinstance(page, list):
                raise ValueError(f"Row {row_idx} ({seq_id}), page {p_idx}: each page must be a list")
            for item_idx, item in enumerate(page, 1):
                if not isinstance(item, dict) or set(item.keys()) != {"speaker", "text"}:
                    raise ValueError(
                        f"Row {row_idx} ({seq_id}), page {p_idx}, item {item_idx}: each prediction must contain exactly 'speaker' and 'text'"
                    )
                speaker, text = item["speaker"], item["text"]
                if not isinstance(speaker, str) or not speaker.strip() or speaker != speaker.strip():
                    raise ValueError(
                        f"Row {row_idx} ({seq_id}), page {p_idx}, item {item_idx}: speaker must be a nonempty, trimmed string"
                    )
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(
                        f"Row {row_idx} ({seq_id}), page {p_idx}, item {item_idx}: text must be nonempty; use [] for empty pages"
                    )
                # Ensure narration speaker formatting matches exact string
                if speaker.upper() == "NARRATION" and speaker != "NARRATION":
                    raise ValueError(
                        f"Row {row_idx} ({seq_id}), page {p_idx}, item {item_idx}: Narration must use the exact speaker label 'NARRATION', got '{speaker}'"
                    )
                # Check for invalid control characters
                if any(unicodedata.category(c) == "Cc" and not c.isspace() for c in text):
                    raise ValueError(
                        f"Row {row_idx} ({seq_id}), page {p_idx}, item {item_idx}: text contains illegal control character"
                    )

    # Check for missing test sequences
    missing_ids = expected_ids_set - seen_ids
    if missing_ids:
        raise ValueError(f"Missing test sequence(s) from predictions: {sorted(missing_ids)}")

    if len(seen_ids) != len(expected_ids_set):
        raise ValueError(f"Expected {len(expected_ids_set)} test sequences, got {len(seen_ids)}")

    # Delegate also to official scorer validation function
    try:
        validate_submission_data(predictions, expected_pages=3)
    except SubmissionValidationError as e:
        raise ValueError(f"Official submission data validation failed: {str(e)}") from e


def atomic_write_jsonl(
    predictions: List[Dict[str, Any]],
    output_path: Path | str,
    dataset_root: Optional[Path | str] = None,
) -> None:
    """Atomically writes predictions to a JSONL file outside the dataset directory.
    
    Raises:
        ValueError: If output path is inside dataset root.
        Exception: If writing or serialization fails, preserving any existing file.
    """
    out = Path(output_path).resolve()

    if dataset_root is not None:
        ds_root = Path(dataset_root).resolve()
        try:
            out.relative_to(ds_root)
            # If relative_to succeeds without ValueError, out is inside dataset_root
            raise ValueError(f"Output path {out} must be outside dataset directory {ds_root}")
        except ValueError as err:
            # If err is our custom error, re-raise it
            if "must be outside dataset directory" in str(err):
                raise
            # Otherwise relative_to raised ValueError, meaning out is NOT inside ds_root (which is correct)
            pass

    out.parent.mkdir(parents=True, exist_ok=True)
    temp_file = out.parent / f".{out.name}.tmp.{os.getpid()}"

    try:
        lines: List[str] = []
        for row in predictions:
            lines.append(json.dumps(row, ensure_ascii=False) + "\n")

        with open(temp_file, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line)
            f.flush()
            os.fsync(f.fileno())

        # Verify parsed back
        with open(temp_file, "r", encoding="utf-8") as f:
            parsed_rows = [json.loads(line) for line in f if line.strip()]
        if len(parsed_rows) != len(predictions):
            raise RuntimeError(f"Parsed row count {len(parsed_rows)} does not match expected {len(predictions)}")

        # Atomic rename
        temp_file.replace(out)
    except Exception:
        if temp_file.is_file():
            temp_file.unlink(missing_ok=True)
        raise


def generate_test_predictions(
    dataset_root: Path | str,
    output_path: Path | str,
    weights_dir: Optional[Path | str] = None,
    predictor: Optional[Any] = None,
) -> Dict[str, Any]:
    """Generates and validates predictions for all test sequences in the dataset.
    
    Returns a summary dictionary of the inference run.
    """
    root_path = Path(dataset_root).resolve()
    out_path = Path(output_path).resolve()

    # Refuse to write inside dataset root
    try:
        out_path.relative_to(root_path)
        raise ValueError(f"Output path {out_path} must be outside dataset directory {root_path}")
    except ValueError as err:
        if "must be outside dataset directory" in str(err):
            raise

    loader = OtakuDatasetLoader(root_path)
    all_sequences = loader.load_sequences()

    test_sequences = [s for s in all_sequences if s.split == "test"]
    dev_sequence_ids = {s.sequence_id for s in all_sequences if s.split == "development"}

    if not test_sequences:
        raise ValueError(f"No test sequences found in dataset metadata at {root_path / 'sequences.json'}")

    # Initialize predictor if not provided
    if predictor is None:
        if weights_dir is None:
            weights_dir = project_root / "models"
        predictor = load_adapted_predictor(weights_dir=weights_dir)

    tracemalloc.start()
    start_time = time.perf_counter()

    prediction_rows: List[Dict[str, Any]] = []
    total_pages = 0
    narration_predicted_count = 0
    empty_pages_count = 0
    retained_regions_count = 0

    print(f"Beginning test inference on {len(test_sequences)} sequences ({len(test_sequences) * 3} pages)...")

    for idx, seq in enumerate(test_sequences, 1):
        if len(seq.image_paths) != 3:
            raise ValueError(f"Sequence {seq.sequence_id} does not have exactly 3 images: {len(seq.image_paths)}")

        # Verify all image files exist before loading
        for p in seq.image_paths:
            if not p.is_file():
                raise FileNotFoundError(f"Image file not found for sequence {seq.sequence_id}: {p}")

        images = loader.load_images(seq, convert_rgb=True)
        try:
            total_pages += len(images)
            pred_pages = predictor.predict_images(images)

            if len(pred_pages) != 3:
                raise ValueError(
                    f"Predictor returned {len(pred_pages)} pages for sequence {seq.sequence_id}, expected 3"
                )

            for p in pred_pages:
                retained_regions_count += len(p)
                for item in p:
                    if item.get("speaker") == "NARRATION":
                        narration_predicted_count += 1
                if not p:
                    empty_pages_count += 1

            prediction_rows.append({
                "sequence_id": seq.sequence_id,
                "pages": pred_pages,
            })
        finally:
            for img in images:
                try:
                    img.close()
                except Exception:
                    pass

        if idx % 5 == 0 or idx == len(test_sequences):
            print(f"  Processed {idx}/{len(test_sequences)} test sequences ({total_pages} pages)")

    elapsed_time = time.perf_counter() - start_time
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("Running strict pre-output validation checks...")
    validate_test_predictions(
        predictions=prediction_rows,
        expected_test_sequences=test_sequences,
        dev_sequence_ids=dev_sequence_ids,
    )

    print(f"Atomically writing validated test predictions to {out_path}...")
    atomic_write_jsonl(
        predictions=prediction_rows,
        output_path=out_path,
        dataset_root=root_path,
    )

    summary = {
        "sequence_count": len(test_sequences),
        "total_pages": total_pages,
        "elapsed_seconds": round(elapsed_time, 3),
        "seconds_per_page": round(elapsed_time / total_pages, 3) if total_pages else 0.0,
        "peak_memory_mb": round(peak_mem / (1024 * 1024), 2),
        "retained_regions_count": retained_regions_count,
        "narration_predicted_count": narration_predicted_count,
        "empty_pages_predicted": empty_pages_count,
        "output_path": str(out_path),
    }

    print("Test inference and validation completed successfully.")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate final OtakuScan test predictions using the adapted RapidOCR pipeline."
    )
    parser.add_argument(
        "--dataset-root",
        "--dataset_root",
        type=Path,
        default=project_root / "dataset",
        help="Path to dataset root directory (default: <project_root>/dataset)",
    )
    parser.add_argument(
        "--output-path",
        "--output_path",
        "-o",
        type=Path,
        default=project_root / "predictions" / "test_predictions.jsonl",
        help="Path to output JSONL file (default: <project_root>/predictions/test_predictions.jsonl)",
    )
    parser.add_argument(
        "--weights-dir",
        "--weights_dir",
        "--models-dir",
        type=Path,
        default=project_root / "models",
        help="Path to directory containing model weights (default: <project_root>/models)",
    )

    args = parser.parse_args()

    try:
        summary = generate_test_predictions(
            dataset_root=args.dataset_root,
            output_path=args.output_path,
            weights_dir=args.weights_dir,
        )
        print("\n=== SUMMARY ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"\nERROR: Inference failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
