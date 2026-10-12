from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from src.data.loader import OtakuDatasetLoader, SequenceRecord
from src.eval.scorer import OtakuScorerBridge, validate_submission_data, SubmissionValidationError

def create_sequence_train_val_split(
    sequences: List[SequenceRecord],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Deterministically partition sequence records into train/val sets by sequence ID.
    
    Guarantees:
    - Every sequence (including all 3 pages) stays strictly within one split.
    - Zero sequence leakage between train and val.
    - Completely reproducible across runs given the same seed.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0.0 and 1.0, got {val_ratio}")

    seq_ids = sorted([s.sequence_id for s in sequences])
    rng = random.Random(seed)
    rng.shuffle(seq_ids)

    n_val = int(round(len(seq_ids) * val_ratio))
    val_ids = sorted(seq_ids[:n_val])
    train_ids = sorted(seq_ids[n_val:])

    return {
        "train": train_ids,
        "val": val_ids,
    }

def create_sequence_kfold_splits(
    sequences: List[SequenceRecord],
    n_splits: int = 5,
    seed: int = 42,
) -> List[Dict[str, List[str]]]:
    """Deterministically partition sequence records into K cross-validation folds.
    
    Guarantees:
    - Exactly n_splits folds with disjoint validation sets.
    - Sequences are never split across pages.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}")

    seq_ids = sorted([s.sequence_id for s in sequences])
    rng = random.Random(seed)
    rng.shuffle(seq_ids)

    folds: List[Dict[str, List[str]]] = []
    fold_size = len(seq_ids) // n_splits
    remainder = len(seq_ids) % n_splits

    start = 0
    for i in range(n_splits):
        # Distribute remainder evenly across earlier folds
        current_fold_size = fold_size + (1 if i < remainder else 0)
        end = start + current_fold_size
        val_subset = sorted(seq_ids[start:end])
        train_subset = sorted(seq_ids[:start] + seq_ids[end:])
        folds.append({
            "fold_index": i,
            "train": train_subset,
            "val": val_subset,
        })
        start = end

    return folds

class ValidationPipeline:
    """Reproducible evaluation pipeline executing official metric assessment on local validation splits."""

    def __init__(self, dataset_root: Path | str) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        self.loader = OtakuDatasetLoader(self.dataset_root)
        self.bridge = OtakuScorerBridge(self.dataset_root)

    def export_split_definition(self, split_data: Any, target_path: Path | str) -> None:
        """Saves a split configuration JSON outside the dataset/ directory."""
        path = Path(target_path).resolve()
        if self.dataset_root in path.parents or path == self.dataset_root:
            raise PermissionError(f"Prohibited write attempt to dataset directory: {path}")

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, indent=2)

    def evaluate_split(
        self,
        val_sequence_ids: List[str],
        predictor: Callable[[SequenceRecord], List[List[Dict[str, str]]]],
        pages: int = 3,
    ) -> Dict[str, Any]:
        """Runs a prediction function on validation sequences and evaluates using the official scorer."""
        all_seqs = {s.sequence_id: s for s in self.loader.load_sequences()}
        full_labels = self.loader.load_labels(split="development")

        val_id_set = set(val_sequence_ids)
        # Construct reference format expected by official score.py
        val_references: Dict[str, Any] = {}
        for seq_id in val_sequence_ids:
            if seq_id not in full_labels:
                raise KeyError(f"Sequence ID '{seq_id}' not found in development labels")
            # score.py format: list of 3 pages, each page list of dicts with speaker and text
            gold_pages = [
                [{"speaker": line.speaker, "text": line.text} for line in page]
                for page in full_labels[seq_id]
            ]
            val_references[seq_id] = gold_pages

        # Generate predictions
        submission_rows: List[Dict[str, Any]] = []
        val_predictions: Dict[str, Any] = {}
        for seq_id in val_sequence_ids:
            seq_record = all_seqs[seq_id]
            pred_pages = predictor(seq_record)
            submission_rows.append({
                "sequence_id": seq_id,
                "pages": pred_pages,
            })
            val_predictions[seq_id] = pred_pages

        # Validate prediction structure
        validate_submission_data(submission_rows, expected_pages=pages)

        # Execute evaluation using official score.py logic
        report = self.bridge.evaluate_dict(
            references=val_references,
            predictions=val_predictions,
            pages=pages,
        )
        return report
