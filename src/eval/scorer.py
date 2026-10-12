from __future__ import annotations

import importlib.util
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

class SubmissionValidationError(ValueError):
    """Raised when a predicted submission violates the official format constraints."""
    pass

def validate_submission_data(data: List[Dict[str, Any]], expected_pages: int = 3) -> None:
    """Strict syntactic validation mirroring the rules in dataset/score.py."""
    seen_seq_ids = set()
    for row_idx, row in enumerate(data, 1):
        if not isinstance(row, dict):
            raise SubmissionValidationError(f"Row {row_idx}: expected dictionary, got {type(row)}")
        if set(row.keys()) != {"sequence_id", "pages"}:
            raise SubmissionValidationError(
                f"Row {row_idx}: expected exactly keys 'sequence_id' and 'pages', got {set(row.keys())}"
            )
        seq_id = row["sequence_id"]
        if not isinstance(seq_id, str) or not seq_id.strip() or seq_id != seq_id.strip():
            raise SubmissionValidationError(f"Row {row_idx}: sequence_id must be a nonempty, trimmed string")
        if seq_id in seen_seq_ids:
            raise SubmissionValidationError(f"Row {row_idx}: duplicate sequence_id '{seq_id}'")
        seen_seq_ids.add(seq_id)

        pages = row["pages"]
        if not isinstance(pages, list) or len(pages) != expected_pages:
            raise SubmissionValidationError(
                f"Row {row_idx}: pages must be a list of exactly {expected_pages} lists, got {len(pages) if isinstance(pages, list) else type(pages)}"
            )

        for p_idx, page in enumerate(pages, 1):
            if not isinstance(page, list):
                raise SubmissionValidationError(f"Row {row_idx}, page {p_idx}: each page must be a list")
            for item_idx, item in enumerate(page, 1):
                if not isinstance(item, dict) or set(item.keys()) != {"speaker", "text"}:
                    raise SubmissionValidationError(
                        f"Row {row_idx}, page {p_idx}, item {item_idx}: each line must contain exactly 'speaker' and 'text'"
                    )
                speaker, text = item["speaker"], item["text"]
                if not isinstance(speaker, str) or not speaker.strip() or speaker != speaker.strip():
                    raise SubmissionValidationError(
                        f"Row {row_idx}, page {p_idx}, item {item_idx}: speaker must be a nonempty, trimmed string"
                    )
                if not isinstance(text, str) or not text.strip():
                    raise SubmissionValidationError(
                        f"Row {row_idx}, page {p_idx}, item {item_idx}: text must be nonempty; use [] for empty pages"
                    )
                if any(unicodedata.category(c) == "Cc" and not c.isspace() for c in text):
                    raise SubmissionValidationError(
                        f"Row {row_idx}, page {p_idx}, item {item_idx}: text contains control characters"
                    )

class OtakuScorerBridge:
    """Programmatic bridge to dataset/score.py preserving its strict read-only nature."""

    def __init__(self, dataset_root: Path | str) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        score_py = self.dataset_root / "score.py"
        if not score_py.is_file():
            raise FileNotFoundError(f"Score script not found at {score_py}")

        # Dynamically import dataset/score.py as a module without modifying it
        spec = importlib.util.spec_from_file_location("official_score", score_py)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module spec for {score_py}")
        self._score_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._score_module)

    def load_reference_jsonl(self, path: Path | str, pages: int = 3) -> Dict[str, Any]:
        """Loads reference file using official score.py load_jsonl(reference=True)."""
        return self._score_module.load_jsonl(Path(path), reference=True, pages=pages)

    def load_prediction_jsonl(self, path: Path | str, pages: int = 3) -> Dict[str, Any]:
        """Loads prediction file using official score.py load_jsonl(reference=False)."""
        return self._score_module.load_jsonl(Path(path), reference=False, pages=pages)

    def evaluate_dict(self, references: Dict[str, Any], predictions: Dict[str, Any], pages: int = 3) -> Dict[str, Any]:
        """Directly evaluates reference dict vs prediction dict using official score.py evaluate()."""
        return self._score_module.evaluate(references, predictions, pages=pages)

    def evaluate_jsonl(self, reference_path: Path | str, prediction_path: Path | str, pages: int = 3) -> Dict[str, Any]:
        """Loads and evaluates JSONL paths using official score.py."""
        refs = self.load_reference_jsonl(reference_path, pages=pages)
        preds = self.load_prediction_jsonl(prediction_path, pages=pages)
        return self.evaluate_dict(refs, preds, pages=pages)
