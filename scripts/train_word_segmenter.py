from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.segmentation import WordSegmenter


def train_word_segmenter(
    dataset_root: Path | str = project_root / "dataset",
    splits_file: Path | str = project_root / "splits" / "dev_splits.json",
    fold_index: int = 0,
    output_path: Path | str = project_root / "models" / "word_segmenter_vocab.json",
    split_penalty: float = 2.5,
) -> WordSegmenter:
    """Trains the WordSegmenter strictly on development training sequences.
    
    Zero-Leakage Enforcement:
    - Splits file specifies train/val sets.
    - Test sequence IDs are loaded from dataset/sequences.json.
    - An explicit disjointness assertion prevents any validation or test data
      from entering vocabulary building or model parameters.
    """
    dataset_root = Path(dataset_root).resolve()
    splits_file = Path(splits_file).resolve()
    output_path = Path(output_path).resolve()

    with open(splits_file, "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    fold_info = splits_data["kfold_5"][fold_index]
    train_ids: Set[str] = set(fold_info["train"])
    val_ids: Set[str] = set(fold_info["val"])

    # Load sequence metadata to identify test sequences
    sequences_file = dataset_root / "sequences.json"
    test_ids: Set[str] = set()
    if sequences_file.is_file():
        with open(sequences_file, "r", encoding="utf-8") as f:
            seq_list = json.load(f)
            for s in seq_list:
                if s.get("split") == "test":
                    test_ids.add(s["sequence_id"])

    # Strict leakage validation
    assert train_ids.isdisjoint(val_ids), "Data leakage: train_ids overlaps with val_ids!"
    assert train_ids.isdisjoint(test_ids), "Data leakage: train_ids overlaps with test_ids!"
    print(f"Permitted training sequences: {len(train_ids)}")
    print(f"Forbidden validation sequences: {len(val_ids)}")
    print(f"Forbidden test sequences: {len(test_ids)}")

    # Load development ground-truth labels
    labels_file = dataset_root / "development" / "labels.jsonl"
    golds_dict: Dict[str, List[List[Dict[str, str]]]] = {}
    with open(labels_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            golds_dict[rec["sequence_id"]] = rec["pages"]

    forbidden_ids = val_ids.union(test_ids)
    segmenter = WordSegmenter(split_penalty=split_penalty)
    segmenter.fit_from_training_sequences(
        golds_dict=golds_dict,
        allowed_train_ids=train_ids,
        forbidden_val_test_ids=forbidden_ids,
        include_common_words=True,
        common_word_prior_count=100,
    )

    print(f"Vocabulary size: {len(segmenter.vocabulary)}")
    print(f"Total word tokens: {segmenter.total_words}")

    # Save artifact
    output_path.parent.mkdir(parents=True, exist_ok=True)
    segmenter.save_to_json(output_path)
    print(f"Saved WordSegmenter vocabulary model to: {output_path}")

    # Self-test reloading
    reloaded = WordSegmenter.load_from_json(output_path)
    test_sample = "socool thenew backinthe together understand HUH?DOES TORNAPART THEREMAINSOF"
    segmented = reloaded.segment_text(test_sample)
    print(f"Self-test sample: '{test_sample}' -> '{segmented}'")
    assert "so cool" in segmented, f"Failed split on 'socool': {segmented}"
    assert "the new" in segmented, f"Failed split on 'thenew': {segmented}"
    assert "together" in segmented, f"Incorrectly split 'together': {segmented}"
    assert "understand" in segmented, f"Incorrectly split 'understand': {segmented}"
    assert "HUH? DOES" in segmented, f"Failed punctuation spacing on 'HUH?DOES': {segmented}"
    assert "TORN APART" in segmented, f"Failed split on 'TORNAPART': {segmented}"
    assert "THE REMAINS OF" in segmented, f"Failed split on 'THEREMAINSOF': {segmented}"

    return reloaded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train WordSegmenter strictly on permitted training data."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=project_root / "dataset",
        help="Dataset root directory",
    )
    parser.add_argument(
        "--splits-file",
        type=Path,
        default=project_root / "splits" / "dev_splits.json",
        help="Path to dev splits JSON",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold index to train (default: 0)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=project_root / "models" / "word_segmenter_vocab.json",
        help="Output path for trained vocabulary JSON",
    )
    parser.add_argument(
        "--split-penalty",
        type=float,
        default=2.5,
        help="Viterbi split penalty (default: 2.5)",
    )

    args = parser.parse_args()

    train_word_segmenter(
        dataset_root=args.dataset_root,
        splits_file=args.splits_file,
        fold_index=args.fold,
        output_path=args.output_path,
        split_penalty=args.split_penalty,
    )


if __name__ == "__main__":
    main()
