from __future__ import annotations

import os
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Optional, Tuple


REQUIRED_DATASET_FILES = [
    "sequences.json",
    os.path.join("development", "labels.jsonl"),
]


def resolve_dataset_root(candidate_path: Path | str) -> Optional[Path]:
    """Finds the directory that directly contains sequences.json and development/labels.jsonl.
    
    Handles both direct flat dataset directories and single-nested 'dataset/' subfolders.
    """
    path = Path(candidate_path).resolve()
    if not path.is_dir():
        return None

    # Check direct path
    if (path / "sequences.json").is_file() and (path / "development" / "labels.jsonl").is_file():
        return path

    # Check immediate nested subfolder (e.g. candidate_path / "dataset")
    nested = path / "dataset"
    if nested.is_dir():
        if (nested / "sequences.json").is_file() and (nested / "development" / "labels.jsonl").is_file():
            return nested

    return None


def is_valid_dataset_directory(candidate_path: Path | str) -> bool:
    """Returns True if the directory or an immediate child contains a valid OtakuScan dataset."""
    return resolve_dataset_root(candidate_path) is not None


def validate_dataset_directory(candidate_path: Path | str) -> Path:
    """Validates that a dataset directory contains all required OtakuScan benchmark files.
    
    Returns:
        Path to the resolved root containing sequences.json.
    Raises:
        FileNotFoundError: If the directory or any required file is missing.
    """
    path = Path(candidate_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset root directory does not exist: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset path is not a directory: {path}")

    resolved = resolve_dataset_root(path)
    if resolved is not None:
        return resolved

    # Detail which file was missing for clear actionable diagnostics
    missing: list[str] = []
    for req in REQUIRED_DATASET_FILES:
        if not (path / req).is_file() and not (path / "dataset" / req).is_file():
            missing.append(req)

    raise FileNotFoundError(
        f"Invalid dataset directory at {path}. Missing required benchmark files: {', '.join(missing)}"
    )


def extract_dataset_archive(
    archive_path: Path | str,
    extract_to: Path | str,
    force: bool = False,
) -> Path:
    """Extracts otakuscan-dataset.tar.gz into extract_to idempotently.
    
    If extract_to already contains a valid dataset and force is False,
    re-extraction is skipped to preserve fast startup and avoid re-downloading.
    """
    target_dir = Path(extract_to).resolve()
    arc = Path(archive_path).resolve()

    # Idempotent reuse: check if target is already extracted and valid
    if not force and is_valid_dataset_directory(target_dir):
        resolved = validate_dataset_directory(target_dir)
        print(f"Dataset already extracted and valid at: {resolved} (reusing without re-extracting)")
        return resolved

    if not arc.is_file():
        raise FileNotFoundError(f"Dataset archive not found at: {arc}")

    print(f"Extracting dataset archive from {arc} to {target_dir}...")
    target_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(arc, "r:*") as tar:
        # Use safe extraction filter if supported in Python 3.12+
        if hasattr(tarfile, "data_filter"):
            tar.extractall(path=target_dir, filter="data")
        else:
            tar.extractall(path=target_dir)

    resolved = validate_dataset_directory(target_dir)
    print(f"Dataset extraction verified successfully at: {resolved}")
    return resolved
