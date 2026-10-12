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
        raise FileNotFoundError(
            f"Dataset archive not found at: {arc}\n"
            "Action required: Upload your local 'dataset/' archive to Google Drive at:\n"
            "  MyDrive/otakuscan-dataset/otakuscan-dataset.tar.gz"
        )

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


def setup_colab_workspace(
    repo_url: str = "https://github.com/neolakshay/otakuscan.git",
    expected_drive_archive_rel: str = "otakuscan-dataset/otakuscan-dataset.tar.gz",
    extract_target: Path | str = "/content/otakuscan-data/dataset",
) -> Tuple[Path, Path]:
    """Automates one-click workspace setup in Google Colab.
    
    Returns:
        (project_root, dataset_root)
    """
    in_colab = "google.colab" in sys.modules

    if in_colab:
        print("Running in Google Colab environment.")
        # 1. Mount Google Drive if not already mounted
        drive_mount_root = Path("/content/drive")
        my_drive = drive_mount_root / "MyDrive"

        if not my_drive.is_dir():
            print("Mounting Google Drive (please approve authorization modal if prompted)...")
            from google.colab import drive  # type: ignore
            drive.mount(str(drive_mount_root))

        # 2. Locate or clone project repository
        content_repo = Path("/content/otakuscan")
        if (content_repo / "splits" / "dev_splits.json").is_file():
            project_root = content_repo
            print(f"Project repository already present at {project_root}")
        else:
            print(f"Cloning project repository from {repo_url} into {content_repo}...")
            os.system(f"git clone {repo_url} {content_repo}")
            project_root = content_repo

        # 3. Locate dataset archive in Google Drive
        primary_archive = my_drive / expected_drive_archive_rel
        fallback_archive = my_drive / "otakuscan-dataset.tar.gz"

        if primary_archive.is_file():
            archive_path = primary_archive
        elif fallback_archive.is_file():
            archive_path = fallback_archive
        else:
            raise FileNotFoundError(
                f"Dataset archive not found!\n"
                f"Checked:\n"
                f"  1. {primary_archive}\n"
                f"  2. {fallback_archive}\n"
                f"Please ensure you uploaded 'otakuscan-dataset.tar.gz' to your Google Drive in:\n"
                f"  MyDrive/otakuscan-dataset/otakuscan-dataset.tar.gz"
            )

        # 4. Idempotently extract dataset archive
        dataset_root = extract_dataset_archive(archive_path, extract_target)

    else:
        # Local development execution
        project_root = Path.cwd().resolve()
        # Search upward if cwd is a subfolder
        while not (project_root / "splits" / "dev_splits.json").is_file() and project_root.parent != project_root:
            project_root = project_root.parent

        dataset_candidate = project_root / "dataset"
        if is_valid_dataset_directory(dataset_candidate):
            dataset_root = validate_dataset_directory(dataset_candidate)
        else:
            raise FileNotFoundError(f"Local dataset not found at {dataset_candidate}")

    # Ensure project root is in sys.path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    os.chdir(project_root)
    return project_root, dataset_root
