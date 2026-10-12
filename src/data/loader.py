from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image

@dataclass(frozen=True)
class DialogueLine:
    speaker: str
    text: str

@dataclass(frozen=True)
class SequenceRecord:
    sequence_id: str
    split: str
    image_paths: List[Path]

class OtakuDatasetLoader:
    """Read-only dataset loader for the OtakuScan benchmark."""

    def __init__(self, dataset_root: Path | str) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(f"Dataset root directory does not exist: {self.dataset_root}")

    def load_sequences(self) -> List[SequenceRecord]:
        """Load and validate all sequence records from dataset/sequences.json."""
        seq_file = self.dataset_root / "sequences.json"
        if not seq_file.is_file():
            raise FileNotFoundError(f"Sequences file not found: {seq_file}")

        with open(seq_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        records: List[SequenceRecord] = []
        for item in data:
            seq_id = str(item["sequence_id"])
            split = str(item["split"])
            image_paths = [self.dataset_root / Path(img_rel) for img_rel in item["images"]]
            records.append(SequenceRecord(
                sequence_id=seq_id,
                split=split,
                image_paths=image_paths,
            ))
        return records

    def load_labels(self, split: str = "development") -> Dict[str, List[List[DialogueLine]]]:
        """Load ground truth dialogue labels from dataset/<split>/labels.jsonl."""
        labels_file = self.dataset_root / split / "labels.jsonl"
        if not labels_file.is_file():
            raise FileNotFoundError(f"Labels file not found: {labels_file}")

        labels: Dict[str, List[List[DialogueLine]]] = {}
        with open(labels_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                seq_id = row["sequence_id"]
                pages: List[List[DialogueLine]] = []
                for page in row["pages"]:
                    page_lines = [
                        DialogueLine(speaker=item["speaker"], text=item["text"])
                        for item in page
                    ]
                    pages.append(page_lines)
                labels[seq_id] = pages
        return labels

    def load_images(self, sequence: SequenceRecord, convert_rgb: bool = True) -> List[Image.Image]:
        """Load PIL images for the 3 pages of a sequence, normalizing to RGB if requested."""
        images: List[Image.Image] = []
        for path in sequence.image_paths:
            if not path.is_file():
                raise FileNotFoundError(f"Image file not found: {path}")
            img = Image.open(path)
            if convert_rgb and img.mode != "RGB":
                # Create detached copy in RGB mode
                rgb_img = img.convert("RGB")
                img.close()
                images.append(rgb_img)
            else:
                img.load()
                images.append(img)
        return images
