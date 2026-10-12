from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
from PIL import Image

from src.data.loader import SequenceRecord, OtakuDatasetLoader
from .ocr import RapidOCREngine, DetectedRegion
from .reading_order import sort_reading_order, sort_panel_aware_reading_order
from .clustering import cluster_text_regions, TextCluster

class RapidOCRBaselinePredictor:
    """Minimal end-to-end baseline predictor for multi-page manga transcription.
    
    Design principles:
    - Modular: Keeps OCR, clustering, and reading-order sorting decoupled and replaceable.
    - Deterministic: Sequences processed page-by-page, strictly sorted.
    - Default Speaker: Assigns sequence-local non-NARRATION speaker ('character1') to all dialogue.
    - Clean Handling: Does not invent text for empty pages, outputs [].
    - Preserves punctuation-only speech ("...", "!", "?") if detected.
    - No arbitrary score filtering: min_confidence defaults to 0.0 to retain all detected candidates.
    - Reproducible: enable_clustering toggle allows running unclustered vs clustered variants identically.
    - Panel-Aware: reading_order_mode allows toggling between 'row_band' and 'panel_aware'.
    """

    def __init__(
        self,
        ocr_engine: Optional[RapidOCREngine] = None,
        reading_order_sorter: Optional[Callable[[List[Any]], List[Any]]] = None,
        reading_order_mode: str = "panel_aware",
        default_speaker: str = "character1",
        min_confidence: float = 0.0,
        enable_clustering: bool = False,
        clustering_fn: Optional[Callable[[List[DetectedRegion]], List[TextCluster]]] = None,
    ) -> None:
        self.ocr_engine = ocr_engine if ocr_engine is not None else RapidOCREngine()
        self.reading_order_mode = reading_order_mode
        if reading_order_sorter is not None:
            self.reading_order_sorter = reading_order_sorter
        elif reading_order_mode == "row_band":
            self.reading_order_sorter = sort_reading_order
        else:
            self.reading_order_sorter = sort_panel_aware_reading_order

        self.default_speaker = default_speaker
        self.min_confidence = min_confidence

        self.enable_clustering = enable_clustering
        self.clustering_fn = clustering_fn if clustering_fn is not None else cluster_text_regions


    def predict_images(self, images: List[Image.Image]) -> List[List[Dict[str, str]]]:
        """Predict dialogue transcription across a sequence of images (usually 3 pages)."""
        pages_output: List[List[Dict[str, str]]] = []

        for img in images:
            raw_regions = self.ocr_engine.detect_and_recognize(img)
            
            # Filter only by documented threshold (defaults to 0.0, keeping all detections)
            if self.min_confidence > 0.0:
                valid_regions = [r for r in raw_regions if r.confidence >= self.min_confidence]
            else:
                valid_regions = raw_regions

            if not valid_regions:
                pages_output.append([])
                continue

            # Optional clustering step before reading-order sorting
            if self.enable_clustering:
                entities_to_sort: List[Any] = self.clustering_fn(valid_regions)
            else:
                entities_to_sort = valid_regions

            if not entities_to_sort:
                pages_output.append([])
                continue

            # Sort detected entities using reading-order logic
            ordered_entities = self.reading_order_sorter(entities_to_sort)

            page_items: List[Dict[str, str]] = []
            for item in ordered_entities:
                cleaned_text = item.text.strip()
                # Empty or whitespace-only text must be excluded per scorer validation rules
                if cleaned_text:
                    page_items.append({
                        "speaker": self.default_speaker,
                        "text": cleaned_text,
                    })

            pages_output.append(page_items)

        return pages_output

    def predict_sequence(
        self,
        sequence: SequenceRecord,
        loader: OtakuDatasetLoader
    ) -> List[List[Dict[str, str]]]:
        """Loads sequence images via OtakuDatasetLoader and generates predicted pages."""
        images = loader.load_images(sequence, convert_rgb=True)
        return self.predict_images(images)
