from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
from pathlib import Path
from PIL import Image

from src.data.loader import SequenceRecord, OtakuDatasetLoader
from .ocr import RapidOCREngine, DetectedRegion
from .reading_order import sort_reading_order, sort_panel_aware_reading_order
from .clustering import cluster_text_regions, TextCluster
from .filtering import (
    extract_cluster_features,
    LearnedClusterClassifier,
    LearnedNarrationClassifier,
)

class RapidOCRBaselinePredictor:
    """Minimal end-to-end baseline predictor for multi-page manga transcription.
    
    Design principles:
    - Modular: Keeps OCR, clustering, filtering, and reading-order sorting decoupled.
    - Deterministic: Sequences processed page-by-page, strictly sorted.
    - Default Speaker: Assigns sequence-local non-NARRATION speaker ('character1') to dialogue.
    - Narration Aware: Optional learned narration classifier identifies caption boxes.
    - Noise Filter: Optional learned classifier filters false-positive detections.
    - Clean Handling: Does not invent text for empty pages, outputs [].
    - Preserves punctuation-only speech ("...", "!", "?") if detected.
    - No arbitrary score filtering: min_confidence defaults to 0.0 to retain all detected candidates.
    - Reproducible: Configuration toggles allow isolating each component's contribution.
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
        enable_dialogue_filter: bool = False,
        dialogue_classifier: Optional[LearnedClusterClassifier] = None,
        dialogue_classifier_path: Optional[str | Path] = None,
        filter_threshold: Optional[float] = None,
        enable_narration_classifier: bool = False,
        narration_classifier: Optional[LearnedNarrationClassifier] = None,
        narration_classifier_path: Optional[str | Path] = None,
        narration_threshold: Optional[float] = None,
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

        self.enable_dialogue_filter = enable_dialogue_filter
        if dialogue_classifier is not None:
            self.dialogue_classifier = dialogue_classifier
        elif dialogue_classifier_path is not None and Path(dialogue_classifier_path).is_file():
            self.dialogue_classifier = LearnedClusterClassifier.load_from_json(dialogue_classifier_path)
        else:
            self.dialogue_classifier = None

        if self.dialogue_classifier is not None and filter_threshold is not None:
            self.dialogue_classifier.decision_threshold = filter_threshold

        self.enable_narration_classifier = enable_narration_classifier
        if narration_classifier is not None:
            self.narration_classifier = narration_classifier
        elif narration_classifier_path is not None and Path(narration_classifier_path).is_file():
            self.narration_classifier = LearnedNarrationClassifier.load_from_json(narration_classifier_path)
        else:
            self.narration_classifier = None

        if self.narration_classifier is not None and narration_threshold is not None:
            self.narration_classifier.decision_threshold = narration_threshold


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

            pw, ph = img.size

            # Optional learned dialogue filtering
            if self.enable_dialogue_filter and self.dialogue_classifier is not None:
                filtered_entities: List[Any] = []
                for entity in entities_to_sort:
                    feats = extract_cluster_features(entity, page_width=pw, page_height=ph)
                    prob = self.dialogue_classifier.predict_single(feats)
                    if prob >= self.dialogue_classifier.decision_threshold:
                        filtered_entities.append(entity)
                entities_to_sort = filtered_entities

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
                    speaker = self.default_speaker
                    if self.enable_narration_classifier and self.narration_classifier is not None:
                        feats = extract_cluster_features(item, page_width=pw, page_height=ph)
                        prob_narr = self.narration_classifier.predict_single(feats)
                        if prob_narr >= self.narration_classifier.decision_threshold:
                            speaker = "NARRATION"

                    page_items.append({
                        "speaker": speaker,
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
