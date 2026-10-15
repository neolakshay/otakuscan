# Manga Transcription Pipeline Invariants

1. **Speaker Attribution Modeling**:
   - Verify ground-truth speaker distributions across sequences.
   - For Hungarian bipartite evaluation metrics, models must support dynamic multi-speaker allocation ($K \ge 3$) to avoid mathematical recall caps.
   - Prevent over-segmentation to avoid `extra_speakers` penalties in balanced macro denominators.

2. **OCR Word Segmentation**:
   - Sanitize unigram vocabularies to exclude detached contraction suffixes (`'s'`, `'t'`, `'d'`, `'m'`).
   - Use fallback general-purpose English unigram lexicons to prevent out-of-vocabulary dynamic programming cost explosions.

3. **Inference Pipeline Parity**:
   - Verify that `scripts/predict_test.py` mirrors the exact configuration of the validated evaluation harness before running final submission predictions.
