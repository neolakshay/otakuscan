# OtakuScan: Manga Dialogue Transcription & Speaker Attribution Pipeline

OtakuScan is an end-to-end computer vision and natural language processing pipeline designed to extract structured dialogue transcripts from multi-page manga sequences. For each three-page manga sequence, the system detects textual regions, reconstructs reading order across speech bubbles and panel layouts, filters spurious artistic noise, attributes speaker identities (character dialogue vs. narration captions), and outputs submission-compliant JSONL predictions accepted by the official evaluation benchmark.

---

## 1. Project Purpose & Task Requirements

### Task Definition
Given a sequence of **three consecutive manga page images**:
1. **Transcription & Detection:** Detect and recognize all speech dialogue and narration captions present across the pages.
2. **Reading Order Sorting:** Order the text items within each page according to standard Japanese manga reading conventions (right-to-left, top-to-bottom within panel strips).
3. **Speaker Attribution:** Assign a speaker label to each dialogue entity:
   - `NARRATION` for narrative captions and exposition boxes.
   - An arbitrary consistent character identifier (e.g., `character1`, `character2`) for dialogue spoken by story characters.
4. **Structural Compliance:**
   - Output formatted as JSON Lines (`.jsonl`).
   - Exactly one JSON record per sequence containing keys `"sequence_id"` and `"pages"`.
   - The `"pages"` key must contain a list of exactly 3 lists (one per page).
   - Each page entry is a list of `{ "speaker": str, "text": str }` dicts.
   - Text strings must be non-empty and stripped of control characters.

---

## 2. Architecture & Inference Pipeline

The OtakuScan pipeline is structured as a modular 5-stage inference architecture:

```
[Manga Page Image (RGB)]
         │
         ▼
[1. RapidOCR Engine (ONNX / PP-OCRv4)] ──► Extracts text bounding boxes, confidence, text
         │
         ▼
[2. DBSCAN Text Clustering]            ──► Merges broken line fragments into speech bubbles
         │
         ▼
[3. Learned Dialogue Filter]           ──► 20-dim feature classifier filters SFX and margin noise
         │
         ▼
[4. Panel-Aware Reading Sorter]        ──► Sorts bubbles right-to-left, top-to-bottom
         │
         ▼
[5. Learned Narration Classifier]      ──► Attributes 'NARRATION' vs 'character1'
         │
         ▼
[Structured Page Transcript List]
```

### Pipeline Components
1. **RapidOCR Engine (`src/models/ocr.py`):**
   - Utilizes `rapidocr_onnxruntime` with PaddleOCR PP-OCRv4 ONNX models for lightweight CPU-friendly inference.
   - Extracts character polygons, bounding boxes, detection confidences, and normalized text.
2. **Spatial & Lexical Text Clustering (`src/models/clustering.py`):**
   - Manga dialogue frequently spans multiple vertical lines within a single bubble.
   - Implements vertical-aware DBSCAN clustering to group adjacent line fragments into coherent speech bubble units before sorting.
3. **Learned Dialogue vs. Noise Filter (`src/models/filtering.py`):**
   - Extracts a 20-dimensional normalized feature vector per candidate cluster capturing bounding-box geometry (aspect ratio, normalized area, margin distance) and lexical cues (punctuation ratio, digit ratio, uppercase ratio, line count).
   - An L2-regularized logistic regression model (`decision_threshold=0.35`) filters margin page numerals, drawing textures, and sound effects (SFX).
4. **Panel-Aware Reading Order (`src/models/reading_order.py`):**
   - Implements a panel-aware topological sort obeying Japanese manga reading order: right-to-left across columns and top-to-bottom within vertical reading bands.
5. **Learned Narration Caption Classifier (`src/models/filtering.py`):**
   - Evaluates dialogue clusters against learned narration box geometric signatures (aspect ratios, border positions, punctuation traits) with threshold `0.25`, attributing `speaker: "NARRATION"` while assigning character speech to `speaker: "character1"`.

---

## 3. Dataset Setup & Integrity Rules

### Dataset Structure
The dataset directory `dataset/` contains:
```
dataset/
├── sequences.json          # Sequence metadata (95 sequences total: 80 development, 15 test)
├── score.py                # Official evaluation script (Hungarian assignment + Levenshtein)
├── development/
│   ├── images/             # 80 sequences x 3 pages = 240 images
│   └── labels.jsonl        # Ground truth dialogue transcripts and speaker labels
└── test/
    └── images/             # 15 sequences x 3 pages = 45 images (labels withheld)
```

### Strict Read-Only Constraint
- The `dataset/` directory and `dataset/score.py` are strictly read-only and must never be modified.
- All experiment outputs, split definitions, predictions, model weights, and logs are stored strictly outside `dataset/` (`predictions/`, `models/`, `splits/`, `logs/`).

---

## 4. Environment Setup & Dependencies

### Prerequisites
- Python 3.10+ (tested and verified on Python 3.12.3 Linux x86_64)
- CPU compute with at least 4 cores and 4 GB RAM

### Installation
```bash
# Clone the repository
git clone https://github.com/neolakshay/otakuscan.git
cd otakuscan

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install --upgrade pip
pip install rapidocr-onnxruntime onnxruntime pillow numpy scipy
```

---

## 5. Model Weights Requirements & Packaging

### Model Files
The adapted pipeline requires two lightweight learned parameter files (total size ~4.1 KB):
- `models/dialogue_filter_weights.json` (2.0 KB): L2 logistic regression weights for dialogue noise suppression.
- `models/narration_classifier_weights.json` (2.1 KB): L2 logistic regression weights for narration caption detection.

### Git Ignore & Artifact Packaging
Because `models/` is ignored by Git in `.gitignore`:
- **Pre-packaged Archive:** The repository includes a compact artifact archive `models_archive.tar.gz` (1.8 KB) in the repository root.
- **Extracting Weights:**
  ```bash
  tar -xzf models_archive.tar.gz
  ```
  This immediately unpacks the verified weights into `models/`.
- **Fail-Safe Inference:** `scripts/predict_test.py` strictly verifies that both weight files exist before inference starts and refuses to run with a missing-file error rather than silently degrading to an unadapted model. A clean clone without unpacked or retrained weights cannot run adapted inference.

---

## 6. How to Train the Learned Component

To retrain the learned classifiers from scratch using permitted development data:

```bash
python3 scripts/train_dialogue_filter.py
```

### Optional Arguments
```bash
python3 scripts/train_dialogue_filter.py \
  --dataset-root dataset \
  --splits-file splits/dev_splits.json \
  --output-dir models
```

### Training Guarantees
- **Permitted Data:** Trains strictly on the 64 development sequences assigned to Fold 0 training split (`splits/dev_splits.json`).
- **Disjointness:** Zero sequences from the 16 validation sequences (`kfold_5[0]['val']`) or the 15 test sequences are included.
- **Training Time:** ~0.57 seconds with precomputed OCR cache, or ~3-4 minutes when running OCR feature extraction directly on 192 training pages.

---

## 7. How to Run Validation Evaluation

To run full validation evaluation on the official Fold 0 validation split (16 sequences, 48 pages):

```bash
python3 scripts/evaluate_adapted_fold0.py
```

### Optional Arguments
```bash
python3 scripts/evaluate_adapted_fold0.py \
  --dataset-root dataset \
  --splits-file splits/dev_splits.json \
  --models-dir models \
  --predictions-dir predictions \
  --logs-dir logs
```

### Output Files
- Predictions: `predictions/adapted_fold0.jsonl`
- Evaluation report: `logs/adapted_fold0_eval.json`

---

## 8. Verified Development Metrics

All metrics reported below were generated directly by the official scoring script (`dataset/score.py`) across the 16 validation sequences (48 pages) of Fold 0:

| Metric | Preserved Baseline (Phase 1.2) | Panel-Aware Pipeline (Phase 2.4) | Adapted Pipeline (Phase 3 Final) | Delta vs Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **`balanced_joint_f1`** | `0.1531` | `0.1727` | **`0.1739`** | **+0.0208** |
| **`text_order_score`** | `0.6126` | `0.8053` | **`0.8152`** | **+0.2026** |
| **`matched_token_coverage`** | `0.5731` | `0.6736` | **`0.6727`** | **+0.0996** |
| **`speaker_accuracy_on_matched`** | `0.5486` | `0.5410` | **`0.5410`** | -0.0076 |
| **`joint_f1`** | `0.3430` | `0.3922` | **`0.3943`** | **+0.0513** |

### Computational Efficiency
- **Total Validation Runtime:** **53.37 seconds** across 48 pages (**1.112 s/page**).
- **Peak Memory Usage:** **63.61 MB** (well below the 5.7 GiB system limit).
- **Runtime Errors:** **0 errors**.

---

## 9. How to Run Test Inference

To generate the final test predictions on the 15 test sequences:

```bash
python3 scripts/predict_test.py
```

### Optional Arguments
```bash
python3 scripts/predict_test.py \
  --dataset-root dataset \
  --output-path predictions/test_predictions.jsonl \
  --weights-dir models
```

### Inference Verification
- Reads all 15 test sequences (45 pages) from `dataset/sequences.json`.
- Validates sequence IDs, page counts, speaker strings, and character encodings.
- Writes atomically to `predictions/test_predictions.jsonl`.
- Measured runtime: **26.05 seconds** (**0.579 s/page**), **57.73 MB** peak memory.

---

## 10. Test Suite Results

The codebase includes comprehensive unit tests verifying data loading, OCR execution, clustering, reading order, classifier mathematics, official score bridging, Colab utilities, test prediction generation, and model packaging:

```bash
python3 -m unittest discover tests
```

**Results:**
- **75 tests passed, 0 failures, 0 errors** in ~3.8 seconds.

---

## 11. Known Limitations

While OtakuScan meets all structural submission requirements and significantly outperforms baseline reading order and token coverage, several known technical limitations remain:

1. **Speaker Attribution Conflation:**
   - Without character facial recognition or visual speaker pointing models, the pipeline attributes dialogue to a single default identifier (`character1`), with narration boxes attributed as `NARRATION`.
   - On sequences with multiple speaking characters, the Hungarian maximum-weight matching in `dataset/score.py` matches `character1` to the most verbose speaker, leaving other characters unmapped. This bounds `speaker_accuracy_on_matched` at ~54.1% and `balanced_joint_f1` at ~0.174.
2. **OCR Recognition Errors on Stylized Manga Fonts:**
   - RapidOCR's PP-OCRv4 model occasionally struggles with hand-drawn sound effects (SFX), extreme vertical-rotated text, low-contrast screentones, and ornate gothic typography. This caps `matched_token_coverage` at ~67.3%.
3. **Complex Non-Grid Panel Layouts:**
   - While panel-aware sorting handles standard 4-koma and conventional comic grids effectively, avant-garde diagonal panels, full-page bleed artwork, and speech bubbles crossing panel borders can occasionally disrupt ideal reading sequence.

---

## 12. Artifact Paths Summary

| Artifact | Path | Description |
| :--- | :--- | :--- |
| **Final Test Predictions** | `predictions/test_predictions.jsonl` | 15 sequences, 45 pages, validated JSONL |
| **Adapted Validation Predictions** | `predictions/adapted_fold0.jsonl` | 16 validation sequences, 48 pages |
| **Validation Evaluation Log** | `logs/adapted_fold0_eval.json` | Official score metrics and runtime log |
| **Dialogue Filter Weights** | `models/dialogue_filter_weights.json` | Learned L2 logistic regression weights |
| **Narration Classifier Weights** | `models/narration_classifier_weights.json` | Learned narration classifier weights |
| **Model Weights Archive** | `models_archive.tar.gz` | Portable 1.8 KB archive of model weights |
| **Development Splits** | `splits/dev_splits.json` | 5-fold cross-validation split definitions |
| **Phase 3 Technical Report** | `docs/phase3_adaptation_report.md` | In-depth adaptation analysis & metrics |
