# Otakuscan Dataset Reconnaissance Report

**Date:** October 12, 2026  
**Repository:** `neolakshay/otakuscan`  
**Dataset Root:** `dataset/`  
**Evaluation Engine:** `dataset/score.py` (v2.1)

---

## 1. Executive Summary

This reconnaissance report documents the empirical structure, characteristics, image properties, ground truth annotations, and evaluation constraints of the **Otakuscan** benchmark dataset.

The core task of Otakuscan is **multi-page manga transcript extraction and sequence-wide speaker attribution (diarization)**. Given sequences of 3 consecutive manga page scans, the system must:
1. Detect and transcribe English dialogue and narration bubbles in the correct manga reading order (right-to-left, top-to-bottom).
2. Assign each transcription to a speaker label (`character1`, `character2`, ..., `NARRATION`), maintaining consistent speaker assignments across the 3-page sequence.

### Key Empirical Findings
- **Zero Discrepancies:** The dataset filesystem is completely clean and matches `sequences.json` with 100% integrity. There are 0 missing referenced images and 0 orphan images.
- **Scale:** 95 total sequences (80 development, 15 test), totaling 285 PNG images (~150 MB disk usage).
- **Split Breakdown:**
  - `development`: 80 sequences (240 images), fully annotated in `development/labels.jsonl` (1,753 dialogue bubbles).
  - `test`: 15 sequences (45 images), unannotated, mirroring `sample_submission.jsonl`.
- **Image Mode Hazard (Critical):** 
  - `development` images use mixed modes: 38.8% `RGB` (93), 32.1% `L` (grayscale, 77), and 29.2% `P` (palette-based, 70).
  - `test` images contain **0 RGB images**: 75.6% `P` (34) and 24.4% `L` (11).
  - **Pipeline Impact:** Models or image loaders expecting 3-channel RGB tensors will fail on 100% of the test set and 61% of the dev set unless explicit `.convert("RGB")` preprocessing is applied.
- **Intra-Sequence Resolution Variance:** In 52 of the 95 sequences (54.7%), the image dimensions or color modes differ between the 3 pages of the same sequence (e.g. width variations of 2–15 pixels across scans).
- **Domain Identification:** The development set comprises exactly 4 major manga series:
  - *Death Note* (42 sequences, 52.5%)
  - *Chainsaw Man* (26 sequences, 32.5%)
  - *Silver Spoon / Gin no Saji* (6 sequences, 7.5%)
  - *Kaguya-sama: Love Is War* (6 sequences, 7.5%)
- **Edge Cases Identified:**
  - 10 pages in the dev set (4.2%) have zero dialogue bubbles (`[]`).
  - 1 sequence (`seq_b3247b4b3a2f591b`, Chainsaw Man fight scene) is completely silent across all 3 pages (`[[], [], []]`).
  - 64 dialogue entries consist entirely of punctuation or reaction symbols (e.g., `"..."`, `"!"`, `"?"`, `"!?"`).

---

## 2. Directory Layout & File Inventory

The physical dataset directory layout under `dataset/` is structured as follows:

```
dataset/
├── development/
│   ├── images/
│   │   ├── seq_032cbb4dc8686963/
│   │   │   ├── 01.png
│   │   │   ├── 02.png
│   │   │   └── 03.png
│   │   └── ... (80 sequence subdirectories, 240 PNG files)
│   └── labels.jsonl              (122,239 bytes, 80 lines)
├── test/
│   └── images/
│       ├── seq_14469aeb38963085/
│       │   ├── 01.png
│       │   ├── 02.png
│       │   └── 03.png
│       └── ... (15 sequence subdirectories, 45 PNG files)
├── sample_submission.jsonl       (870 bytes, 15 lines)
├── score.py                      (13,621 bytes, evaluation harness)
└── sequences.json                (25,138 bytes, 95 sequence index items)
```

### File Size & Volume Summary

| Path | Type | Count / Lines | Disk Size | Description |
| :--- | :--- | :--- | :--- | :--- |
| `dataset/sequences.json` | JSON | 95 items (858 lines) | 25.1 KB | Master index mapping sequence IDs to splits and page image paths |
| `dataset/score.py` | Python Script | 287 lines | 13.6 KB | Official evaluation metric script (v2.1) |
| `dataset/sample_submission.jsonl` | JSONL | 15 lines | 870 B | Template submission file containing all 15 test sequences with empty pages |
| `dataset/development/labels.jsonl` | JSONL | 80 lines | 122.2 KB | Ground truth annotations for all 80 development sequences |
| `dataset/development/images/` | Directory | 80 dirs, 240 PNGs | ~135 MB | 3-page scan sequences for development |
| `dataset/test/images/` | Directory | 15 dirs, 45 PNGs | ~15 MB | 3-page scan sequences for test |
| **Total Dataset Footprint** | | **285 images, 4 files** | **~150 MB** | Verified complete |

### Integrity Verification
- **Sequence Mapping:** Every sequence listed in `sequences.json` has all 3 referenced images physically present on disk.
- **Orphan Files:** No unindexed images, temporary files, `.DS_Store`, or extra artifacts exist anywhere in `development/` or `test/`.
- **Naming Standard:** Sequence IDs follow the pattern `seq_<16-character-hex-hash>`. Page scans are strictly named `01.png`, `02.png`, and `03.png`.

---

## 3. Split Analysis: Development vs. Test

| Attribute | Development Set | Test Set | Total / Combined |
| :--- | :--- | :--- | :--- |
| **Sequence Count** | 80 sequences (84.2%) | 15 sequences (15.8%) | 95 sequences |
| **Page / Image Count** | 240 images | 45 images | 285 images |
| **Pages per Sequence** | Exactly 3 (100%) | Exactly 3 (100%) | Exactly 3 (100%) |
| **Ground Truth Available** | Yes (`labels.jsonl`) | No (Hidden) | 80 labeled, 15 unlabeled |
| **Submission Match** | N/A | Exactly matches `sample_submission.jsonl` | 15 sequence IDs aligned |

---

## 4. Image Properties & Technical Specifications

All 285 scans in the dataset are encoded in the **PNG** format. However, their pixel encodings, dimensions, and file sizes exhibit significant variance.

### 4.1 Color Mode Distribution

| Split | Palette (`P`) | Grayscale (`L`) | Truecolor RGB (`RGB`) | Total Images |
| :--- | :--- | :--- | :--- | :--- |
| **Development** | 70 (29.2%) | 77 (32.1%) | 93 (38.8%) | 240 |
| **Test** | 34 (75.6%) | 11 (24.4%) | **0 (0.0%)** | 45 |
| **Overall** | **104 (36.5%)** | **88 (30.9%)** | **93 (32.6%)** | **285** |

> [!WARNING]
> **Zero RGB Scans in Test Set:** None of the 45 test images are stored as standard 3-channel RGB. Over 75% are 8-bit palette-indexed (`P`) images and 24% are single-channel grayscale (`L`). If an ML pipeline does not convert images via `.convert("RGB")`, it will crash or produce degraded input tensors during evaluation.

### 4.2 Image Dimension Statistics

Manga scans have varying resolutions depending on the source volume and scanning process.

| Metric | Width (px) | Height (px) | Aspect Ratio (W/H) |
| :--- | :--- | :--- | :--- |
| **Minimum** | 799 px | 1,200 px | 0.551 |
| **Maximum** | 997 px | 1,500 px | 0.722 |
| **Mean (Development)** | 855.5 px | 1,254.4 px | 0.682 |
| **Mean (Test)** | 892.2 px | 1,293.3 px | 0.690 |
| **Overall Mean** | 861.3 px | 1,260.5 px | 0.683 |

#### Most Frequent Dimensions
1. `(800, 1200)`: 93 images (32.6%) — predominantly Death Note scans.
2. `(891, 1300)`: 30 images (10.5%) — Silver Spoon / Chainsaw Man scans.
3. `(834, 1200)`: 21 images (7.4%) — Chainsaw Man scans.
4. `(799, 1200)`: 9 images (3.2%) — Test set sequences (`seq_e456...`, `seq_c562...`, `seq_75ae...`).
5. `(980, 1400)`: 6 images (2.1%) — High-resolution Death Note chapters.

### 4.3 Intra-Sequence Inconsistency
In **52 out of 95 sequences (54.7%)**, the three pages within a single sequence do not share identical dimensions or color modes. For instance:
- `seq_2032620aa4e4ac7f`: Page 1 is `(915, 1300)`, Page 2 is `(914, 1300)`, Page 3 is `(919, 1300)`.
- `seq_952f154fb1505883`: Page 1 is mode `P`, Page 2 is mode `L`, Page 3 is mode `P`.
- `seq_14469aeb38963085` (Test): Page 1 is `(991, 1400)` `P`, Page 2 is `(979, 1400)` `P`, Page 3 is `(978, 1400)` `L`.

**Implication:** Preprocessing pipelines must not assume sequence-level batching with fixed tensor dimensions across all 3 pages without proper padding or resizing.

### 4.4 File Size Distribution
- **Development Scans:** 148.5 KB to 1,319.1 KB (average: 572.2 KB).
- **Test Scans:** 155.5 KB to 585.5 KB (average: 321.6 KB).
- No corrupted or unreadable images were found. All 285 images open cleanly via standard PIL and OpenCV decoders.

---

## 5. Ground Truth Annotation Analysis (`development/labels.jsonl`)

The ground truth file `development/labels.jsonl` contains 80 lines corresponding to the 80 development sequences.

### 5.1 JSON Schema
Each JSON line conforms to the following schema:
```json
{
  "sequence_id": "seq_2032620aa4e4ac7f",
  "pages": [
    [
      {"speaker": "character1", "text": "HUH? DOES KOMABA SEEM DIFFERENT THAN USUAL?"},
      {"speaker": "character2", "text": "HE'S DOING REALLY WELL. IT LOOKS LIKE HE'S BEEN WORKING HARD FOR THE FALL TOURNAMENT."},
      {"speaker": "UNKNOWN", "text": "THAT'S THE WAY!"}
    ],
    [],
    [
      {"speaker": "character3", "text": "OKAWA-SENPAI, YOU LOOK SO COOL!"}
    ]
  ]
}
```

### 5.2 Dialogue and Density Statistics

| Metric | Page-Level | Sequence-Level | Full Dev Split |
| :--- | :--- | :--- | :--- |
| **Total Count** | 240 pages | 80 sequences | 1,753 dialogue bubbles |
| **Minimum Bubbles** | 0 | 0 | — |
| **Maximum Bubbles** | 19 | 47 | — |
| **Average (Mean)** | 7.30 bubbles / page | 21.91 bubbles / sequence | — |
| **Median** | 7.00 bubbles / page | 21.00 bubbles / sequence | — |

### 5.3 Empty Pages and Silent Sequences
- **Empty Pages:** 10 of the 240 dev pages (4.2%) have `[]` (no text).
- **Silent Sequences:** 1 sequence (`seq_b3247b4b3a2f591b`) is entirely empty across all 3 pages (`"pages": [[], [], []]`). It depicts an intense action scene in *Chainsaw Man* where combat occurs without speech.
- **Handling:** The evaluation harness supports empty pages and awards a perfect score (`1.0`) if an empty page is correctly predicted as `[]`.

### 5.4 Text Characteristics & Typography

- **Casing:**
  - **ALL-UPPERCASE:** 1,597 bubbles (**91.1%**) — standard manga scan typesetting.
  - **Mixed Case:** 111 bubbles (**6.3%**) — whispering, sound effects, or emphasis.
  - **All-Lowercase:** 45 bubbles (**2.6%**) — small background dialogue or murmurs.
- **Length Distribution:**
  - Character count per bubble: 1 to 201 characters (mean: 32.2 characters).
  - Word count per bubble: 1 to 37 words (mean: 6.0 words).
- **Punctuation-Only Bubbles:**
  - 64 dialogue items consist solely of punctuation marks or reaction symbols.
  - Examples: `"..."`, `"!"`, `"?"`, `"!?"`, `"...!!"`, `"!!!"`.
- **Character Encoding:**
  - 99.9% standard ASCII.
  - The only non-ASCII characters present in the entire ground truth are smart quotation marks:
    - `“` (`U+201C`, Left Double Quotation Mark)
    - `”` (`U+201D`, Right Double Quotation Mark)
  - `score.py` explicitly normalizes these to standard ASCII `"` during metric computation.

---

## 6. Speaker Attribution System (Diarization)

### 6.1 Speaker Label Vocabulary

The ground truth vocabulary contains exactly 10 unique labels:

| Speaker Label | Total Occurrences | Frequency (%) | Role / Meaning |
| :--- | :--- | :--- | :--- |
| `character1` | 657 | 37.48% | Primary speaker in the sequence |
| `character2` | 531 | 30.29% | Secondary speaker in the sequence |
| `character3` | 261 | 14.89% | Tertiary speaker in the sequence |
| `character4` | 109 | 6.22% | 4th character |
| `NARRATION` | 76 | 4.34% | Voiceover / narrative box (Fixed identifier) |
| `character5` | 57 | 3.25% | 5th character |
| `UNKNOWN` | 34 | 1.94% | Crowd voice / ambiguous speaker (Masked in gold) |
| `character6` | 23 | 1.31% | 6th character |
| `character7` | 3 | 0.17% | 7th character |
| `character8` | 2 | 0.11% | 8th character |

### 6.2 Sequence-Local Scope & Hungarian Matching
1. **Sequence-Local IDs:** Character labels (`character1`, `character2`, etc.) are **local to each sequence**, not global across the dataset. For example, `character1` is Denji in sequence A, but Light Yagami in sequence B.
2. **Hungarian Assignment:** `score.py` uses the Hungarian matching algorithm (`best_mapping()`) to establish a 1-to-1 maximum-weight matching between predicted speaker labels and ground truth labels per sequence.
   - Predictions do not need to literally predict `"character1"`. A model can output `"speaker_A"` or character names, and the evaluator will map them to ground truth character IDs to maximize alignment.
3. **`NARRATION` is Fixed:** Unlike character labels, `NARRATION` is a globally fixed identity. `score.py` hardcodes `mapping["NARRATION"] = "NARRATION"`. Predictions attributed to narrator boxes must explicitly be labeled `"NARRATION"`.
4. **`UNKNOWN` is Masked:** In the reference annotations, `UNKNOWN` indicates ambiguous background voices. The scorer converts reference `UNKNOWN` to `None`, meaning tokens aligned to `UNKNOWN` in ground truth do not penalize the speaker accuracy F1.

### 6.3 Speaker Distribution per Sequence
- **Distinct Characters per Sequence:** Ranges from 0 to 8 (mean: 3.23 characters).
- Distribution:
  - 0 characters: 1 sequence (the silent action sequence)
  - 1 character: 2 sequences
  - 2 characters: 12 sequences
  - 3 characters: 27 sequences (most frequent)
  - 4 characters: 20 sequences
  - 5 characters: 11 sequences
  - 6 to 8 characters: 6 sequences

---

## 7. Manga Series & Domain Analysis

Cross-referencing dialogue entities, character names, and visual layouts revealed that the 80 development sequences originate from 4 distinct manga series representing different visual styles and genres:

```
┌──────────────────────────────────────────────────────────┐
│  Manga Series Distribution in Development (80 seqs)     │
├──────────────────────────────────────────────────────────┤
│  Death Note (Ohba / Obata)         : 42 seqs (52.5%)     │
│  Chainsaw Man (Fujimoto)           : 26 seqs (32.5%)     │
│  Silver Spoon (Arakawa)            :  6 seqs (7.5%)      │
│  Kaguya-sama: Love Is War (Akasaka):  6 seqs (7.5%)      │
└──────────────────────────────────────────────────────────┘
```

### Domain Characteristics by Series

1. **Death Note (52.5%):**
   - **Characteristics:** Highly dense, expository dialogue; multi-paragraph text bubbles; complex panel layouts; detailed realistic art style.
   - **Key Entities:** Light, L, Ryuk, Misa, Shinigami, Kira.
   - **Scans:** Mostly 800x1200 or 980x1400. High proportion of grayscale and truecolor scans.
2. **Chainsaw Man (32.5%):**
   - **Characteristics:** Cinematic paneling, abrupt scene transitions, colloquial/profane dialogue, rapid back-and-forth exchanges, textless action spreads.
   - **Key Entities:** Denji, Pochita, Makima, Aki, Devil, Yakuza.
   - **Scans:** Mixed resolutions (834x1200, 891x1300, 997x1400).
3. **Silver Spoon (7.5%):**
   - **Characteristics:** Slice-of-life agricultural school setting; multi-character banter; speech bubbles clustered near heads.
   - **Key Entities:** Hachiken, Mikage, Komaba, Okawa.
4. **Kaguya-sama: Love Is War (7.5%):**
   - **Characteristics:** Romantic comedy psychological warfare; heavy use of narrative boxes (`NARRATION`); theatrical font variations and internal monologues.
   - **Key Entities:** Shinomiya, Shirogane, Fujiwara.

### Test Set Domain Clues
The 15 test sequences exhibit resolution and aspect ratio clusters identical to the development series:
- Sequences with height 1300 (`seq_6b18...`, `seq_c780...`, `seq_5303...`, `seq_d455...`): Match Silver Spoon / Chainsaw Man profiles.
- Sequences with height 1200 (`seq_e04e...`, `seq_6aa3...`, `seq_32c4...`): Match Chainsaw Man profiles.
- Sequences with height 1400 (`seq_81b1...`, `seq_1446...`, `seq_e9ff...`, `seq_316b...`, `seq_cf7d...`): Match Death Note / Chainsaw Man profiles.
- Sequences with width 799 and height 1200 (`seq_e456...`, `seq_c562...`, `seq_75ae...`): Match Death Note standard scan profiles.

---

## 8. Evaluation Architecture & Metric Deep Dive (`score.py`)

The evaluation script `score.py` (v2.1) executes strict syntactic verification followed by algorithmic sequence alignment and Hungarian speaker matching.

### 8.1 Strict Syntactic & Format Validation
The submission parser (`load_jsonl()`) enforces strict constraints. Any violation causes an immediate fatal error:
1. **JSON Object Keys:** Must contain **only** `"sequence_id"` and `"pages"`. Extra or missing keys trigger a `ValueError`.
2. **Duplicate Keys:** `no_duplicate_keys()` rejects any JSON object with duplicated keys.
3. **Sequence IDs:** Must be non-empty, trimmed strings matching the test set exactly. Extra or missing sequence IDs fail the evaluation.
4. **Page Count:** `pages` must be a list containing **exactly 3 page lists**.
5. **Item Keys:** Each line item within a page must contain **only** `"speaker"` and `"text"`.
6. **Speaker Rules:** In predicted submissions, `speaker` must be a non-empty, trimmed string. `speaker` cannot be empty, whitespace, or `None`.
7. **Text Rules:** `text` must be non-empty after normalization (`norm(text)` must not be empty). To indicate an empty page, the page list must be `[]` rather than containing empty strings.
8. **Control Characters:** Rejects text containing non-space Unicode control characters (`category(c) == "Cc"`).

### 8.2 Text Normalization Engine
Before scoring, text undergoes strict normalization in `norm()`:
- `NFKC` Unicode normalization.
- Transliteration of smart punctuation: `“` / `”` $\to$ `"`, `‘` / `’` $\to$ `'`, `…` $\to$ `...`, `–` / `—` $\to$ `-`.
- Casefolding: Converts all text to lowercase.
- Removal of format characters (category `Cf`).
- Multiple dots collapsed: `.{3,}` $\to$ `...`.
- Whitespace normalization and standard punctuation attachment.

### 8.3 Metric Formulations

The scorer reports 5 macro-averaged metrics:

$$\text{Macro Metric} = \frac{1}{|S|} \sum_{s \in S} \text{Metric}(s)$$

1. **`text_order_score` (Reading Order & OCR Accuracy):**
   - For each active page, concatenates all dialogue texts with spaces: $T = \text{norm}(\sum \text{text}_i)$.
   - Computes normalized Levenshtein edit distance between gold and predicted page text:
     $$\text{PageScore} = \max\left(0, 1 - \frac{\text{Levenshtein}(T_{\text{gold}}, T_{\text{pred}})}{\max(|T_{\text{gold}}|, |T_{\text{pred}}|, 1)}\right)$$
   - Sequence score is the average across active pages. Penalizes incorrect reading order because concatenated order changes edit distance.
2. **`matched_token_coverage` (Token Recall):**
   - Monotonic Dynamic Programming alignment (`align_tokens()`) matching tokens with fuzzy similarity (Levenshtein ratio $\ge 0.6$ for tokens $\ge 3$ characters; exact match for shorter tokens).
   - Ratio of aligned token weight to total gold tokens.
3. **`speaker_accuracy_on_matched`:**
   - Evaluates speaker attribution accuracy strictly on aligned tokens where gold speaker is known.
4. **`balanced_joint_f1` (Primary Competition Objective):**
   - Weights matched tokens inversely by speaker frequency:
     $$w = \frac{2 \times \text{weight}}{\text{freq}(\text{gold}) + \text{freq}(\text{pred})}$$
   - Runs Hungarian algorithm on balanced weights to prevent high-dialogue monologue characters from dominating rare characters.
   - Computes balanced macro-F1 over all speaker identities in the sequence.
5. **`joint_f1`:**
   - Standard micro F1 combining token recognition and speaker classification.

### 8.4 Empirical Baseline Benchmarks

Running `score.py` under controlled test inputs confirms:
- **Ground Truth vs. Ground Truth:** All 5 metrics achieve exactly `1.0000`.
- **All-Empty Submission (`[[], [], []]` on Dev):**
  - All macro metrics score `0.0000`.
  - For the 1 silent sequence (`seq_b3247b4b3a2f591b`), `text_order_score = 1.0` and `matched_token_coverage = 1.0`, but F1 is `None` (excluded from macro F1).

---

## 9. Engineering Implications for Upcoming ML Pipeline

Based on this reconnaissance, the upcoming ML pipeline design must incorporate the following architectural requirements:

| Pipeline Component | Finding from Reconnaissance | Technical Mandate |
| :--- | :--- | :--- |
| **Image Loading** | 100% of test images are `P` or `L` mode; 0% `RGB` | Always apply `Image.open(p).convert("RGB")` or `cv2.cvtColor` conversion immediately upon reading. |
| **Resizing / Padding** | Intra-sequence dimensions vary (e.g. 915 vs 919 px) | Avoid naive batching without aspect-ratio-preserving padding or processing pages independently. |
| **Reading Order** | Concatenated text edit distance evaluates sequence order | Order detected bubbles according to manga reading conventions: Top-to-Bottom, Right-to-Left within panels and across pages. |
| **Punctuation & Reactions** | 64 bubbles contain only `"..."`, `"!"`, `"?"` | Text post-processing must not filter out short or punctuation-only bubbles. |
| **Silent Pages** | 10 empty pages in dev; 1 entirely silent sequence | Model must be capable of predicting empty page lists `[]` when no dialogue bubbles exist, avoiding false positive hallucinations. |
| **Speaker Attribution** | Hungarian algorithm matches sequence-local IDs; `NARRATION` is fixed | Assign consistent arbitrary IDs (e.g. `c1`, `c2`) per sequence based on character detection / speech tail pointing, but strictly reserve `NARRATION` for narrative captions. |
| **Formatting Validation** | `score.py` rejects submissions with duplicate keys, trailing spaces, control characters, or non-trimmed IDs | Implement a validation check script mirroring `load_jsonl()` before generating final submission files. |

---

## 10. Conclusion

The dataset is verified, consistent, and ready for model development. No data corruption, missing files, or schema violations exist. All pipeline constraints and evaluation characteristics are documented and mapped to actionable engineering guidelines.
