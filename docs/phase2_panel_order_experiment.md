# OtakuScan: Phase 2.4 — Panel-Aware Reading Order Experiment

**Date:** October 13, 2026  
**Split Inspected:** Fold 0 (16 sequences, 48 pages)  
**Predictions Evaluated:** `predictions/panel_order_fold0.jsonl`  
**Evaluation Report:** `logs/panel_order_fold0_eval.json`  
**Previous Baselines:**
- Preserved Baseline (Phase 1.2): `predictions/baseline_fold0.jsonl` | `logs/baseline_fold0_eval.json`
- Preserved Clustered Pipeline (Phase 2.2): `predictions/clustered_fold0.jsonl` | `logs/clustered_fold0_eval.json`

---

## 1. Executive Summary

In Phase 2.3 error analysis, cross-panel column interleaving was identified as the root defect preventing reading order scores from advancing beyond ~0.78. Because standard manga layout requires reading down an entire right-hand panel column before transitioning to the adjacent left-hand panel column, full-page horizontal row-banding frequently inverted conversational exchanges across side-by-side panels.

In Phase 2.4, we designed and implemented a lightweight, geometry-based panel-aware reading order sorter using **Recursive Spatial Decomposition (Hierarchical XY-Cut)**, avoiding naive 50/50 page splits and preserving existing behaviors behind the `reading_order_mode` configuration parameter.

### Macro-Level Metric Comparison (Fold 0)

| Metric | Preserved Baseline | Clustered Pipeline | Panel-Aware Order | Delta vs Clustered | Relative Change |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`text_order_score`** | `0.6126` | `0.7855` | **`0.8053`** | **+0.0198** | **+2.5%** |
| **`matched_token_coverage`** | `0.5731` | `0.6617` | **`0.6736`** | **+0.0119** | **+1.8%** |
| **`joint_f1`** | `0.3430` | `0.3860` | **`0.3922`** | **+0.0062** | **+1.6%** |
| **`balanced_joint_f1`** (Primary) | `0.1531` | `0.1710` | **`0.1727`** | **+0.0017** | **+1.0%** |
| **`speaker_accuracy_on_matched`** | `0.5486` | `0.5397` | **`0.5410`** | **+0.0013** | **+0.2%** |

- **All five official macro metrics improved.**
- `text_order_score` successfully crossed the **0.80 barrier** (reaching **0.8053**).
- Computational overhead remains negligible: total evaluation runtime across all 48 pages was **29.1s** (0.607s/page) with peak memory of **64.9 MB**, well below VM constraints.

---

## 2. Algorithm Design: Hierarchical XY-Cut

Rather than assuming rigid grid layouts or splitting every page down the vertical centerline, the panel-aware sorter uses recursive whitespace gutter analysis:

```
                          Input Candidate Clusters
                                     │
                      Does a continuous horizontal gutter
                       exist across the current block?
                                    ╱ ╲
                             Yes   ╱   ╲   No
                                  ╱     ╲
                  Split into Tiers       Does a continuous vertical gutter
                   (Top to Bottom)        exist across the current block?
                          │                             ╱ ╲
                   Recurse on each               Yes   ╱   ╲   No
                        Tier                          ╱     ╲
                          │             Split into Columns   Fallback: Row-Banding
                          │              (Right to Left)      Within-Row Right-to-Left
                          │                     │             (sort_reading_order)
                          │              Recurse on each                │
                          │                  Column                     │
                          │                     │                       │
                          └─────────────┬───────┴───────────────────────┘
                                        ▼
                              Concatenated Reading Order
```

### Mathematical & Geometric Rules
1. **Tier Partition (Horizontal Cuts):**
   - Regions are swept vertically along $y_{min}$.
   - A horizontal cut is established if the vertical gap between the cumulative maximum $y_{max}$ of the upper set and the $y_{min}$ of the next region satisfies:
     $$\Delta y = y_{min}^{(i)} - \max_{j < i} (y_{max}^{(j)}) \ge g_y$$
     where $g_y = 40.0\text{ px}$.
   - Tiers are ordered strictly Top-to-Bottom.
2. **Column Partition (Vertical Cuts):**
   - If no horizontal cut divides the current block, regions are swept horizontally along $x_{min}$.
   - A vertical cut is established if the horizontal gap between the cumulative maximum $x_{max}$ of the left set and the $x_{min}$ of the next region satisfies:
     $$\Delta x = x_{min}^{(i)} - \max_{j < i} (x_{max}^{(j)}) \ge g_x$$
     where $g_x = 25.0\text{ px}$.
   - In manga convention, columns are ordered strictly Right-to-Left ($\text{reversed}$).
3. **Recursive Resolution & Fallback:**
   - Vertically stacked speech bubbles inside a right column are fully resolved top-to-bottom before the algorithm processes adjacent left-column bubbles.
   - If neither axis has a qualifying gutter (e.g. tightly packed dialogue inside a single panel), the algorithm falls back to horizontal row-banding (`sort_reading_order`).

---

## 3. Per-Sequence Performance Comparison

Every sequence was evaluated using the official scoring bridge ([`src/eval/scorer.py`](file:///home/neo/Desktop/otakuscan/src/eval/scorer.py)):

| Sequence ID | Series Domain | Clustered `text_order` | Panel-Aware `text_order` | `text_order` Delta | `matched_coverage` Delta | `balanced_f1` Delta |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`seq_2032620aa4e4ac7f`** | *Silver Spoon* | 0.6053 | 0.5968 | **-0.0085** | +0.0000 | +0.0000 |
| **`seq_45c9871f52e99417`** | *Jujutsu Kaisen* | 0.6920 | 0.6920 | +0.0000 | +0.0000 | +0.0000 |
| **`seq_5f9ee35846b7e635`** | *Death Note* | 0.8400 | 0.8400 | +0.0000 | +0.0000 | +0.0000 |
| **`seq_84541a1f4b159018`** | *Death Note* | 0.8225 | 0.8225 | +0.0000 | +0.0000 | +0.0000 |
| **`seq_952f154fb1505883`** | *Silver Spoon* | 0.7562 | 0.8911 | **+0.1348** | **+0.0244** | **+0.0020** |
| **`seq_96839890d69d7698`** | *Chainsaw Man* | 0.8098 | 0.8098 | +0.0000 | +0.0000 | +0.0000 |
| **`seq_a3bd36201a0cd61b`** | *Kaguya-sama* | 0.7341 | 0.7657 | **+0.0316** | **+0.0286** | +0.0000 |
| **`seq_b136f32c2233b97c`** | *Death Note* | 0.9593 | 0.9593 | +0.0000 | +0.0000 | +0.0000 |
| **`seq_b8b5563330fdc89d`** | *Death Note* | 0.7496 | 0.8020 | **+0.0523** | **+0.0211** | **+0.0072** |
| **`seq_b9b5a9ad9a64ced9`** | *Chainsaw Man* | 0.8019 | 0.8655 | **+0.0637** | **+0.0200** | +0.0000 |
| **`seq_d749b853cf795b6a`** | *Death Note* | 0.7882 | 0.7882 | +0.0000 | +0.0000 | +0.0000 |
| **`seq_d8afceb0e8f9122d`** | *Chainsaw Man* | 0.8585 | 0.8569 | **-0.0016** | +0.0000 | +0.0000 |
| **`seq_da71f7a842c91090`** | *Jujutsu Kaisen* | 0.8146 | 0.8146 | +0.0000 | +0.0000 | +0.0000 |
| **`seq_de542b63e6b19bc2`** | *Death Note* | 0.7128 | 0.7573 | **+0.0444** | **+0.0962** | **+0.0183** |
| **`seq_e8b4f75454f9491f`** | *Chainsaw Man* | 0.9051 | 0.9051 | +0.0000 | +0.0000 | +0.0000 |
| **`seq_f7f9d9dbf5b05cb5`** | *Death Note* | 0.7184 | 0.7184 | +0.0000 | +0.0000 | +0.0000 |

---

## 4. In-Depth Analysis of Major Improvements

### 1. `seq_952f154fb1505883` (*Silver Spoon*): +0.1348 (`text_order`: 0.7562 $\to$ 0.8911)
- **Failure Mode in Clustered Sorter:** In Page 1, a middle tier contained a tall right-hand speech bubble ($y \in [480, 680]$) alongside two stacked dialogue bubbles in the left panel ($y \in [510, 580]$ and $y \in [620, 690]$). The row-band sorter interleaved the left-hand bubbles between the upper and lower halves of the right-hand bubble.
- **Resolution:** The vertical gutter ($g_x = 25\text{ px}$) cleanly separated the right and left panel columns. The right panel was completely emitted before the left panel, fixing the entire conversational flow.

### 2. `seq_b8b5563330fdc89d` (*Death Note*): +0.0523 (`text_order`: 0.7496 $\to$ 0.8020)
- **Diagnosed Error from Phase 2.3:** Bottom tier contained right panel bubble `"W'HO DARLIN'..."` ($y \in [980, 1068]$) and left panel bubbles `"THERE THEY GO AGAIN"` ($y \in [834, 876]$) and `"BRIGHT AS THE NIGHT IS LONG"` ($y \in [833, 890]$).
- **Resolution:** Tier 4 was partitioned into a right column ($x \ge 433$) and a left column ($x \le 395$). The right column was traversed first (emitting C9 $\to$ C10 $\to$ C13), followed by the left column (C12 $\to$ C11), perfectly recovering ground truth order.

### 3. `seq_b9b5a9ad9a64ced9` (*Chainsaw Man*): +0.0637 (`text_order`: 0.8019 $\to$ 0.8655)
- **Resolution:** Multi-panel action sequence with vertical panel splits was cleanly resolved into sequential columns, eliminating cross-panel jumps.

---

## 5. In-Depth Analysis of Regressions & Known Failure Cases

### 1. `seq_2032620aa4e4ac7f` (*Silver Spoon*): -0.0085 (`text_order`: 0.6053 $\to$ 0.5968)
- **Root Cause:** Artwork / Screentone False Positive (`大级夫装` at $x=809, y=655$).
- **Mechanism:** In Page 1, two dialogue panels sit side-by-side: Right panel has `"HUH? DOES KOMABA SEEM DIFFERENT..."` ($y=420$); Left panel has `"HE'S DOING REALLY WELL"` ($y=498$) and `"IT LOOKS LIKE HE'S BEEN WORKING HARD..."` ($y=546$). An unrecognized Japanese kanji/screentone noise detection sat at the bottom right margin ($x=809, y=655$).
- Because the vertical gutter ran down the right side of the page, the sorter grouped the noise glyph with the right panel, emitting the noise glyph *before* the legitimate left-panel dialogue.
- In the baseline row-bander, the noise glyph had $y=655$ and was naturally ordered *after* the dialogue ($y \le 546$).

### 2. `seq_d8afceb0e8f9122d` (*Chainsaw Man*): -0.0016 (`text_order`: 0.8585 $\to$ 0.8569)
- **Root Cause:** Non-uniform bubble spacing in multi-bubble character monologue.
- **Mechanism:** In Page 3, a single character's dialogue was split into two bubbles separated by a 63px gap ($y \in [157, 233]$ and $y \in [296, 391]$). Because $63 > g_y = 40$, the algorithm treated the gap as a horizontal tier boundary, slightly displacing an adjacent background whimpering text.

---

## 6. Limitations of Pure Bounding-Box Geometry

This experiment illuminates the theoretical performance ceiling of purely geometric text-region ordering:
1. **Absence of Ground Truth Panel Border Signals:** Pure bounding-box geometry operates on speech bubbles without observing panel dividing gutters. When speech bubbles are sparse or artwork noise intrudes, pure geometric gutters can slice across visual panel lines.
2. **False Positive Sensitivity:** As shown in `seq_2032620aa4e4ac7f`, non-dialogue noise (SFX, screentone artifacts) distort geometric column boundaries. Filtering SFX and margin text will naturally clean up these boundaries.

---

## 7. Recommendation

### **Retain the Panel-Aware Sorter Experiment as the Active Pipeline Default**

**Rationale:**
1. **Net Empirical Advantage:** All five macro metrics improved across Fold 0:
   - `text_order_score`: **+0.0198** (0.7855 $\to$ 0.8053)
   - `matched_token_coverage`: **+0.0119** (0.6617 $\to$ 0.6736)
   - `balanced_joint_f1`: **+0.0017** (0.1710 $\to$ 0.1727)
   - `joint_f1`: **+0.0062** (0.3860 $\to$ 0.3922)
2. **5 Sequences Substantially Improved vs 2 Negligible Regressions:** Net gains (+0.1348, +0.0637, +0.0523, +0.0444, +0.0316) drastically outweigh the minor regressions (-0.0085, -0.0016), which are themselves primarily byproducts of unpruned SFX/noise.
3. **Completely Decoupled & Preserved:** Both `sort_reading_order` (row-banding) and `sort_panel_aware_reading_order` (recursive XY-cut) are preserved in `src/models/reading_order.py`, selectable via `reading_order_mode="row_band"` or `reading_order_mode="panel_aware"`.
