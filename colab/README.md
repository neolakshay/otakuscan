# OtakuScan: Google Colab GPU Workflow & VLM Pilot Experiment (Phase 3B)

This guide documents the execution of the GPU-assisted training workflow and Vision-Language Model (VLM) pilot experiment in **Google Colab Premium**.

---

## 1. System Architecture & Compute Roles

| Environment | Primary Role | Hardware Specs | Constraints |
| :--- | :--- | :--- | :--- |
| **Local Ubuntu VM** (Antigravity IDE) | Code authoring, unit tests, fast local baseline evaluation | 4 CPU cores, ~5.7 GiB RAM | Lightweight CPU inference only (<1.2s/page); no large models |
| **Google Colab Premium** | GPU-accelerated model verification & VLM pilot experiments | NVIDIA T4 / V100 / A100 / L4 GPU, 15–40 GB VRAM | Reserved for heavy deep-learning workloads |

### Architectural Clarification: GPU vs CPU Classifier
- Our deployed dialogue filter is an **L2-regularized linear model over 20 geometric and lexical features**. It trains in **0.57 seconds on CPU**, evaluates in **< 0.1 ms per page**, and requires zero deep-learning GPU dependencies for production inference.
- Moving this 20-feature logistic regression to a GPU purely for the sake of using a GPU adds unnecessary overhead. We preserve the verified lightweight classifier for local CPU inference.
- Colab GPU resources provide genuine value for:
  1. Running and evaluating Vision-Language Models (`Qwen2-VL-2B-Instruct`) on difficult manga typography without CPU throttling.
  2. Teacher-distillation and offline error recovery analysis.

---

## 2. Directory Contents

- **`phase3_colab_training.ipynb`**: Complete, self-contained interactive Jupyter notebook ready to open in Google Colab.
- **`train_classifier_colab.py`**: Standalone training & reproducibility script enforcing strict Fold 0 training split isolation and hyperparameter sensitivity exploration.
- **`vlm_manga_transcription_experiment.py`**: Standalone pilot script evaluating `Qwen/Qwen2-VL-2B-Instruct` vs RapidOCR on permitted development pages, recording loading time, per-page latency, VRAM allocation, and token F1.
- **`README.md`**: Step-by-step setup and execution instructions.

---

## 3. Step-by-Step Instructions for Google Colab

### Step 1: Open the Notebook in Google Colab
1. In your browser, navigate to [Google Colab](https://colab.research.google.com).
2. Click **Upload** and upload [colab/phase3_colab_training.ipynb](file:///home/neo/Desktop/otakuscan/colab/phase3_colab_training.ipynb).

### Step 2: Enable GPU Hardware Accelerator
1. In the top menu, select: **Runtime > Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU** (or A100 GPU if available).
3. Click **Save**.

### Step 3: Provide Project Files & Dataset
Choose one of the following methods to make the code and data available in Colab:
- **Option A (Google Drive - Recommended for persistence):**
  1. Upload the `otakuscan/` project folder to your Google Drive under `MyDrive/otakuscan`.
  2. In Cell 2 of the notebook, uncomment the drive mount code:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     project_root = Path('/content/drive/MyDrive/otakuscan')
     ```
- **Option B (Direct Upload / ZIP):**
  1. Zip the repository (including `dataset/`, `splits/`, `src/`, `colab/`, `models/`).
  2. Upload to `/content/` and unzip: `!unzip -q otakuscan.zip -d /content/otakuscan`.

### Step 4: Run the Notebook Cells Sequentially
Execute the notebook cells from top to bottom:
- **Cell 1 (GPU Detection):** Verifies active GPU (e.g. `Tesla T4`, `A100-SXM4`) and available VRAM.
- **Cell 2 (Workspace Setup):** Sets `project_root` and verifies that `dataset/` is present and read-only.
- **Cell 3 (Install Dependencies):** Installs `rapidocr-onnxruntime`, `transformers>=4.45.0`, `accelerate`, and `qwen-vl-utils`.
- **Cell 4 (Leakage Audit):** Confirms that Fold 0 training sequences have **zero intersection** with validation sequences.
- **Cell 5 (Classifier Verification):** Runs training and hyperparameter sensitivity exploration across $L_2 \in [0.001, 0.1]$ and decision thresholds.
- **Cell 6 (VLM Pilot Experiment):** Downloads weights for `Qwen/Qwen2-VL-2B-Instruct` (~4.5 GB, cached in Colab memory), runs inference on 3 development pages, and records GPU latency, VRAM allocation, and token extraction metrics.
- **Cell 7 (Results Comparison):** Displays the side-by-side comparison table between RapidOCR and Qwen2-VL.

---

## 4. What Logs and Metrics to Return

After executing Cell 6 and Cell 7 in Google Colab:
1. **Download the Pilot Log:**
   Download the generated JSON file located at:
   `logs/colab_vlm_pilot_results.json`
2. **Copy the Output Table:**
   Copy the text output from Cell 1 (GPU info) and Cell 7 (Comparison table), for example:
   - GPU Model & Allocated VRAM
   - Qwen2-VL Checkpoint Load Time (seconds)
   - Per-page Latency for RapidOCR vs Qwen2-VL
   - Token Precision, Recall, and F1 for both models
   - Peak VRAM allocated by Qwen2-VL (MB / GB)
   - Any observed error modes (e.g. hallucinations, sound-effect filtering, reading-order inversions).
