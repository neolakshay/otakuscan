# OtakuScan: Google Colab GPU Workflow (One-Click Setup)

This directory provides the one-click Google Colab workflow for **OtakuScan Phase 3**.

---

## Quickstart (4 Steps)

### 1. Upload the Dataset Archive Once
Create a tarball of your local `dataset/` directory and upload it to your Google Drive:
- Archive name: `otakuscan-dataset.tar.gz`
- Google Drive path: `MyDrive/otakuscan-dataset/otakuscan-dataset.tar.gz`

*(Command to create archive locally if needed: `tar -czf otakuscan-dataset.tar.gz -C dataset .`)*

### 2. Open the Notebook from GitHub
Open the notebook directly in Google Colab from GitHub:
- [Open phase3_colab_training.ipynb in Colab](https://colab.research.google.com/github/neolakshay/otakuscan/blob/main/colab/phase3_colab_training.ipynb)

### 3. Select a T4 GPU Runtime
In Google Colab:
- Navigate to **Runtime > Change runtime type**.
- Select **T4 GPU** (or A100 GPU) under *Hardware accelerator*.
- Click **Save**.

### 4. Run the Cells Sequentially
Run all cells via **Runtime > Run all** (or step-by-step):
- **Cell 1 (GPU Audit):** Confirms GPU model and VRAM.
- **Cell 2 (Dependencies):** Installs required packages.
- **Cell 3 (Workspace Setup):** Mounts Google Drive, clones the repository, and extracts the dataset into `/content/otakuscan-data/dataset` (reusing existing extraction if already present).
- **Cell 4 (Leakage Audit):** Verifies zero sequence overlap between Fold 0 training and validation.
- **Cell 5 (Classifier Verification):** Retrains and verifies the dialogue filter and narration classifier weights on Fold 0 training data.
- **Cell 6 (VLM Pilot):** Runs `Qwen/Qwen2-VL-2B-Instruct` side-by-side with RapidOCR on 3 permitted development pages.
- **Cell 7 (Results):** Prints the benchmark table and saves `logs/colab_vlm_pilot_results.json`.

---

## Output to Return
After running the notebook, download and share:
1. `logs/colab_vlm_pilot_results.json`
2. The benchmark comparison table printed in Cell 7.
