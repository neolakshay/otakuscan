# Evaluation Record: Vision-Language Model (VLM) Pilot Experiment

## 1. Rationale & Objective
The VLM pilot experiment evaluated whether an open-weight Vision-Language Model (`Qwen/Qwen2-VL-2B-Instruct`) could replace or augment our localized OCR and heuristic layout pipeline. 
Manga transcription presents unique visual challenges:
- Hand-drawn and stylized typography
- Vertical and diagonal text orientations
- Screentone interference and artistic background noise
- Non-linear reading order across dynamic comic panels

The objective was to test whether a modern multimodal transformer could perform end-to-end page recognition, reading order sorting, and speaker classification in a unified pass.

---

## 2. Quantitative Comparison: RapidOCR vs. Qwen2-VL-2B Pilot

The pilot was evaluated under controlled conditions on permitted development pages from the Fold 0 training split:

| Metric / Attribute | RapidOCR + Adapted Pipeline (Deployed) | Qwen2-VL-2B-Instruct Pilot |
|---|---|---|
| **Primary Compute Engine** | ONNX Runtime (CPU) | PyTorch / Transformers (CUDA GPU) |
| **Hardware Requirement** | CPU only (4–6 vCPUs, 63 MB RAM) | NVIDIA T4 / A100 GPU (4.8 GB VRAM) |
| **Inference Latency (GPU)** | N/A | ~1.80 s / page (NVIDIA T4) |
| **Inference Latency (Local CPU)** | **~0.58 – 1.15 s / page** | **>24.0 s / page** (20×–40× slower, unviable) |
| **Model Footprint on Disk** | **~16 MB** (ONNX) + 4 KB weights | **~4.5 GB** (PyTorch float16 weights) |
| **Memory Footprint (RAM/VRAM)** | **~63.6 MB RAM** | **~4.8 GB VRAM** (exceeds local memory headroom) |
| **Local VM Budget Compliance** | **100% compliant** (₹0 budget, CPU VM) | **Non-compliant** (requires GPU / extreme CPU timeout) |
| **Page 1 Token F1 (Dev)** | **0.8667** (P: 0.9286, R: 0.8125) | Variable / Prompt-sensitive (unstable formatting) |
| **Page 2 Token F1 (Dev)** | **0.8750** (P: 0.8750, R: 0.8750) | Incomplete free-form JSON generation |
| **Page 3 Token F1 (Dev)** | **0.9091** (P: 0.9091, R: 0.9091) | Truncated lines on complex panel layout |

---

## 3. Decision Rationale: Why Qwen2-VL Was Not Selected

1. **Extreme Latency & Compute Incompatibility**:
   - On the target deployment environment (Ubuntu CPU VM, ₹0 budget), Qwen2-VL-2B required over 24 seconds per page on CPU, exceeding acceptable inference bounds by over 20× and risking system memory exhaustion.
   - RapidOCR operates in 0.58s per page on CPU with 63.6 MB RAM.
2. **Structural Reliability & Output Hallucinations**:
   - Zero-shot / few-shot prompting of Qwen2-VL-2B frequently produced conversational preambles, dropped multi-line bubbles, or generated malformed JSON strings requiring brittle heuristic regex fallbacks.
   - In contrast, the RapidOCR + DBSCAN clustering + learned linear filter pipeline outputs deterministic bounding boxes and consistent structured dialogue without generation noise.
3. **Budget & Dependency Footprint**:
   - Deploying Qwen2-VL required heavy dependencies (`torch`, `transformers>=4.45.0`, `accelerate`, `qwen-vl-utils`) and 4.5 GB of model weights.
   - RapidOCR uses lightweight `onnxruntime` with a self-contained 16 MB footprint and 4 KB JSON weight files.

---

## 4. Limitations of the Pilot Study

This pilot was a preliminary exploration subject to specific constraints:
1. **Zero-Shot / Prompt-Only Baseline**: The pilot evaluated off-the-shelf `Qwen2-VL-2B-Instruct` using prompt templates without task-specific supervised fine-tuning (SFT) or LoRA adaptation on manga transcripts.
2. **Model Scale & Quantization**: Larger open models (e.g., Qwen2-VL-7B, Qwen2-VL-72B) or 4-bit quantized variants (AWQ/GPTQ) were not comprehensively evaluated locally due to hardware and budget limits.
3. **Scope of Conclusion**: The negative result demonstrates that **zero-shot 2B-scale general VLMs on CPU are unsuited for this specific zero-budget, low-latency deployment constraints**. It does **not** prove that all vision-language architectures are fundamentally inferior for manga transcription when trained with domain-specific fine-tuning on dedicated GPU infrastructure.
