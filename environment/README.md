# Environments

Two conda environments were used; `requirements-vlm.txt` and `requirements-m3ae.txt` are their exact `pip freeze` outputs.

| environment | used for | Python | key versions |
|---|---|---|---|
| `requirements-vlm.txt` | Qwen2-VL-2B and SmolVLM2-2.2B fine-tuning and predictions (`src/med_vlm`, `src/harness/extract_predictions_circle.py`), the PubMedBERT and TF-IDF classifiers, every analysis, the numbers sheet, tables and figure | 3.13.15 | torch 2.13.0+cu130, transformers 5.15.1, peft 0.20.0, datasets 4.8.5, accelerate 0.34.2, scikit-learn 1.9.0, numpy 2.5.2, scipy 1.18.1, pyarrow 25.0.1, matplotlib 3.11.1 |
| `requirements-m3ae.txt` | M3AE (official code + `src/m3ae_patch`) | 3.8.20 | torch 1.9.0+cu111, pytorch-lightning 1.3.2, transformers 4.6.0, sacred 0.8.2, timm 0.4.12 |

Notes

- Both VLM families are loaded in **bfloat16** for the vision encoder and the language model alike (`src/med_vlm/models/loader.py`). With `torch_dtype="auto"`, SmolVLM2 loads its vision encoder in float16 and its language model in bfloat16, which breaks the fine-tuned-encoder approach; the explicit dtype avoids it. Qwen2-VL's image processor is capped at 512×512 pixels (`_QWEN2VL_MAX_PIXELS` in the loader).
- Hardware: one NVIDIA RTX 5000 Ada Generation (32 GB) per run; driver 580.178.04 / CUDA 13.0 at the time of the runs. Training times and per-process peak memory are in `results/training_runs/` and the numbers sheet (section 9).
- The figure is byte-reproducible with matplotlib 3.11.1 and its bundled STIX fonts (the rc asks for Nimbus Roman / STIXGeneral / Times New Roman; STIXGeneral was the one resolved).
- `MEDVQA_ROOT`: scripts resolve the original run tree from this environment variable (README, "Paths").
