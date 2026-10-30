import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import enum


class ModelName(str, enum.Enum):
    # Medical CLIP encoders (used in encoder-swap experiments)
    BLIP = "Salesforce/blip2-opt-2.7b"
    MEDCLIP = "ZiyueWang/med-clip"
    BIOMEDCLIP = "BiomedCLIP-vit-bert-hf"
    MEDVLM = "JZPeterPan/MedVLM-R1"
    PUBMEDCLIP = "flaviagiammarino/pubmed-clip-rn50-224"
    # DINOv2-based medical encoders (have CLS token → encoder_type=clip)
    # Phikon-v2: pathology-pretrained ViT-L/16 (Owkin, 1024-dim)
    PHIKON_V2 = "owkin/phikon-v2"
    # Rad-DINO: radiology DINOv2 ViT-B/14 (Microsoft, 768-dim)
    RAD_DINO = "microsoft/rad-dino"
    # General-purpose small VLMs for comparison
    GRANITE_VISION = "ibm-granite/granite-vision-3.2-2b"
    QWEN2VL = "Qwen/Qwen2-VL-2B-Instruct"
    SMOLVLM = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    # Medical-domain VLMs
    LLAVA_MED = "chaoyinshe/llava-med-v1.5-mistral-7b-hf"
    # Qwen2.5-VL-7B + RL sobre PMC-VQA -- candidato ao quadrante acuracia
    # alta / VDD baixa (ver docs/superpowers/specs/2026-08-25-medvlthinker-third-model-design.md)
    MEDVLTHINKER = "UCSC-VLAA/MedVLThinker-7B-RL_PMC"
    # v7 (2026-10-09): largura zero-shot do artigo B -- base generalista do MedVLThinker, sem RL medico
    QWEN25VL = "Qwen/Qwen2.5-VL-7B-Instruct"
    # v7 (2026-10-10): MedGemma 1.5 4B-IT (gated; o cache tem medgemma-1.5-4b-it, nao o medgemma-4b-it do plano)
    MEDGEMMA = "google/medgemma-1.5-4b-it"
    # v7 (2026-10-13, candidato): zero-shot novos. Lingshu e Huatuo (versao Qwen2.5-VL) reaproveitam o caminho do Qwen2.5-VL; InternVL3 (-hf, nativo do transformers) entrada propria
    LINGSHU = "lingshu-medical-mllm/Lingshu-7B"
    HUATUO_QWEN25 = "FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL"
    INTERNVL3 = "OpenGVLab/InternVL3-8B-hf"
    GEMMA3 = "google/gemma-3-4b-it"   # v7 (2026-10-13): restrito; termos aceitos e login feito pelo candidato


class PEFTMethod(str, enum.Enum):
    LORA = "lora"
    ADALORA = "adalora"
    DORA = "dora"
    NONE = "none"


class DatasetName(str, enum.Enum):
    SLAKE = "mdwiratathya/SLAKE-vqa-english"
    VQA_MED = "simwit/vqa-med"
    PATH_VQA = "flaviagiammarino/path-vqa"
    VQA_RAD = "flaviagiammarino/vqa-rad"
    # Gastrointestinal endoscopy VQA
    KVASIR_VQA = "SimulaMet/Kvasir-VQA-x1"
    # HEAL-MedVQA (VinDr-CXR, credenciado): arrows locais gerados por
    # harness/isbi_occlusion/heal/make_heal_train_arrows.py (E5, 15/09/2026);
    # nunca passa pelo load_dataset() do HF.
    HEAL = "heal-local"
    # MIMIC-CXR-JPG 2.1.0 + Medical-Diff-VQA 1.0.1 (imagem única, PhysioNet, credenciado): arrows locais
    # gerados por ${MEDVQA_ROOT}/v7_mimic/scripts/05_make_arrows.py (P4, 2026-10-10); nunca passa pelo HF.
    MIMIC = "mimic-local"
    # Virtual union of all four datasets above
    COMBINED = "combined"
