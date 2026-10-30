# Licence of the data files in this repository

The files we produced ourselves — the image-level split of Kvasir-VQA-x1 (`data/splits/kvasir_vqa_x1_image_split/`), the two
re-splits of VQA-RAD (`data/splits/vqarad_rediv/`, `data/splits/vqarad_qlevel/`), the image-swap maps (`data/swaps/`), the
normalisation tables (`data/normalization/`), the training-run metadata (`results/training_runs/`), the aggregated results
(`results/aggregates/`) and `paper/numbers.csv` — are released under the Creative Commons Attribution 4.0 International licence
(CC BY 4.0, https://creativecommons.org/licenses/by/4.0/).

**Files derived from a dataset follow that dataset's licence, not the MIT licence of the code and not CC BY 4.0.** This
applies in particular to the per-question prediction files, which contain for each test question the question identifier,
the model's answer and the gold answer of the dataset (no image):

| files | derived from | licence of the dataset (checked on 30 Oct 2026) |
|---|---|---|
| `results/predictions/KVASIR/*`, `data/swaps/image_swap_KVASIR.json`, `data/splits/kvasir_vqa_x1_image_split/*`, `data/normalization/kvasir_*` | Kvasir-VQA-x1 (SimulaMet; Gautam et al., 2025) | **CC BY-NC 4.0** — the dataset card at https://huggingface.co/datasets/SimulaMet/Kvasir-VQA-x1 states "Released under CC BY-NC 4.0 – for academic and non-commercial use. Please cite appropriately." Academic and non-commercial use only; attribution to the dataset |
| `results/predictions/SLAKE/*`, `data/swaps/image_swap_SLAKE.json` | SLAKE (Liu et al., 2021) | **CC BY 4.0** as declared by the licence tag of https://huggingface.co/datasets/BoKelvin/SLAKE; the official site https://www.med-vqa.com/slake/ states no licence, and the English copy we used (https://huggingface.co/datasets/mdwiratathya/SLAKE-vqa-english) declares none. Attribution to the dataset |
| `results/predictions/VQARAD_REDIV/*`, `results/predictions/VQARAD_QLEVEL/*`, `data/splits/vqarad_rediv/*`, `data/splits/vqarad_qlevel/*`, `data/swaps/image_swap_VQARAD_*.json` | VQA-RAD (Lau et al., 2018) | **CC0 1.0 Universal** — licence of the OSF project https://osf.io/89kps/ (also the tag of https://huggingface.co/datasets/flaviagiammarino/vqa-rad) |

No file in this repository is derived from HEAL-MedVQA, VinDr-CXR, MIMIC-CXR, MIMIC-CXR-JPG, Medical-Diff-VQA or Chest
ImaGenome beyond aggregated accuracies, gains and confidence intervals; those datasets are distributed by PhysioNet under
credentialed-access agreements that forbid redistribution.

The code is under the MIT licence (`LICENSE`).
