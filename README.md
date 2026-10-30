# medvqa-image-gain

Code, configurations, data splits and aggregated results for the paper

> **How Much Does the Image Add? Decomposing the Accuracy of Fine-Tuned Medical VQA Models**
> Claudio Leite and Jurandy Almeida (Department of Computing, Federal University of São Carlos, Sorocaba, Brazil). Submitted to IEEE ISBI 2027.

The paper fine-tunes the same model, with the same recipe and seeds, in three approaches (text-decoder-only, frozen visual
encoder, fine-tuned visual encoder) on five closed-form Med-VQA benchmarks and three model families (Qwen2-VL-2B, SmolVLM2-2.2B,
M3AE), compares the text-decoder-only accuracy with external baselines (PubMedBERT and TF-IDF question classifiers, majority answer,
per-question majority, grey image), decomposes the image gain into the frozen-encoder and fine-tuning parts, and uses an image swap as
a control. Every number of the paper is listed in `paper/numbers.csv` with the file and field it was read from; see
[Reproducing the paper](#reproducing-the-paper).

## What is here, and what is not

| in the repository | not in the repository (and why) |
|---|---|
| the training and inference code (`src/`), every training configuration (`configs/`), the data splits and image-swap maps we created for the public benchmarks (`data/`), per-question predictions for the public benchmarks (`results/predictions/`), aggregated results for all five benchmarks (`results/aggregates/`), the generators of the tables and the figure, the number list and its verifier (`paper/`) | **anything derived from the credentialed datasets** HEAL-MedVQA / VinDr-CXR, MIMIC-CXR / MIMIC-CXR-JPG, Medical-Diff-VQA and Chest ImaGenome: no image, no question or answer text, no identifier, no per-question prediction. For those two benchmarks we ship only aggregates and the scripts that rebuild the subset, the split and the swap from the official files that a credentialed user downloads from PhysioNet (`scripts_credentialed/`). |
| | **model weights** (LoRA adapters, M3AE checkpoints), available on request; see [Weights](#weights). |
| | the HuggingFace copies of the public datasets (re-downloadable), the scheduler logs and process states of the runs, intermediate tables that no number of the paper uses, and side studies that are not in the paper. See `docs/SOURCES_NOT_INCLUDED.md`. |

## Repository map

```
README.md, LICENSE (code: MIT), LICENSE-DATA.md (our splits/maps/aggregates: CC BY 4.0; files derived from a dataset follow that dataset's licence), CITATION.cff
environment/        pinned requirements of the two environments (VLM fine-tuning; M3AE) and hardware/software notes
src/med_vlm/        data loading, LoRA fine-tuning and generation for Qwen2-VL-2B and SmolVLM2-2.2B (run_experiment.py)
src/harness/        everything else that ran for the paper: arrow building, grey image, image swap, classifiers, M3AE drivers,
                    bootstrap statistics, the analyses that produce results/aggregates
src/m3ae_patch/     the patch over the official M3AE code (frozen ViT; no-image text branch; Kvasir/HEAL/Medical-Diff-VQA tasks) and its tests
configs/            every training configuration: 36 YAMLs of the two VLM families (configs/vlm), the hparams.yaml of the 53 M3AE runs
                    behind the results (configs/m3ae_hparams), job lists of the classifier and M3AE batteries (configs/jobs)
data/splits/        Kvasir-VQA-x1 re-split by image; VQA-RAD re-split by image (the paper's) and by question (the control)
data/swaps/         image-swap maps (condition "A - A_swap") of Kvasir-VQA-x1, SLAKE and the two VQA-RAD splits
data/normalization/ answer normalisation rules, the ten closed-form question templates of Kvasir-VQA-x1 and its answer vocabulary, grey-image manifests
scripts_credentialed/  HEAL-MedVQA and Medical-Diff-VQA: scripts only (subset, split, PNG derivation, grey image, swap, TF-IDF) + a synthetic self-test
results/aggregates/ accuracies, gains and confidence intervals per run, family and benchmark (all five benchmarks); results/aggregates/m3ae_no_image/ = the
                    same analyses with the no-image M3AE text branch as A_text (the paper's version); results/predictions/ = per-question predictions (public benchmarks)
results/training_runs/  metadata of the training runs (time, peak memory, parameters)
paper/              generated tables (tables/), Fig. 1 (figs/), numbers.csv (every number of the paper -> source file and field), the generators and the
                    number verifier; the text of the paper (.tex) is not distributed
docs/manifest.json  sha256 of every file, and the origin map (which file of the original run tree each file came from)
tests/              unit tests of the Kvasir-VQA-x1 split code
```

## Glossary: identifiers used in the code and files → terms of the paper

| identifier | paper term |
|---|---|
| `C0` | intact (original) test image |
| `C1` | grey image (constant 384×384 RGB 128 image at test time) |
| `C2`, `C2v2` | image swap: every test image replaced by another test image (derangement, seed 42; `C2v2` = the swap redone for HEAL-MedVQA by radiograph and for Medical-Diff-VQA without same-patient donors) |
| `TXT`, `_LORA_TXT_`, `no_image` | text-decoder-only approach (trained and evaluated without image; `A_text`) |
| `QWEN2VL_LORA_<B>`, `SMOLVLM_LORA_FT_<B>` | frozen visual encoder (LoRA in the text decoder only; `A_frozen`) |
| `QWEN2VL_LORA_VT_<B>`, `SMOLVLM_LORA_<B>` | fine-tuned visual encoder (LoRA also in the encoder's attention; `A_ft`) |
| `M3AE_<B>`, `M3AE_SCRATCH_<B>` | M3AE, full fine-tuning (`A_ft`) |
| `M3AE_FROZEN_<B>` | M3AE with the ViT frozen (`A_frozen`) |
| `M3AE_SEMIMG_<B>`, "M3AE no-image" | M3AE text branch without the image path (`no_image=True`; the paper's M3AE `A_text`) |
| `M3AE_TXT_<B>` | earlier M3AE text-only variant trained with a constant grey image; kept for provenance, replaced in the paper by `M3AE_SEMIMG` |
| `TFIDF`, `tfidf_*`, `TF-IDF` | TF-IDF + logistic-regression question classifier |
| `BERT`, `BERT2`, `PubMedBERT` | PubMedBERT (BiomedBERT) question classifier, one per seed |
| `MAJ`, "majoritaria" | majority answer of the training data; "majoritaria por pergunta" = per-question majority |
| `own_text_only_baseline`, `c0_congelada`, `c0_adaptada` | `A_text`, `A_frozen`, `A_ft` |
| `external_baselines`, `baseline_*` | the external baselines (TF-IDF, PubMedBERT, majority, grey image) |
| `ganho_torre`, `ganho_adaptacao`, `soma` | `G_enc`, `G_ft`, `G` |
| `joint_ci` / `ic_conjunto` | 95 % CI of the mean over images and seeds (bootstrap, 2 000 replicates, seed 20260913) |
| `fechadas` | closed-form questions |
| `KVASIR` | Kvasir-VQA-x1, re-split by image (`data/splits/kvasir_vqa_x1_image_split`) |
| `SLAKE` | SLAKE, English questions, official split |
| `HEAL` | HEAL-MedVQA (closed-form test questions; credentialed) |
| `MIMIC` | Medical-Diff-VQA single-image subset (credentialed) |
| `VQARAD_REDIV` | VQA-RAD re-split by image (the paper's VQA-RAD) |
| `VQARAD_QLEVEL` | VQA-RAD re-split by question (the split-protocol control) |
| `VQARAD_RADORIG`, `VQARAD` (HF) | VQA-RAD official split / HuggingFace copy (supplementary runs, not in the paper) |
| `decomposicao_fechadas`, `joint_ci`, `m3ae_frozen_vit`, `heal_classifiers_train_only`, `text_analyses`, `analise_troca`, `cauldron_overlap`, `m3ae_no_image*` | names of the analyses (aggregate files in `results/aggregates/` and the scripts that produce them in `src/harness/v7_txt/`) |
| `article_runs` | the directory of the runs behind the paper in the original run tree (appears in paths) |
| `occlusion_study`, `v7_occlusion` | the companion occlusion study on the same models (a separate paper); appears in paths and comments of shared code |

Comments and docstrings inside `src/` are often in Portuguese; the logic is unchanged from the runs that produced the paper (see "Paths" for the mechanical edits).

## Reproducing the paper

Everything below runs on CPU in a few seconds from the shipped files (checked in a clean checkout with `environment/requirements-vlm.txt`):

```bash
python paper/make_tables.py    paper/numbers.csv paper/tables_regen      # Table 1 and Table 2 (byte-identical to paper/tables/*.tex)
python paper/make_tables_4p.py paper/numbers.csv paper/tables_regen/4p   # the 4-page variant's results table
python paper/make_fig1.py --aggregates results/aggregates --out paper/figs_regen   # Fig. 1: byte-identical to paper/figs/fig1_v11.{pdf,png}
python paper/verify_numbers.py <paper.tex> --numbers paper/numbers.csv              # checks a .tex of the paper against numbers.csv (exit 0 = every number has a source);
                                                                                   # the text of the paper is not distributed, so this needs the authors' .tex
python scripts_credentialed/selftest_synthetic.py                                  # the credentialed-data scripts on synthetic inputs
```

### `paper/numbers.csv`

One row per value used by the paper, with the columns `label` (what the value is), `value` (as read from the source), `used_in`
(`Table 1`, `Table 2`, `Table 2 (4-page version)`, or `text: <section>`), `source_file` (path in this repository, or
`not distributed: <file>` for the few values read from files that are not shipped, see `docs/SOURCES_NOT_INCLUDED.md`), `field`
(the path of the value inside the source file, e.g. `benchmarks.KVASIR.by_family.QWEN2VL.own_text_only_baseline.mean`, or `derived`
when the value was computed from the file), `source_sha256` (of the source file at the time the value was read) and `group`. The table
generators read this file by `label`; the verifier (`paper/verify_numbers.py`) checks every number printed in a `.tex` of the paper (not distributed) against its `value` column; with the submitted sources it exits 0 (278 distinct numbers in the 5-page version, 273 in the 4-page one, none without a source). A last group of
rows lists the numbers that do not come from a result file (grant numbers, hardware, parameter counts in millions, literature values).

### From the paper to the files

**Table 2** (`paper/tables/table2_main.tex`, generated by `paper/make_tables.py` from `paper/numbers.csv`): for benchmark `<B>` and family `<F>`

| column | numbers.csv label | aggregate file | produced by |
|---|---|---|---|
| `A_text` (mean (SD)), PubMedBERT, TF-IDF | `<B> / <F> (n=…): A_text / TF-IDF / PubMedBERT / majority (%)` (M3AE: suffix ` [M3AE no-image]`) | `results/aggregates/decomposicao_fechadas.json` (M3AE: `results/aggregates/m3ae_no_image/`) | `src/harness/v7_txt/decomposicao.py --fechadas` over `results/predictions/<B>/` |
| Grey (VLM) | `<B> / <F> (same set, n=…): … grey image (%) …` | same | same (`C1` predictions) |
| Grey (M3AE) | `<B> / M3AE frozen encoder: grey image at test time only …` | `results/aggregates/m3ae_frozen_vit.json` | `src/harness/v7_txt/m3ae_frozen_vit.py` |
| `G_enc`, `G_ft`, `G` (VLM) | `<B> / <F>: G_enc / G_ft / G (pp)` and `…: joint CI (images and seeds) of G_enc / G_ft / G (pp)` | `decomposicao_fechadas.json`, `joint_ci.json` | `decomposicao.py`, `src/harness/v7_txt/joint_ci.py` (joint bootstrap, `src/harness/stats/bootstrap.py`) |
| `G_enc`, `G_ft`, `G` (M3AE) | `<B> / M3AE frozen encoder: G_enc / G_ft / G (pp) [M3AE no-image]` and its `joint CI` line | `results/aggregates/m3ae_no_image/m3ae_frozen_vit.json` | `m3ae_frozen_vit.py` fed with the no-image text branch (`results/aggregates/m3ae_no_image/m3ae_no_image_summary.json` documents the swap) |
| `A - A_swap` | `<B> / <F>: A / A_text / A_swap (%); A - A_swap / A_swap - A_text (pp)` and `…: joint CI … of A - A_swap …` | `analise_troca_v2.json`, `joint_ci.json` | `src/harness/v7_txt/analise_troca.py` (`TROCA_V2=1`), `joint_ci.py` |

Table 2 rows for Kvasir-VQA-x1, SLAKE and VQA-RAD can be recomputed from `results/predictions/`; the HEAL-MedVQA and Medical-Diff-VQA rows come from the shipped aggregates (their per-question predictions are not distributed).

**Table 1** (`table1_data.tex`): the split sizes (`<benchmark> · VLM and M3AE · train/val/test · questions/images` rows of `numbers.csv`, read from the arrow files, and `data/splits/*/manifest.json`).

**Fig. 1** (`paper/figs/fig1_v11.pdf`): `paper/make_fig1.py`, from `decomposicao_fechadas.json`, `heal_classifiers_train_only.json` (HEAL-MedVQA classifiers, train-only protocol) and `m3ae_no_image/m3ae_frozen_vit.json`.

### Rebuilding the runs

1. **Data.** Kvasir-VQA-x1: `src/harness/prepro/build_kvasir_clean_split.py` (merges the official partitions, decomposes composite answers into atomic (question, answer) facts, selects the closed-form questions by the ten prefixes of `data/normalization/kvasir_closed_form_templates.json`, and splits 70/10/20 **by image**, stratified by finding label, `--seed 42 --ratios 0.7 0.1 0.2`; the resulting image→split map is `data/splits/kvasir_vqa_x1_image_split/kvasir_clean_split_map.csv`; `tests/prepro/` checks zero image overlap). SLAKE: English questions, official split, arrows in the M3AE arrow format. VQA-RAD: all 3 515 questions pooled and re-split 70/10/20 by image with `numpy.random.default_rng(20260913)` (`src/harness/v7_txt/make_vqarad_rediv.py`; question lists in `data/splits/vqarad_rediv/`) and, as a control, by question (`make_vqarad_qlevel.py`, `data/splits/vqarad_qlevel/`). Grey images: `src/harness/interventions/make_c1_blank_images.py`. Image swap: `src/harness/interventions/make_c2_shuffled_pairs.py` (seed 42), maps in `data/swaps/`. HEAL-MedVQA and Medical-Diff-VQA: `scripts_credentialed/README.md`.
2. **Fine-tuning the VLMs.** `python -m med_vlm.run_experiment --config configs/vlm/<B>/<F>_<approach>.yaml --seed {0,1,2}` (LoRA rank 16, α 32, lr 2e-4, 3 epochs, effective batch 16 (batch 4 × accumulation 4) or 32 (4 × 8); the YAML carries the recipe; the text-decoder-only YAMLs set `data.no_image: true`).
3. **Fine-tuning M3AE.** Clone https://github.com/zhjohnchan/M3AE at commit `14da377255280eed547e451f368cf71a43f3ca1c`, download its pre-trained `m3ae.ckpt` (sha256 `469cf69b…`), apply `src/m3ae_patch/m3ae_article2.patch` and copy `src/m3ae_patch/new_files/` into the clone; the drivers are `src/harness/v7_txt/m3ae_txt_train.py` (grey text-only), `m3ae_frozen_train.py` (`freeze_vision_encoder=True`), `m3ae_semimg_train.py` (`no_image=True`); every run's exact hyper-parameters are in `configs/m3ae_hparams/`. The no-image flag is tested by `src/m3ae_patch/tests/test_no_image.py` (CPU; logits identical for grey, black, random, real and no image; ViT never called; results in `tests/test_results_no_image.json` and, rerun from this repository, `tests/test_results_no_image_from_package.json`).
4. **Predictions.** VLMs: `python -m harness.extract_predictions_circle --model QWEN2VL|SMOLVLM --peft LORA|LORA_VT|LORA_FT|LORA_TXT --dataset <B> --condition C0|C1|C2 --condition-arrow <arrow> --adapter-dir <adapter> --output <json> [--no-image]`. M3AE: `src/harness/v7_txt/m3ae_infer_v7.py` and `m3ae_infer_semimg.py`. Each output JSON holds `qid`, `predicted`, `gold`, `correct` per question and a provenance block (checkpoint sha256, input arrow sha256, versions, command).
5. **Baselines.** TF-IDF: `src/harness/baselines/question_only.py`; PubMedBERT: `src/harness/v7_txt/bert_question_only.py` (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`, max length 64, batch 32, AdamW 2e-5, weight decay 0.01, linear warm-up 10 %, up to 10 epochs, epoch chosen by validation accuracy; jobs in `configs/jobs/pubmedbert_jobs.json`); majority and per-question majority: `src/harness/v7_txt/dados_texto.py`.
6. **Analyses.** `ARTICLE2_RUNS=1 python -m harness.v7_txt.decomposicao --fechadas`; `joint_ci.py` (joint CIs); `m3ae_frozen_vit.py` (M3AE frozen ViT); `m3ae_no_image_analysis.py` (no-image M3AE); `analise_troca.py` (swap); `text_analyses.py`, `analises_rodada2.py heal_classifiers_train_only` (baselines, HEAL-MedVQA train-only protocol); `cauldron_pairing.py`, `cauldron_accuracy.py` (The Cauldron overlap, VQA-RAD); then the tables and the figure as above. `src/harness/v7_txt/txt_floors.py` maps every (benchmark, family, seed) to its prediction files. `text_analyses.py` and `m3ae_no_image_summary.py` also read two intermediate tables that are not distributed (`docs/SOURCES_NOT_INCLUDED.md`).

## Paths

The scripts under `src/` and `scripts_credentialed/` are the ones that ran, with mechanical edits only: every absolute path of the original run tree was replaced by `{_MEDVQA_ROOT}` (`os.environ["MEDVQA_ROOT"]`) in code and by the placeholder `${MEDVQA_ROOT}` in data files; disk-space checks of the queue utilities read `DATA_ROOT`; internal directory and identifier names were replaced by the descriptive English names of the glossary, consistently in code and data (so the scripts still read the shipped files). In the provenance blocks of the result files, paths of the original tree were reduced to the file name, or to the path of the same file in this repository when it is shipped. `docs/manifest.json` (`origin`) tells which original file each shipped file is, with the sha256 of the original. Rerunning the full pipeline requires the arrows and weights that are not distributed.

## Environment

Two Python environments were used (exact `pip freeze` in `environment/`):

- **VLM fine-tuning, predictions and analyses** (`requirements-vlm.txt`): Python 3.13, torch 2.13.0+cu130, transformers 5.15.1, peft 0.20.0, datasets 4.8.5, accelerate 0.34.2, matplotlib 3.11.1 (the figure is byte-reproducible with this version).
- **M3AE** (`requirements-m3ae.txt`): Python 3.8, torch 1.9.0+cu111, pytorch-lightning 1.3.2, transformers 4.6.0 (the versions the official M3AE code requires).

Both families are loaded in bfloat16 for the vision encoder and the language model alike (`src/med_vlm/models/loader.py`): with `dtype="auto"` SmolVLM2 would load its vision encoder in float16 and its language model in bfloat16. Qwen2-VL's image processor is capped at 512×512 pixels (`_QWEN2VL_MAX_PIXELS`). Hardware: every run used one NVIDIA RTX 5000 Ada Generation (32 GB); wall-clock times and per-process peak memory are in `results/training_runs/`.

## Weights

Not in the repository (size), available on request from the authors: the LoRA adapters of the 108 VLM fine-tuning runs (2 families × 3 approaches × 6 splits × 3 seeds; ≈43 MB each) and the M3AE checkpoints behind the 53 configurations in `configs/m3ae_hparams/` (1.39 GB each). The sha256 of every checkpoint used by a prediction is in that prediction's provenance block.

## Licences

| component | licence | source checked (30 Oct 2026) | note |
|---|---|---|---|
| our code | MIT (`LICENSE`) | — | — |
| our splits, swap maps, normalisation tables, aggregates, `numbers.csv` | CC BY 4.0 (`LICENSE-DATA.md`) | — | files derived from a dataset follow that dataset's licence instead (next rows) |
| Kvasir-VQA-x1 (`SimulaMet/Kvasir-VQA-x1`) | **CC BY-NC 4.0** | https://huggingface.co/datasets/SimulaMet/Kvasir-VQA-x1 — "Released under CC BY-NC 4.0 – for academic and non-commercial use. Please cite appropriately." | `results/predictions/KVASIR/*` contain the dataset's questions' identifiers and gold answers, and `data/splits`, `data/swaps`, `data/normalization` files for Kvasir are derived from it: **non-commercial use only, attribution to the dataset; not MIT** |
| SLAKE | **CC BY 4.0** (as tagged on the copy we used) | https://huggingface.co/datasets/BoKelvin/SLAKE (tag `cc-by-4.0`); the official site https://www.med-vqa.com/slake/ states no licence; https://huggingface.co/datasets/mdwiratathya/SLAKE-vqa-english (the English copy) declares none | `results/predictions/SLAKE/*` and `data/swaps/image_swap_SLAKE.json` follow it; attribution |
| VQA-RAD (Lau et al., 2018) | **CC0 1.0 Universal** | https://osf.io/89kps/ (OSF project licence); https://huggingface.co/datasets/flaviagiammarino/vqa-rad (tag `cc0-1.0`) | `results/predictions/VQARAD_*`, `data/splits/vqarad_*`, `data/swaps/image_swap_VQARAD_*` |
| Qwen2-VL-2B-Instruct | Apache 2.0 | https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct | fine-tuned with LoRA; adapters not redistributed here |
| SmolVLM2-2.2B-Instruct | Apache 2.0 | https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct | idem |
| PubMedBERT (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`) | MIT | https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext | question classifier |
| M3AE (code and pre-trained checkpoint) | no licence declared in the repository | https://github.com/zhjohnchan/M3AE | we distribute neither the code nor the checkpoint: `src/m3ae_patch/m3ae_article2.patch` contains only our modifications plus the minimal unchanged context lines that `git apply` needs to locate them in commit `14da377`, and `src/m3ae_patch/new_files/` only files we wrote |
| HEAL-MedVQA / VinDr-CXR, Medical-Diff-VQA / MIMIC-CXR-JPG / Chest ImaGenome | PhysioNet credentialed health data | https://physionet.org | nothing derived from them is distributed except aggregates and scripts |

## Citation

See `CITATION.cff`. Please cite the paper (submitted to IEEE ISBI 2027) and the datasets and models listed above.
