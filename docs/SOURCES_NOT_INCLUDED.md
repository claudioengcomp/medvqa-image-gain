# Sources that are not in this repository

`paper/numbers.csv` gives, for every number printed in the paper, the file it was read from. A few rows carry
`not distributed: <file name>` in `source_file`; those files, and other material of the original run tree, are not
distributed for the reasons below. Where the original directory names were internal they appear here with the same
descriptive names used throughout the repository (`${MEDVQA_ROOT}/v7_txt/article_runs/...` is the directory of the runs
behind the paper).

| category | examples | why |
|---|---|---|
| per-question predictions of the credentialed benchmarks | `v7_txt/results/*HEAL*_TXT.json`, `v7_mimic/results/*MIMIC*.json`, `v7_txt/article_runs/m3ae_frozen/results/M3AE_FROZEN_{HEAL,MIMIC}_*`, `v7_txt/article_runs/m3ae_semimg/results/M3AE_SEMIMG_{HEAL,MIMIC}_*` | they carry question identifiers and gold answers of HEAL-MedVQA (VinDr-CXR) and Medical-Diff-VQA (MIMIC-CXR); only aggregates are shipped |
| arrow files (images + questions) of every benchmark | `doutorado/M3AE/data/finetune_arrows/*.arrow`, `v7_txt/article_runs/data/*/*.arrow`, `v7_txt/m3ae_arrows/*_gray.arrow`, `v6_medida/results/arrows/stable_vqa_heal_test.arrow`, the HuggingFace cache arrows of SLAKE | images; the public ones are rebuilt from the datasets with `src/harness/prepro` and `src/harness/v7_txt/make_vqarad_*.py`; manifests with their sha256 are in `data/`. The split sizes printed in the paper (Table 1 and the setup paragraph) were read from these arrows; `numbers.csv` marks them `not distributed` |
| inventories and text tables of HEAL-MedVQA / Medical-Diff-VQA | `inventory_heal_test.json`, `v7_mimic/data/items_*.parquet`, `v7_txt/article_runs/data/texto/{HEAL,MIMIC}.json`, swap maps `troca2/{heal,mimic}/mapa_troca.json` | question texts and identifiers of credentialed data |
| model weights | `v7_txt/article_runs/models/*/adapter_model.safetensors`, `doutorado/M3AE/checkpoints/*.ckpt`, `v7_txt/article_runs/m3ae_*/ckpt/*` | size; available on request (README, "Weights") |
| scheduler states and training logs | `*/logs/infer_state_*.json`, `train_state_*.json`, `*.log`, `*_cost.json` of runs outside the paper | operational records (process ids, GPU slots); the cost figures used by the paper are in `results/training_runs/` |
| The Cauldron pairing file | `v7_txt/article_runs/cauldron_vqarad/pareamento.json` | contains perceptual hashes and question texts matched against The Cauldron; the counts are in `results/aggregates/cauldron_overlap.json` |
| intermediate tables of the analysis chain | the per-benchmark reordering table (`reordenacao_fechadas.json`), the per-run count of 95 % CIs (`ic_contagem.json`), the list of no-image M3AE runs with checkpoint sha256, the M3AE initialisation check, the hyper-parameter tables of the text-only M3AE | not used by any number, table or figure of the final paper. `src/harness/v7_txt/text_analyses.py` and `m3ae_no_image_summary.py` read the first two as inputs for analyses that are not in the paper; the checkpoint sha256 of every run used by a prediction is in that prediction's provenance block |
| side studies not in the paper | zero-shot batteries (`v7_zeroshot/`), the black-image M3AE variant, the Qwen2.5-VL-7B study, the official-split VQA-RAD runs (`_RADORIG`) and the HuggingFace VQA-RAD runs, the earlier Qwen2-VL/SmolVLM2 DoRA runs | outside the paper |
| the internal numbers sheet | the Markdown sheet from which `paper/numbers.csv` was derived (one line per number, Portuguese labels, 2 041 lines) | internal working document; the rows the paper uses are in `paper/numbers.csv`, with the source file and field of each value |
| M3AE result directories (`hparams.yaml`) of runs of the credentialed benchmarks | `doutorado/M3AE/result/task_finetune_vqa_heal-*`, `v7_mimic/m3ae*/result/*` | shipped (`configs/m3ae_hparams/`): they contain only hyper-parameters and paths |
