"""Divide o arrow de treino do Kvasir (harness/prepro/prepare_kvasir_arrow.py
--split train) em train/val — o `SimulaMet/Kvasir-VQA-x1` só publica splits
`train`/`test` (achado confirmado em `docs/superpowers/plans/
2026-08-21-kvasir-vqa-integration.md`), e o M3AE precisa dos três.

Achado real (não suposição), confirmado lendo o código-fonte do repositório
`zhjohnchan/M3AE`, commit `14da377255280eed547e451f368cf71a43f3ca1c` (pinado
por `environment/colab_setup_m3ae.sh`), arquivo
`m3ae/datamodules/vqa_vqa_rad_datamodule.py`: `VQAVQARADDataModule.setup()`
constrói `self.answer2id`/`self.id2answer` (usados por
`harness/extract_predictions_m3ae.py` pra decodificar a predição) lendo as
colunas `answers`/`answer_labels` de `self.train_dataset` + `self.val_dataset`
— NÃO de `test_dataset`. Sem um `vqa_kvasir_val.arrow`, o `setup()` do
`VQAKVASIRDataModule` equivalente quebraria por falta do split.

Diferente de `harness/prepro/prepare_vqa_rad_splits.py` (que divide o JSON
bruto do VQA-RAD *por pergunta*, antes do arrow existir): aqui o arrow do
Kvasir já agrupa todas as perguntas de uma imagem numa única linha (schema
`harness/interventions/arrow_io.py`) — dividir por pergunta exigiria duplicar
a imagem entre train e val. Dividimos por LINHA (imagem) em vez disso: mesmo
espírito (determinístico, seed fixa, documentado — não é "a divisão certa"
clinicamente, só uma escolha estável).

Uso:
    python -m harness.prepro.split_kvasir_train_val \
        --input data/finetune_arrows/vqa_kvasir_train_full.arrow \
        --train-output data/finetune_arrows/vqa_kvasir_train.arrow \
        --val-output data/finetune_arrows/vqa_kvasir_val.arrow \
        --val-fraction 0.1 --seed 42
"""
from __future__ import annotations

import argparse
import random

import pandas as pd

from harness.interventions.arrow_io import read_arrow, write_arrow


def split_train_val_df(df: pd.DataFrame, val_fraction: float = 0.1,
                        seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide `df` (uma linha por imagem) em (train_df, val_df), embaralhado
    deterministicamente por `seed`. `val_fraction` do total (arredondado) vai
    para val; o resto para train. União dos dois é exatamente `df`, sem
    sobreposição. Reescreve a coluna `split` ('train'/'val') em cada parte."""
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction={val_fraction} deve estar em (0, 1)")

    order = sorted(range(len(df)), key=lambda i: df.iloc[i]["image_id"])  # ordem determinística antes de embaralhar
    rng = random.Random(seed)
    shuffled = order.copy()
    rng.shuffle(shuffled)

    n_val = round(len(shuffled) * val_fraction)
    val_positions = shuffled[:n_val]
    train_positions = shuffled[n_val:]

    val_df = df.iloc[val_positions].reset_index(drop=True).copy()
    train_df = df.iloc[train_positions].reset_index(drop=True).copy()
    val_df["split"] = "val"
    train_df["split"] = "train"
    return train_df, val_df


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Arrow de treino completo (prepare_kvasir_arrow.py --split train)")
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--val-output", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    df = read_arrow(args.input)
    train_df, val_df = split_train_val_df(df, val_fraction=args.val_fraction, seed=args.seed)
    write_arrow(train_df, args.train_output)
    write_arrow(val_df, args.val_output)
    n_train_q = int(sum(len(qs) for qs in train_df["questions"]))
    n_val_q = int(sum(len(qs) for qs in val_df["questions"]))
    print(f"train: {len(train_df)} imagens, {n_train_q} perguntas -> {args.train_output}")
    print(f"val:   {len(val_df)} imagens, {n_val_q} perguntas -> {args.val_output}")
