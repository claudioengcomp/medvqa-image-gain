"""Divide o trainset.json bruto do VQA-RAD (formato MEVF/data_RAD, sem
valset.json próprio — confirmado nesta sessão baixando o pacote oficial de
`https://huggingface.co/datasets/aiozai/MEVF/resolve/main/data_RAD.zip`) em
train/val, já que `prepro_vqa_vqa_rad()` do M3AE espera os três arquivos
(trainset.json/valset.json/testset.json).

Divisão por PERGUNTA (não por imagem), com seed fixa — consistente com o
próprio split oficial train/test do VQA-RAD, que já tem sobreposição de
imagem entre train e test (achado confirmado nesta sessão: as 203 imagens
de `testset.json` aparecem todas também em `trainset.json` — é assim que o
benchmark oficial funciona; não é um vazamento introduzido por nós, é uma
limitação conhecida do dataset a registrar no artigo, não a "corrigir").

Uso:
    python -m harness.prepro.prepare_vqa_rad_splits \
        --trainset data_RAD/trainset.json \
        --train-output data_RAD/trainset_split.json \
        --val-output data_RAD/valset.json \
        --val-fraction 0.1 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random


def split_train_val(records: list[dict], val_fraction: float = 0.1,
                     seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Divide `records` (cada um com uma chave 'qid' única) em
    (train, val), embaralhado deterministicamente por `seed`. `val_fraction`
    do total (arredondado) vai para val; o resto para train. União dos dois
    é exatamente `records`, sem sobreposição."""
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction={val_fraction} deve estar em (0, 1)")
    records_sorted = sorted(records, key=lambda r: r["qid"])  # ordem determinística antes de embaralhar
    rng = random.Random(seed)
    shuffled = records_sorted.copy()
    rng.shuffle(shuffled)
    n_val = round(len(shuffled) * val_fraction)
    val_records = shuffled[:n_val]
    train_records = shuffled[n_val:]
    return train_records, val_records


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainset", required=True, help="Caminho do trainset.json bruto (3064 registros)")
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--val-output", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    with open(args.trainset, "r", encoding="utf-8") as fp:
        records = json.load(fp)
    train_records, val_records = split_train_val(records, val_fraction=args.val_fraction, seed=args.seed)
    with open(args.train_output, "w", encoding="utf-8") as fp:
        json.dump(train_records, fp, indent=2, ensure_ascii=False)
    with open(args.val_output, "w", encoding="utf-8") as fp:
        json.dump(val_records, fp, indent=2, ensure_ascii=False)
    print(f"train: {len(train_records)} -> {args.train_output}")
    print(f"val:   {len(val_records)} -> {args.val_output}")
