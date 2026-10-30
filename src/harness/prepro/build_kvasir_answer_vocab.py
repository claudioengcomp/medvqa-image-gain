"""Constrói o vocabulário fechado de respostas do Kvasir (mesma lógica
que prepro_vqa_vqa_rad() usa pro VQA-RAD: um índice por resposta única,
ordenado por frequência decrescente), e re-escreve o campo
answer_labels dos arrows de train+val+test com os índices reais.

`--val-arrow` é obrigatório: o vocabulário real do M3AE (`answer2id`,
construído em `VQAVQARADDataModule.setup()`, confirmado lendo o código-fonte
de `zhjohnchan/M3AE` commit `14da377255280eed547e451f368cf71a43f3ca1c`) é
montado a partir de train+val, não train+test — sem o split de val (ver
`harness/prepro/split_kvasir_train_val.py`), o `answer_labels` do val
ficaria com o placeholder `[[0]]` de `prepare_kvasir_arrow.py`, quebrando o
treino. Incluímos test também (não só train+val) pra cobrir qualquer
resposta que só apareça lá — mais amplo que o mínimo exigido pelo M3AE em
tempo de execução, sem custo (mesmo espírito documentado pro VQA-RAD:
"vocabulário próprio, construído... a partir de train+val+test").

Uso:
    python -m harness.prepro.build_kvasir_answer_vocab \
        --train-arrow data/finetune_arrows/vqa_kvasir_train.arrow \
        --val-arrow data/finetune_arrows/vqa_kvasir_val.arrow \
        --test-arrow data/finetune_arrows/vqa_kvasir_test.arrow \
        --vocab-output data/finetune_arrows/kvasir_answer_vocab.json
"""
from __future__ import annotations

import argparse
import collections
import json

from harness.interventions.arrow_io import read_arrow, write_arrow


def build_vocab(answer_lists: list[list[str]]) -> dict[str, int]:
    """answer_lists: lista de [resposta_única] por pergunta (1 ref cada,
    ver build_arrow_dataframe). Retorna {resposta: índice}, ordenado por
    frequência decrescente (desempate por ordem alfabética, determinístico)."""
    counter = collections.Counter(a[0] for a in answer_lists)
    ordered = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    return {answer: i for i, (answer, _count) in enumerate(ordered)}


def apply_vocab(answer_lists: list[list[str]], vocab: dict[str, int]) -> list[list[int]]:
    return [[vocab.get(a[0], -1)] for a in answer_lists]


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-arrow", required=True)
    parser.add_argument("--val-arrow", required=True)
    parser.add_argument("--test-arrow", required=True)
    parser.add_argument("--vocab-output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    df_train = read_arrow(args.train_arrow)
    df_val = read_arrow(args.val_arrow)
    df_test = read_arrow(args.test_arrow)

    all_answers = []
    for df in (df_train, df_val, df_test):
        for row_answers in df["answers"]:
            all_answers.extend(row_answers)
    vocab = build_vocab(all_answers)

    with open(args.vocab_output, "w", encoding="utf-8") as fp:
        json.dump(vocab, fp, indent=2, ensure_ascii=False)

    for path, df in [(args.train_arrow, df_train), (args.val_arrow, df_val), (args.test_arrow, df_test)]:
        df["answer_labels"] = [apply_vocab(row_answers, vocab) for row_answers in df["answers"]]
        write_arrow(df, path)

    print(f"vocabulário: {len(vocab)} respostas únicas -> {args.vocab_output}")
    print(f"arrows re-escritos com answer_labels reais: {args.train_arrow}, {args.val_arrow}, {args.test_arrow}")
