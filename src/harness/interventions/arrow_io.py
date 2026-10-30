"""Utilitários compartilhados para ler/escrever arquivos .arrow no mesmo
formato usado por prepro/make_arrow.py::make_arrow_vqa do repositório LaPA/M3AE.

Schema (ordem das colunas, confirmada lendo o código-fonte de make_arrow_vqa):
    image          bytes      - conteúdo binário do arquivo de imagem (jpg)
    questions      list[str]
    answers        list[list[str]]
    answer_labels  list[list[int]]
    answer_scores  list[list[float]]
    image_id       str        - caminho do arquivo de imagem (usado como chave)
    question_id    list[int]
    answer_type    list[int]  - 0=closed, 1=open
    split          str        - "train" | "val" | "test"
"""
from __future__ import annotations

import os

import pyarrow as pa
import pandas as pd

ARROW_COLUMNS = [
    "image",
    "questions",
    "answers",
    "answer_labels",
    "answer_scores",
    "image_id",
    "question_id",
    "answer_type",
    "split",
]


def read_arrow(path: str) -> pd.DataFrame:
    """Lê um arquivo .arrow escrito no formato do make_arrow_vqa e retorna um DataFrame."""
    with pa.OSFile(path, "rb") as source:
        table = pa.ipc.open_file(source).read_all()
    df = table.to_pandas()
    missing = [c for c in ARROW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Arquivo {path} não tem as colunas esperadas: {missing}")
    return df


def write_arrow(df: pd.DataFrame, path: str) -> None:
    """Escreve um DataFrame em .arrow no mesmo formato usado pelo make_arrow_vqa
    (mesma ordem de colunas, mesmo RecordBatchFileWriter).

    Cria o diretório de `path` se não existir (achado real em execução,
    Colab, 21/08/2026: `data/finetune_arrows/` é gitignored e nunca
    existiu no clone fresco do harness — pa.OSFile levanta
    `FileNotFoundError` de baixo nível, sem mensagem clara, depois de
    minutos de trabalho já feito, ex. baixar ~4.000 imagens do Kvasir)."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    df = df[ARROW_COLUMNS]
    table = pa.Table.from_pandas(df, preserve_index=False)
    with pa.OSFile(path, "wb") as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)


def count_questions(df: pd.DataFrame) -> int:
    """Conta o número real de perguntas avaliadas num DataFrame de arrow.

    Cada LINHA do arrow é uma IMAGEM, que carrega uma LISTA de perguntas na
    coluna 'questions' — contar linhas (len(df)) conta imagens, não
    perguntas. Foi exatamente esse bug que produziu um IC95%(bootstrap)
    errado no primeiro resultado do M3AE (n=96 imagens em vez de n=1061
    perguntas) — ver HANDOFF_P0_Execucao_Harness.md §3.0."""
    return int(sum(len(qs) for qs in df["questions"]))
