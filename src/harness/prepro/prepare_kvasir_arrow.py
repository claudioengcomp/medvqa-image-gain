"""Constrói o .arrow de avaliação do Kvasir-VQA no schema M3AE
(harness/interventions/arrow_io.py), a partir de SimulaMet/Kvasir-VQA-x1.

Decompõe cada linha do x1 (pergunta composta, resposta em linguagem
natural) de volta nas perguntas atômicas que a geraram, via o campo
`original` (JSON com a lista {"q":..., "a":...} das perguntas-molde do
Kvasir-VQA base que foram combinadas). Isso é necessário porque:
- M3AE precisa de vocabulário de resposta fechado e curto pra
  classificação — as respostas compostas do x1 são frases longas.
- A avaliação sob condição (C0-C4, C5) reaproveita esse mesmo arrow
  pros modelos generativos via harness/circle_data_bridge.py — eles
  recebem a pergunta atômica igual aos outros, sem desvantagem.

Também captura, separadamente, a resposta da pergunta-molde de
categoria de achado ("Are there any abnormalities in the image? Check
all that are present.") como `pathology_label` por imagem — usada pela
condição C5 (harness/interventions/make_c5_matched_pathology_swap.py)
pra restringir a troca por categoria diferente. Ver docs/superpowers/
plans/2026-08-21-kvasir-targeted-pathology-swap-condition.md.

O TREINO dos modelos generativos (LoRA) NÃO usa este arrow — usa
Kvasir-VQA-x1 direto via med_vlm.data.loader (perguntas compostas),
porque med_vlm já está configurado pra esse formato nativamente.

Uso:
    python -m harness.prepro.prepare_kvasir_arrow \
        --split test \
        --output data/finetune_arrows/vqa_kvasir_test.arrow

Achado real em execução (Colab, 21-22/08/2026), não coberto pelos 17
padrões mapeados na sessão de planejamento (que só olhou o split de
TESTE): o split de TREINO tem 2 padrões extras — "Does this image contain
any finding?" (2263/6212 imagens do train, 36%, tinham SÓ essa pergunta e
estavam sendo descartadas por inteiro; resposta 100% "no" nas 2263+237
ocorrências medidas em train+test — pergunta degenerada, sem variância,
mas legitimamente closed) e "What color is the anatomical landmark?..."
(vocabulário pequeno, 22 combinações no train — mesmo padrão categórico já
tratado pro "what color is the abnormality" irmão). Os dois aparecem
também no split de teste (237 e 55 ocorrências) — o arrow de teste
precisou ser regenerado depois deste fix pra ficar consistente com o de
treino.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from typing import Callable

import pandas as pd

from harness.interventions.arrow_io import write_arrow

# Cada padrão de pergunta-molde do Kvasir-VQA base mapeado pro tipo
# closed/open. Comparação por prefixo, case-insensitive. 17 confirmados via
# inspeção do split de TESTE em 21/08/2026 + 2 achados só no split de
# TREINO em execução real (21-22/08/2026, ver docstring do módulo:
# "does this image contain any finding" recupera 36% das imagens do train;
# "what color is the anatomical landmark" tem vocabulário pequeno,
# verificado antes de incluir, mesmo padrão do "abnormality" abaixo).
_CLOSED_PATTERNS = [
    "is there", "are there", "have all polyps", "how many",
    "what type of", "what is the size", "where in the image",
    "does this image contain any finding",
    "what color is the anatomical landmark",
]

_PATHOLOGY_QUESTION = "are there any abnormalities in the image? check all that are present."


def classify_question(question_text: str) -> str:
    """Classifica uma pergunta atômica do Kvasir como 'closed' ou 'open'.
    Levanta ValueError se o texto não bater com nenhum dos padrões
    conhecidos — sinal de que o Kvasir-VQA-x1 mudou ou foi mal decomposto,
    não deve ser silenciado."""
    q = question_text.strip().lower()
    for pat in _CLOSED_PATTERNS:
        if q.startswith(pat):
            return "closed"
    if q.startswith("what color is the abnormality"):
        return "closed"  # vocabulário curto na prática, verificado antes de incluir
    raise ValueError(f"Pergunta atômica não reconhecida entre os padrões conhecidos: {question_text!r}")


def decompose_original_field(original_json: str) -> list[dict]:
    """Faz parse do campo `original` (string JSON) de uma linha do
    Kvasir-VQA-x1, retornando a lista de {"question":..., "answer":...}
    atômicas. Retorna [] se o campo estiver vazio ou malformado (não
    levanta exceção — dados de terceiros, tratar como ausência de dado)."""
    if not original_json:
        return []
    try:
        items = json.loads(original_json)
    except (json.JSONDecodeError, TypeError):
        return []
    return [{"question": it["q"], "answer": it["a"]} for it in items if "q" in it and "a" in it]


def build_image_records(rows: list[dict]) -> list[dict]:
    """`rows` é a lista de linhas cruas do Kvasir-VQA-x1 (dict com
    img_id/image/original). Decompõe cada linha e agrupa por img_id,
    deduplicando perguntas atômicas idênticas que aparecem repetidas em
    múltiplas linhas compostas da mesma imagem, e capturando
    separadamente a pergunta de categoria de achado.

    Retorna uma lista de dicts, um por imagem:
        {"img_id": str, "image_url": str,
         "questions": [str, ...], "answers": [str, ...],
         "pathology_label": str | None}
    """
    by_img: dict[str, dict] = {}
    for row in rows:
        img_id = row["img_id"]
        if img_id not in by_img:
            by_img[img_id] = {
                "img_id": img_id,
                "image_url": row["image"],
                "seen_questions": {},  # question_text -> answer (dedup)
                "pathology_label": None,
            }
        for atom in decompose_original_field(row["original"]):
            q, a = atom["question"], atom["answer"]
            if q.strip().lower() == _PATHOLOGY_QUESTION:
                by_img[img_id]["pathology_label"] = a.strip().lower()
            try:
                if classify_question(q) != "closed":
                    continue
            except ValueError:
                continue  # padrão desconhecido — pular, não travar o build inteiro
            by_img[img_id]["seen_questions"][q] = a

    records = []
    for img_id, data in by_img.items():
        if not data["seen_questions"]:
            continue  # imagem sem nenhuma pergunta closed reconhecida — fora do escopo
        records.append({
            "img_id": img_id,
            "image_url": data["image_url"],
            "questions": list(data["seen_questions"].keys()),
            "answers": list(data["seen_questions"].values()),
            "pathology_label": data["pathology_label"],
        })
    return records


def _download_image_bytes(url: str, max_retries: int = 5, backoff_base_s: float = 1.0) -> bytes:
    """Baixa os bytes de `url`, com retentativa exponencial. Achado real em
    execução (21/08/2026, Colab): `ConnectionResetError`/`URLError`
    transitório aconteceu no meio de ~4.000 downloads sequenciais via
    urllib — sem retentativa, uma única falha de rede derrubava o script
    inteiro depois de já ter baixado a maior parte com sucesso, sem nenhum
    progresso salvo (`build_arrow_dataframe` só escreve o arrow no final).
    5 tentativas, backoff 1s/2s/4s/8s/16s — generoso o bastante pra
    absorver instabilidade transitória do host de imagens sem mascarar uma
    falha real (URL quebrada, 404, etc., que `URLError` também cobre)."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return resp.read()
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(backoff_base_s * (2 ** attempt))
    assert last_exc is not None
    raise last_exc


_PLACEHOLDER_JPEG_BYTES: bytes | None = None


def _placeholder_image_bytes(url: str) -> bytes:
    """Retorna sempre os mesmos bytes de um JPEG 1x1 preto, ignorando `url`
    — usado no lugar de `_download_image_bytes` quando os pixels nunca vão
    ser decodificados de verdade (achado real, 22/08/2026: `train`/`val` só
    entram no pipeline do M3AE pra montar o vocabulário de resposta —
    `VQAKVASIRDataModule.setup()`, confirmado lendo m3ae/datamodules/
    base_datamodule.py, só lê as colunas `answers`/`answer_labels`, nunca
    chama `train_dataloader()`/`val_dataloader()` — que seriam os únicos
    pontos que decodificam a coluna `image`). Poupa baixar milhares de
    imagens de treino inteiras quando só se quer reconstruir o vocabulário
    pra rodar inferência a partir de um checkpoint já pronto (ex.: retomar
    depois de uma queda de sessão do Colab, sem re-treinar)."""
    global _PLACEHOLDER_JPEG_BYTES
    if _PLACEHOLDER_JPEG_BYTES is None:
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (1, 1), (0, 0, 0)).save(buf, format="JPEG")
        _PLACEHOLDER_JPEG_BYTES = buf.getvalue()
    return _PLACEHOLDER_JPEG_BYTES


def build_arrow_dataframe(records: list[dict], split: str, qid_offset: int = 0,
                           download_fn: Callable[[str], bytes] = _download_image_bytes) -> pd.DataFrame:
    """Converte `records` (saída de build_image_records) num DataFrame no
    schema ARROW_COLUMNS de arrow_io.py. question_id é gerado
    deterministicamente por ordem de iteração + qid_offset (documentar o
    offset usado por split pra não colidir entre train/test).
    `download_fn` é injetável pra testes não fazerem download real."""
    rows: dict[str, list] = {col: [] for col in [
        "image", "questions", "answers", "answer_labels", "answer_scores",
        "image_id", "question_id", "answer_type", "split",
    ]}
    next_qid = qid_offset
    for i, rec in enumerate(records):
        if i > 0 and i % 200 == 0:
            print(f"  ... {i}/{len(records)} imagens baixadas")
        image_bytes = download_fn(rec["image_url"])
        n_q = len(rec["questions"])
        qids = list(range(next_qid, next_qid + n_q))
        next_qid += n_q

        rows["image"].append(image_bytes)
        rows["questions"].append(rec["questions"])
        rows["answers"].append([[a] for a in rec["answers"]])  # 1 resposta de referência por pergunta
        rows["answer_labels"].append([[0]] * n_q)  # placeholder — vocabulário real em build_kvasir_answer_vocab.py
        rows["answer_scores"].append([[1.0]] * n_q)
        rows["image_id"].append(f"kvasir/{rec['img_id']}")
        rows["question_id"].append(qids)
        rows["answer_type"].append([0] * n_q)  # 0 = closed, todas as perguntas aqui já são closed por construção
        rows["split"].append(split)
    return pd.DataFrame(rows)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=["train", "test"])
    parser.add_argument("--qid-offset", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-images", action="store_true",
                         help="Não baixa os pixels de verdade — escreve um JPEG 1x1 placeholder no "
                              "lugar de cada imagem. Só use pra --split train quando o objetivo é "
                              "reconstruir o vocabulário de resposta pra rodar inferência a partir de "
                              "um checkpoint já pronto (sem treinar de verdade) — ver "
                              "_placeholder_image_bytes(). NUNCA use pro --split test, cujas imagens "
                              "são de fato avaliadas.")
    return parser.parse_args()


if __name__ == "__main__":
    from datasets import load_dataset

    args = _parse_args()
    ds = load_dataset("SimulaMet/Kvasir-VQA-x1", split=args.split)
    records = build_image_records(list(ds))
    download_fn = _placeholder_image_bytes if args.skip_images else _download_image_bytes
    df = build_arrow_dataframe(records, split="test" if args.split == "test" else "train",
                                qid_offset=args.qid_offset, download_fn=download_fn)
    write_arrow(df, args.output)
    n_questions = sum(len(q) for q in df["questions"])
    print(f"{len(df)} imagens, {n_questions} perguntas -> {args.output}")

    pathology_labels = {f"kvasir/{r['img_id']}": r["pathology_label"] for r in records if r["pathology_label"]}
    labels_path = args.output.replace(".arrow", "_pathology_labels.json")
    with open(labels_path, "w", encoding="utf-8") as fp:
        json.dump(pathology_labels, fp, indent=2, ensure_ascii=False)
    print(f"{len(pathology_labels)} rótulos de patologia -> {labels_path}")
