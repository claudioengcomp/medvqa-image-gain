"""Traduz um arquivo .arrow de condição (schema M3AE/LaPA — uma linha por
imagem, com listas de perguntas/respostas por imagem, ver
harness/interventions/arrow_io.py) para um `datasets.Dataset` do
HuggingFace no formato que o pacote `med_vlm` espera de cada exemplo:
{"image": PIL.Image, "question": str, "answers": list[str], "id": int}
(mesmo contrato produzido por med_vlm.data.normalizer.build_normalizer).

Achata cada imagem nas N perguntas que ela carrega, reaproveitando os
mesmos bytes de imagem (decodificados automaticamente pela feature
`datasets.Image()` do HuggingFace ao acessar cada linha, não decodificados
manualmente aqui). `id` é o `question_id` real do arrow — diferente do
`id_key=None` que `med_vlm.data.loader._normalize` usa por padrão —
necessário para preservar o pareamento por qid entre condições, usado pelo
McNemar pareado (harness.stats.compute_ldv.compute_summary_vs_baseline).

Ver docs/superpowers/specs/2026-08-18-circle-bridge-design.md §3."""
from __future__ import annotations

from datasets import Dataset, Features, Image as HFImage, Sequence, Value

from harness.interventions.arrow_io import read_arrow

CONDITION_DATASET_FEATURES = Features({
    "image": HFImage(),
    "question": Value("string"),
    "answers": Sequence(Value("string")),
    "id": Value("int64"),
})


def load_condition_dataset(condition_arrow_path: str) -> Dataset:
    """Lê `condition_arrow_path` (schema M3AE) e retorna um
    `datasets.Dataset` com uma linha por pergunta, pronto para os
    coletores (`build_chat_template_vqa_collator`) e o loop de geração
    (`generate_chat_template`) de `med_vlm` consumirem sem nenhuma
    alteração no pacote copiado."""
    df = read_arrow(condition_arrow_path)

    images: list[bytes] = []
    questions: list[str] = []
    answers: list[list[str]] = []
    ids: list[int] = []

    for _, row in df.iterrows():
        image_bytes = row["image"]
        for question, answer_list, qid in zip(
            row["questions"], row["answers"], row["question_id"]
        ):
            images.append(image_bytes)
            questions.append(question)
            answers.append(list(answer_list))
            ids.append(int(qid))

    return Dataset.from_dict(
        {"image": images, "question": questions, "answers": answers, "id": ids},
        features=CONDITION_DATASET_FEATURES,
    )
