"""TFIDF: baseline degenerado question-only para SLAKE e VQA-RAD.

Treina um classificador simples (TF-IDF + regressão logística) que prevê a
resposta olhando SÓ o texto da pergunta, nunca a imagem. Serve de controle:
por construção, sua acurácia é idêntica em C0/C1/C2/C4x6/C5/C6 (ele nunca vê
a imagem, e nenhuma dessas condições muda o texto da pergunta — C1/C2/C4
perturbam a imagem, C5/C6 trocam a imagem entre pares doadores, mas o
enunciado da pergunta é sempre o mesmo), então não tem LDV própria — o
papel dele é servir de baseline de comparação para os modelos que realmente
usam visão.

Escopo desta primeira versão: apenas perguntas fechadas (answer_type=="CLOSED"),
que têm um vocabulário de resposta fixo e permitem correspondência exata.
Perguntas abertas ficam fora de escopo, na mesma linha de corte de escopo já
aplicada pelo P0 em outras frentes (ver spec §1.3 do handoff).

Generalizado pra VQA-RAD (03/09/2026, execução local): o filtro por
q_lang=='en' era hardcoded pro formato do SLAKE (bilíngue, PT/EN) — o
VQA-RAD não tem esse campo (é monolíngue inglês por natureza), e aplicar o
filtro do mesmo jeito descartava o dataset inteiro (s.get("q_lang") retorna
None, None != "en", tudo cai fora). Agora o filtro só é aplicado quando o
campo existe no registro.

Uso:
    python -m harness.baselines.question_only \
        --train_json Slake1.0/train.json \
        --test_json Slake1.0/test.json \
        --dataset SLAKE \
        --condition C0 \
        --output results/TFIDF_question_only_SLAKE_C0.json

    python -m harness.baselines.question_only \
        --train_json vqa_rad_prepared/trainset.json \
        --test_json vqa_rad_prepared/testset.json \
        --dataset VQA-RAD \
        --condition C0 \
        --output results/TFIDF_question_only_VQARAD_C0.json
"""
from __future__ import annotations

import argparse
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from harness.results_schema import VALID_CONDITIONS, validate_results_dict


def load_closed_qa_pairs(json_path: str) -> list[dict]:
    """Lê um arquivo train/validate/test.json (SLAKE ou VQA-RAD) e filtra
    só as perguntas de tipo fechado (answer_type=='CLOSED'). O filtro de
    idioma (q_lang=='en') só é aplicado se o campo existir no registro —
    SLAKE é bilíngue e usa esse campo, VQA-RAD não tem esse campo (é
    monolíngue por natureza do dataset original)."""
    with open(json_path, "r", encoding="utf-8") as fp:
        samples = json.load(fp)

    pairs = []
    for s in samples:
        if "q_lang" in s and s.get("q_lang") != "en":
            continue
        if str(s.get("answer_type", "")).strip().upper() != "CLOSED":
            continue
        pairs.append({
            "qid": s["qid"],
            "question": s["question"],
            "answer": str(s["answer"]).strip().lower(),
        })
    return pairs


def train_question_only_model(train_pairs: list[dict]) -> Pipeline:
    if not train_pairs:
        raise ValueError("train_pairs está vazio — nada para treinar")
    questions = [p["question"] for p in train_pairs]
    answers = [p["answer"] for p in train_pairs]
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    model.fit(questions, answers)
    return model


def evaluate_question_only_model(model: Pipeline, test_pairs: list[dict],
                                  condition: str = "C0", dataset: str = "SLAKE") -> dict:
    if not test_pairs:
        raise ValueError("test_pairs está vazio — nada para avaliar")
    questions = [p["question"] for p in test_pairs]
    predicted = model.predict(questions)

    predictions = []
    n_correct = 0
    for p, pred in zip(test_pairs, predicted):
        correct = bool(pred == p["answer"])
        n_correct += int(correct)
        predictions.append({
            "qid": p["qid"], "predicted": str(pred), "gold": p["answer"], "correct": correct,
        })

    n = len(test_pairs)
    record = {
        "model": "TFIDF_question_only",
        "dataset": dataset,
        "condition": condition,
        "accuracy": n_correct / n,
        "n": n,
        "n_correct": n_correct,
        "predictions": predictions,
    }
    validate_results_dict(record)
    return record


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--test_json", required=True)
    parser.add_argument("--dataset", default="SLAKE", choices=["SLAKE", "VQA-RAD"])
    parser.add_argument("--condition", default="C0", choices=list(VALID_CONDITIONS))
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_pairs = load_closed_qa_pairs(args.train_json)
    test_pairs = load_closed_qa_pairs(args.test_json)
    model = train_question_only_model(train_pairs)
    results = evaluate_question_only_model(model, test_pairs, condition=args.condition, dataset=args.dataset)
    with open(args.output, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)
    print(f"TFIDF: acc={results['accuracy']:.4f} ({results['n_correct']}/{results['n']}) -> {args.output}")
