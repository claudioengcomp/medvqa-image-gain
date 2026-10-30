"""TFIDF: degenerate question-only baseline for SLAKE and VQA-RAD.

Trains a simple classifier (TF-IDF + logistic regression) that predicts the
answer looking ONLY at the question text, never the image. Serves as a control:
by construction, its accuracy is identical across C0/C1/C2/C4x6/C5/C6 (it never sees
the image, and none of these conditions changes the question text — C1/C2/C4
perturb the image, C5/C6 swap the image between donor pairs, but the
question statement is always the same), so it has no LDV of its own — its
role is to serve as a comparison baseline for the models that actually
use vision.

Scope of this first version: only closed-form questions (answer_type=="CLOSED"),
which have a fixed answer vocabulary and allow exact matching.
Open questions are out of scope, following the same scope cutoff already
applied by P0 in other fronts (see spec §1.3 of the handoff).

Generalized for VQA-RAD (03/09/2026, local run): the filter by
q_lang=='en' was hardcoded for the SLAKE format (bilingual, PT/EN) — the
VQA-RAD does not have this field (it is monolingual English by nature), and applying the
filter the same way discarded the entire dataset (s.get("q_lang") returns
None, None != "en", everything falls out). Now the filter is only applied when the
field exists in the record.

Usage:
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
    """Reads a train/validate/test.json file (SLAKE or VQA-RAD) and filters
    only the closed-type questions (answer_type=='CLOSED'). The language
    filter (q_lang=='en') is only applied if the field exists in the record —
    SLAKE is bilingual and uses this field, VQA-RAD does not have this field (it is
    monolingual by nature of the original dataset)."""
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
        raise ValueError("train_pairs is empty — nothing to train")
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
        raise ValueError("test_pairs is empty — nothing to evaluate")
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
