"""Lógica pura de pontuação por item para o pipeline generativo (Qwen2-VL,
SmolVLM2 etc. via `med_vlm`) -- reaproveita as funções REAIS de
`med_vlm.metrics.process_text` (extract_answer, normalize_text), sem
reimplementar nada, mas isolada das partes de `med_vlm` que exigem
torch/transformers (models/loader.py, training/*) -- os __init__.py de
`med_vlm` e `med_vlm.metrics` são vazios, então importar só
`med_vlm.metrics.process_text` não arrasta nenhuma dependência pesada,
permitindo testar este módulo localmente sem o stack pesado instalado.

Ver docs/superpowers/specs/2026-08-18-circle-bridge-design.md §4."""
from __future__ import annotations

from med_vlm.metrics.process_text import answer_matches, extract_answer, normalize_text

from harness.results_schema import validate_results_dict


def score_prediction(
    raw_prediction: str, question: str, gold_answers: list[str]
) -> tuple[str, bool]:
    """Extrai a resposta central de `raw_prediction` (remove blocos de
    raciocínio, marcador 'the answer is:', trunca resposta aberta) e
    verifica se bate com alguma das `gold_answers`, normalizado -- mesma
    lógica exata usada por `med_vlm.metrics.accuracy.accuracy` internamente
    (extract_answer aplicado primeiro, depois normalize_text), aplicada
    aqui por item em vez de em lote, para gravar `predicted`/`correct` por
    pergunta."""
    extracted = extract_answer(raw_prediction, question)
    extracted_norm = normalize_text(extracted)
    gold_norms = [normalize_text(a) for a in gold_answers]
    correct = answer_matches(extracted_norm, gold_norms)
    return extracted, correct


def build_results_record(
    model: str, dataset: str, condition: str, predictions: list[dict]
) -> dict:
    """Monta o results record completo (harness.results_schema) a partir de
    uma lista de predições por item já pontuadas ({"qid", "predicted",
    "gold", "correct"}), computando accuracy/n/n_correct diretamente
    delas -- mesmo formato gravado por harness/extract_predictions_m3ae.py,
    herda o mesmo aparato estatístico (LDV, bootstrap, McNemar pareado)."""
    n = len(predictions)
    n_correct = sum(1 for p in predictions if p["correct"])
    record = {
        "model": model,
        "dataset": dataset,
        "condition": condition,
        "accuracy": (n_correct / n) if n else 0.0,
        "n": n,
        "n_correct": n_correct,
        "predictions": predictions,
    }
    validate_results_dict(record)
    return record
