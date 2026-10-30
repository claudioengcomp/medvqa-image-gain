"""Schema e validação para os registros de resultado do harness do P0.

Um "results record" descreve o desempenho de UM modelo em UMA condição
(C0/C1/C2) sobre UM dataset. Dois níveis de detalhe são suportados:

- Modo agregado (sempre obrigatório): model, dataset, condition, accuracy.
  'n'/'n_correct' são opcionais, mas necessários para bootstrap CI.
- Modo pareado (opcional): 'predictions', uma lista de
  {"qid": ..., "predicted": str, "gold": str, "correct": bool} por item.
  Necessário para o teste de McNemar e o bootstrap pareado.
"""
from __future__ import annotations

REQUIRED_KEYS = ("model", "dataset", "condition", "accuracy")
VALID_CONDITIONS = (
    "C0", "C1", "C2", "C5", "C6",
    "C4_OCC_25", "C4_OCC_50", "C4_OCC_75",
    "C4_NOISE_25", "C4_NOISE_50", "C4_NOISE_75",
    # C7 reservado (26/08/2026): condição occlusion_study de oclusão posicional
    # (antiga "Tarefa 4" do PLANO_AJUSTES_P0_v2.md) -- renomeada de C6
    # pra C7 quando C6 passou a significar "mesmo órgão+modalidade,
    # categoria livre" (ver docs/superpowers/plans/2026-08-21-kvasir-targeted-pathology-swap-condition.md).
    # Ainda não implementada -- adicionada aqui só pra reservar o nome.
    "C7",
    # ISBI 2027, occlusion_study posicional (docs/isbi/occlusion_study/study_plan_v2.md): C0 restrita
    # as perguntas avaliaveis (e C0_orig, diagnostico com bytes originais), alvo T,
    # occlusion_study P, suficiencia S, dose 1,5x/2x, e preenchimentos media/blur.
    "C0_orig", "T", "P", "S", "T1.5", "P1.5", "T2", "P2", "S2",
    "T_mean", "P_mean", "T_blur", "P_blur",
    # ablacao de sigma do blur (execucao, 14/09/2026): sigma = 1/3/10 % da largura, e cor media da regiao
    "T_blur1", "P_blur1", "T_blur3", "P_blur3", "T_blur10", "P_blur10", "T_regmean", "P_regmean",
    "T_ringmean", "P_ringmean",   # E3: cor media do entorno (anel de 15 px)
    # v7 (2026-10-09): braco so-texto (TXT), C0 das perguntas cruzadas do occlusion (C0_cross) e as
    # condicoes occlusion do Kvasir/SLAKE (oclusor em imagem sem o achado)
    "TXT", "C0_cross",
    "OCCLUSION_ART_black", "OCCLUSION_ART_blur10", "OCCLUSION_POL_Q1_black", "OCCLUSION_POL_Q1_blur10",
    "OCCLUSION_POL_Q3_black", "OCCLUSION_POL_Q3_blur10", "OCCLUSION_INS_black", "OCCLUSION_INS_blur10",
    "OCCLUSION_DIS_black", "OCCLUSION_DIS_blur10",
    # v7_mimic (2026-10-11): suficiencia com blur/cor media (S_blur10, S_regmean) e occlusion do MIMIC (HANDOFF_OCCLUSION_STUDY_GPU.md)
    "S_blur10", "S_regmean", "OCCLUSION_black", "OCCLUSION_blur10",
    # m3ae_black_image (22/10/2026): M3AE com imagem constante PRETA -- so-texto treinado com preta e testado com preta; qualquer modelo testado com preta
    "TXT_PRETO", "C1_PRETO",
)


class ResultsSchemaError(ValueError):
    pass


def validate_results_dict(d: dict) -> None:
    """Levanta ResultsSchemaError se `d` não seguir o schema. Não retorna nada em sucesso."""
    missing = [k for k in REQUIRED_KEYS if k not in d]
    if missing:
        raise ResultsSchemaError(f"Campos obrigatórios ausentes: {missing}")

    if d["condition"] not in VALID_CONDITIONS:
        raise ResultsSchemaError(
            f"condition={d['condition']!r} inválida, deve ser uma de {VALID_CONDITIONS}"
        )

    if not isinstance(d["accuracy"], (int, float)) or not (0.0 <= float(d["accuracy"]) <= 1.0):
        raise ResultsSchemaError(f"accuracy={d['accuracy']!r} deve ser um número entre 0 e 1")

    has_n = "n" in d
    has_n_correct = "n_correct" in d
    if has_n != has_n_correct:
        raise ResultsSchemaError("'n' e 'n_correct' devem estar ambos presentes ou ambos ausentes")
    if has_n and (not isinstance(d["n"], int) or d["n"] <= 0):
        raise ResultsSchemaError(f"n={d['n']!r} deve ser um inteiro positivo")
    if has_n_correct and (not isinstance(d["n_correct"], int) or not (0 <= d["n_correct"] <= d["n"])):
        raise ResultsSchemaError(f"n_correct={d['n_correct']!r} deve estar entre 0 e n")

    if "predictions" in d:
        preds = d["predictions"]
        if not isinstance(preds, list) or not preds:
            raise ResultsSchemaError("'predictions', se presente, deve ser uma lista não vazia")
        for i, p in enumerate(preds):
            for key in ("qid", "predicted", "gold", "correct"):
                if key not in p:
                    raise ResultsSchemaError(f"predictions[{i}] sem o campo '{key}'")
            if not isinstance(p["correct"], bool):
                raise ResultsSchemaError(f"predictions[{i}]['correct'] deve ser bool")


def has_paired_data(d: dict) -> bool:
    return "predictions" in d and len(d["predictions"]) > 0
