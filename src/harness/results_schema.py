"""Schema and validation for the harness result records.

A "results record" describes the performance of ONE model under ONE condition
(C0/C1/C2) on ONE dataset. Two levels of detail are supported:

- Aggregate mode (always required): model, dataset, condition, accuracy.
  'n'/'n_correct' are optional, but required for bootstrap CI.
- Paired mode (optional): 'predictions', a list of
  {"qid": ..., "predicted": str, "gold": str, "correct": bool} per item.
  Required for the McNemar test and the paired bootstrap.
"""
from __future__ import annotations

REQUIRED_KEYS = ("model", "dataset", "condition", "accuracy")
VALID_CONDITIONS = (
    "C0", "C1", "C2", "C5", "C6",
    "C4_OCC_25", "C4_OCC_50", "C4_OCC_75",
    "C4_NOISE_25", "C4_NOISE_50", "C4_NOISE_75",
    # C7 reserved (26/08/2026): occlusion_study condition for positional occlusion
    # renamed from C6 to C7 when C6 came to mean "same organ+modality,
    # free category".
    # Not yet implemented -- added here only to reserve the name.
    "C7",
    # ISBI 2027, positional occlusion study: C0 restricted to
    # the evaluable questions (and C0_orig, diagnostic with the original bytes), target T,
    # occlusion_study P, sufficiency S, dose 1,5x/2x, and mean/blur fills.
    "C0_orig", "T", "P", "S", "T1.5", "P1.5", "T2", "P2", "S2",
    "T_mean", "P_mean", "T_blur", "P_blur",
    # blur sigma ablation (run, 14/09/2026): sigma = 1/3/10 % of the width, and mean color of the region
    "T_blur1", "P_blur1", "T_blur3", "P_blur3", "T_blur10", "P_blur10", "T_regmean", "P_regmean",
    "T_ringmean", "P_ringmean",   # E3: mean color of the surroundings (15 px ring)
    # v7 (2026-10-09): text-only arm (TXT), C0 of the occlusion cross questions (C0_cross) and the
    # occlusion conditions of Kvasir/SLAKE (occluder on image without the finding)
    "TXT", "C0_cross",
    "OCCLUSION_ART_black", "OCCLUSION_ART_blur10", "OCCLUSION_POL_Q1_black", "OCCLUSION_POL_Q1_blur10",
    "OCCLUSION_POL_Q3_black", "OCCLUSION_POL_Q3_blur10", "OCCLUSION_INS_black", "OCCLUSION_INS_blur10",
    "OCCLUSION_DIS_black", "OCCLUSION_DIS_blur10",
    # v7_mimic (2026-10-11): sufficiency with blur/mean color (S_blur10, S_regmean) and MIMIC occlusion
    "S_blur10", "S_regmean", "OCCLUSION_black", "OCCLUSION_blur10",
    # m3ae_black_image (22/10/2026): M3AE with constant BLACK image -- text-only trained with black and tested with black; any model tested with black
    "TXT_PRETO", "C1_PRETO",
)


class ResultsSchemaError(ValueError):
    pass


def validate_results_dict(d: dict) -> None:
    """Raises ResultsSchemaError if `d` does not follow the schema. Returns nothing on success."""
    missing = [k for k in REQUIRED_KEYS if k not in d]
    if missing:
        raise ResultsSchemaError(f"Missing required fields: {missing}")

    if d["condition"] not in VALID_CONDITIONS:
        raise ResultsSchemaError(
            f"condition={d['condition']!r} invalid, must be one of {VALID_CONDITIONS}"
        )

    if not isinstance(d["accuracy"], (int, float)) or not (0.0 <= float(d["accuracy"]) <= 1.0):
        raise ResultsSchemaError(f"accuracy={d['accuracy']!r} must be a number between 0 and 1")

    has_n = "n" in d
    has_n_correct = "n_correct" in d
    if has_n != has_n_correct:
        raise ResultsSchemaError("'n' and 'n_correct' must be both present or both absent")
    if has_n and (not isinstance(d["n"], int) or d["n"] <= 0):
        raise ResultsSchemaError(f"n={d['n']!r} must be a positive integer")
    if has_n_correct and (not isinstance(d["n_correct"], int) or not (0 <= d["n_correct"] <= d["n"])):
        raise ResultsSchemaError(f"n_correct={d['n_correct']!r} must be between 0 and n")

    if "predictions" in d:
        preds = d["predictions"]
        if not isinstance(preds, list) or not preds:
            raise ResultsSchemaError("'predictions', if present, must be a non-empty list")
        for i, p in enumerate(preds):
            for key in ("qid", "predicted", "gold", "correct"):
                if key not in p:
                    raise ResultsSchemaError(f"predictions[{i}] without the field '{key}'")
            if not isinstance(p["correct"], bool):
                raise ResultsSchemaError(f"predictions[{i}]['correct'] must be a bool")


def has_paired_data(d: dict) -> bool:
    return "predictions" in d and len(d["predictions"]) > 0
