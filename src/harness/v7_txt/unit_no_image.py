"""P1.0 teste unitario (2026-10-09): um batch SEM imagem passa no forward das duas familias (Qwen2-VL-2B,
SmolVLM2-2.2B) com LoRA so no modelo de linguagem, a perda e finita, uma geracao devolve texto, e o numero de
parametros treinaveis bate com a linha congelada publicada (Qwen 4.358.144; SmolVLM FT 6.291.456).
Roda em CPU (CUDA_VISIBLE_DEVICES vazio) ou numa placa. Saida: JSON em --out. cwd = worktree."""
from __future__ import annotations
import argparse, json, os, platform, re, time
import torch
from med_vlm.enums import ModelName, PEFTMethod
from med_vlm.models.loader import load_model_bundle
from med_vlm.models.peft_adapters import apply_peft_adapter
from med_vlm.training.spec_registry import MODEL_SPECS
from harness.circle_data_bridge import load_condition_dataset

REGEX = {"QWEN2VL": r"^.*model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj$",
         "SMOLVLM": r"^.*text_model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj$"}
EXPECTED = {"QWEN2VL": 4358144, "SMOLVLM": 6291456}

def group_counts(model):
    c = {"visual": 0, "language": 0, "other": 0, "n_lora_modules": 0}
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if "visual" in n or "vision" in n: c["visual"] += p.numel()
        elif "language" in n or "text_model" in n or ".layers." in n: c["language"] += p.numel()
        else: c["other"] += p.numel()
    c["n_lora_modules"] = sum(1 for n, _ in model.named_modules() if n.endswith("lora_A.default"))
    return c

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arrow", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="QWEN2VL,SMOLVLM"); ap.add_argument("--n", type=int, default=2)
    a = ap.parse_args()
    ds = load_condition_dataset(a.arrow)
    raw = [ds[i] for i in range(a.n)]
    ex_noimg = [{"image": None, "question": r["question"], "answers": list(r["answers"]), "id": r["id"]} for r in raw]
    ex_img = [{"image": r["image"], "question": r["question"], "answers": list(r["answers"]), "id": r["id"]} for r in raw]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    report = {"host": platform.node(), "device": dev, "torch": torch.__version__, "questions": [r["question"] for r in raw],
              "gold": [list(r["answers"]) for r in raw], "models": {}}
    import transformers, peft; report["transformers"] = transformers.__version__; report["peft"] = peft.__version__
    for name in a.models.split(","):
        t0 = time.time(); mn = ModelName[name]
        bundle = load_model_bundle(mn)
        model = apply_peft_adapter(bundle.model, PEFTMethod.LORA, {"r": 16, "lora_alpha": 32, "target_modules": REGEX[name], "task_type": "CAUSAL_LM"})
        tr, tot = model.get_nb_trainable_parameters()
        gc = group_counts(model)
        model.to(dev)
        spec = MODEL_SPECS[mn](max_length=1024, no_image=True)
        collate = spec.build_collator(bundle.processor)
        batch = collate(ex_noimg)
        keys = sorted(batch.keys())
        tok = getattr(bundle.processor, "tokenizer", bundle.processor)
        img_tok = tok.convert_tokens_to_ids(getattr(tok, "image_token", "<image>"))
        has_img_tok = bool((batch["input_ids"] == img_tok).any().item()) if img_tok is not None and img_tok != tok.unk_token_id else None
        prompt_txt = bundle.processor.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": raw[0]["question"]}]}], tokenize=False, add_generation_prompt=True)
        model.train()
        out = model(**{k: (v.to(dev) if hasattr(v, "to") else v) for k, v in batch.items()})
        loss = float(out.loss.detach().float().item())
        out.loss.backward()
        gnorm = float(sum(p.grad.float().norm() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5)
        model.zero_grad(); model.eval()
        gen = spec.generate_fn(model, bundle.processor, ex_noimg, max_new_tokens=8)
        # regressao: o caminho COM imagem continua montando pixel_values (so o coletor; sem forward)
        spec_img = MODEL_SPECS[mn](max_length=1024, no_image=False)
        b_img = spec_img.build_collator(bundle.processor)(ex_img)
        report["models"][name] = {"trainable_params": int(tr), "total_params": int(tot), "expected_trainable": EXPECTED[name],
                                  "match_expected": int(tr) == EXPECTED[name], "trainable_by_group": gc, "lora_regex": REGEX[name],
                                  "batch_keys_no_image": keys, "input_ids_shape": list(batch["input_ids"].shape),
                                  "image_token_present_no_image": has_img_tok, "prompt_no_image_example": prompt_txt,
                                  "loss": loss, "loss_finite": bool(torch.isfinite(torch.tensor(loss))), "grad_norm_trainable": gnorm,
                                  "generated_no_image": gen, "with_image_batch_keys": sorted(b_img.keys()),
                                  "seconds": round(time.time() - t0, 1)}
        print(json.dumps({name: report["models"][name]}, indent=1, ensure_ascii=False), flush=True)
        del model, bundle; import gc as _gc; _gc.collect()
    report["all_ok"] = all(m["match_expected"] and m["loss_finite"] and m["image_token_present_no_image"] in (False, None) and all(isinstance(g, str) for g in m["generated_no_image"]) and "pixel_values" not in m["batch_keys_no_image"] for m in report["models"].values())
    json.dump(report, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("ALL_OK" if report["all_ok"] else "NOT_OK", flush=True)

if __name__ == "__main__":
    main()
