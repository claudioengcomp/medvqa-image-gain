"""17/10/2026 (article_runs, pubmedbert_classifier): strong question-only classifier -- PubMedBERT (microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext)
fine-tuned only on the closed-form questions of the SAME training set as the TF-IDF of each benchmark (harness/v7_txt/dados_texto.py; HEAL: training + validation),
classification over the distinct answers of the training set. Fixed recipe for all benchmarks (declared before running): max_len 64,
batch 32, AdamW lr 2e-5, weight decay 0.01, linear warmup 10 % and linear decay, fp32. Epoch selection by accuracy on the
closed-form validation (up to 10 epochs; tie -> the earliest) where there is validation; HEAL (validation within the training set): 5 fixed epochs,
no selection. Rule: answer_matches(normalize_text(...)), the same as TF-IDF. Resumption: state at the end of each epoch in <save-dir>/ultimo
(model, optimizer, scheduler, RNG) and one row per epoch in --progress-checkpoint. Output: predictions JSON (the scheduler stamps
'provenance'; the training data goes in 'treino') and the best model in <save-dir>/melhor (never deleted)."""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import argparse, copy, hashlib, json, os, random, sys, time
sys.path.insert(0, f"{_MEDVQA_ROOT}/v7_txt/worktree")
import numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from med_vlm.metrics.process_text import answer_matches, normalize_text
MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"; REV = "e1354b7a3a09615f6aba48dfad4b7a613eef7062"
HP = {"max_len": 64, "batch": 32, "lr": 2e-5, "weight_decay": 0.01, "warmup_frac": 0.1, "epochs_max_com_validacao": 10, "epochs_fixas_sem_validacao": 5, "dtype": "fp32", "selecao": "acuracia na validacao fechada; empate -> epoca mais cedo"}
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def ok(pred, refs): return bool(answer_matches(normalize_text(str(pred)), [normalize_text(r) for r in refs]))
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True); ap.add_argument("--save-dir", required=True); ap.add_argument("--progress-checkpoint", required=True); a = ap.parse_args()
    t0 = time.time(); D = json.load(open(a.data)); tr, va, te = D["train"], D["val"], D["test"]; dev = torch.device(os.environ.get("BERT_DEVICE", "cuda"))   # BERT_DEVICE=cpu only in the path test (17/10)
    if os.environ.get("BERT_TESTE_N"): k = int(os.environ["BERT_TESTE_N"]); tr, va, te = tr[:k], va[:k], te[:k]; HP["epochs_max_com_validacao"] = HP["epochs_fixas_sem_validacao"] = 2   # only a path test
    seed_all(a.seed); labels = sorted({x["answer"] for x in tr}); L = {l: i for i, l in enumerate(labels)}
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    enc = lambda xs: tok([x["question"] for x in xs], padding="max_length", truncation=True, max_length=HP["max_len"], return_tensors="pt")
    ntok = [len(tok(x["question"])["input_ids"]) for x in tr + va + te]; trunc = sum(n > HP["max_len"] for n in ntok)
    E = enc(tr); y = torch.tensor([L[x["answer"]] for x in tr])
    g = torch.Generator(); g.manual_seed(a.seed)
    dl = DataLoader(TensorDataset(E["input_ids"], E["attention_mask"], y), batch_size=HP["batch"], shuffle=True, generator=g)
    has_val = len(va) > 0; n_ep = HP["epochs_max_com_validacao"] if has_val else HP["epochs_fixas_sem_validacao"]
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, revision=REV, num_labels=len(labels)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"]); steps = n_ep * len(dl)
    sch = get_linear_schedule_with_warmup(opt, int(HP["warmup_frac"] * steps), steps)
    @torch.no_grad()
    def predict(xs):
        model.eval(); out = []; Ex = enc(xs)
        for i in range(0, len(xs), 256):
            lo = model(input_ids=Ex["input_ids"][i:i + 256].to(dev), attention_mask=Ex["attention_mask"][i:i + 256].to(dev)).logits
            out += [labels[j] for j in lo.argmax(-1).tolist()]
        model.train(); return out
    last = os.path.join(a.save_dir, "ultimo"); os.makedirs(last, exist_ok=True); os.makedirs(os.path.dirname(a.progress_checkpoint), exist_ok=True)   # 17/10 01:2x: faltava criar ultimo/ (1a fila caiu no 1o torch.save)
    hist, start_ep, best = [], 0, {"epoch": None, "val_acc": -1.0}
    if os.path.exists(os.path.join(last, "estado.pt")) and os.path.exists(a.progress_checkpoint):   # retomada
        st = torch.load(os.path.join(last, "estado.pt"), map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sch.load_state_dict(st["sch"]); g.set_state(st["gen"]); torch.set_rng_state(st["rng"]); (torch.cuda.set_rng_state(st["cuda_rng"]) if st["cuda_rng"] is not None else None)
        random.setstate(st["py_rng"]); np.random.set_state(st["np_rng"]); hist = st["hist"]; start_ep = st["epoch"] + 1; best = st["best"]; print("retomado na epoca", start_ep, flush=True)
    best_state = torch.load(os.path.join(last, "melhor_estado.pt"), map_location="cpu", weights_only=False) if (start_ep and has_val and os.path.exists(os.path.join(last, "melhor_estado.pt"))) else None
    model.train()
    for ep in range(start_ep, n_ep):
        te0 = time.time(); tot = 0.0
        for ids, am, yy in dl:
            loss = model(input_ids=ids.to(dev), attention_mask=am.to(dev), labels=yy.to(dev)).loss; loss.backward(); opt.step(); sch.step(); opt.zero_grad(set_to_none=True); tot += loss.item()
        h = {"epoch": ep + 1, "loss_media": tot / len(dl), "s": round(time.time() - te0, 1)}
        if has_val:
            pv = predict(va); h["val_acc"] = sum(ok(p, x["refs"]) for p, x in zip(pv, va)) / len(va)
            if h["val_acc"] > best["val_acc"]: best = {"epoch": ep + 1, "val_acc": h["val_acc"]}; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; torch.save(best_state, os.path.join(last, "melhor_estado.pt"))
        hist.append(h); print(json.dumps(h), flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sch": sch.state_dict(), "gen": g.get_state(), "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state() if dev.type == "cuda" else None, "py_rng": random.getstate(), "np_rng": np.random.get_state(), "hist": hist, "epoch": ep, "best": best}, os.path.join(last, "estado.pt"))
        with open(a.progress_checkpoint, "a") as f: f.write(json.dumps({**h, "t": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
    pl = predict(te); ult = [{"qid": x["qid"], "predicted": p, "correct": ok(p, x["refs"])} for p, x in zip(pl, te)]   # 19/10 (heal_classifiers_train_only): test with the model from the LAST epoch, before loading the best one
    if has_val: model.load_state_dict(best_state)
    else: best = {"epoch": n_ep, "val_acc": None}
    mdir = os.path.join(a.save_dir, "melhor"); model.save_pretrained(mdir); tok.save_pretrained(mdir); json.dump(labels, open(os.path.join(mdir, "rotulos.json"), "w"))
    pt = predict(te); preds = [{"qid": x["qid"], "predicted": p, "gold": x["refs"][0] if len(x["refs"]) == 1 else x["refs"], "correct": ok(p, x["refs"])} for p, x in zip(pt, te)]
    k = sum(p["correct"] for p in preds); n = len(preds)
    rec = {"model": "BERT_question_only", "dataset": a.base, "condition": "C0", "accuracy": k / n, "n": n, "n_correct": k, "predictions": preds,
           "treino": {"script": os.path.abspath(__file__), "script_sha256": sha(__file__), "modelo": MODEL, "revisao": REV, "hp": HP, "seed": a.seed, "dados": a.data, "dados_sha256": sha(a.data), "fontes": D["fontes"], "fontes_sha256": D["fontes_sha256"],
                      "n_train": len(tr), "n_val": len(va), "n_test": n, "n_rotulos": len(labels), "truncadas_max_len": trunc, "epocas": n_ep, "epoca_escolhida": best["epoch"], "val_acc_escolhida": best["val_acc"], "historico": hist,
                      "modelo_salvo": mdir, "modelo_salvo_sha256": sha(os.path.join(mdir, "model.safetensors")) if os.path.exists(os.path.join(mdir, "model.safetensors")) else None, "segundos": round(time.time() - t0, 1), "pico_cuda_mib": round(torch.cuda.max_memory_allocated() / 2 ** 20) if dev.type == "cuda" else None, "teste_de_caminho": bool(os.environ.get("BERT_TESTE_N")), "ultima_epoca": {"epoca": n_ep, "accuracy": sum(x["correct"] for x in ult) / len(ult), "predictions": ult}}}
    json.dump(rec, open(a.output, "w"), indent=1); print("FIM", a.base, a.seed, "acc %.4f" % (k / n), "epoca", best["epoch"], flush=True)
if __name__ == "__main__": main()
