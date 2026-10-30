"""m3ae_no_image (22/10/2026): tests of option 2 (A3/B1/C1/D1), CPU only, fp32. Run with cwd = code worktree and PYTHONPATH = that worktree.
Usage: test_no_image.py logits <ckpt> <no_image 0|1> <data_root> <saida.pt>   |   test_no_image.py novo <ckpt_txt_cinza> <saida.json>"""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import copy, json, sys, os
import torch, torch.nn.functional as F
import pytorch_lightning as pl
from m3ae.config import ex
from m3ae.datamodules.multitask_datamodule import MTDataModule
from m3ae.modules import M3AETransformerSS
from m3ae.modules import m3ae_utils
PJ = f"{_MEDVQA_ROOT}/v7_txt/article_runs"
GRAY_ROOT = PJ + "/data/vqarad_rediv/data_root_cinza/"; REAL_ROOT = PJ + "/data/vqarad_rediv/"
NC = ["task_finetune_vqa_vqa_rad", "clip16", "text_roberta"]
def upd(load_path, test_only, data_root, no_image=None, seed=0):
    u = {"data_root": data_root, "num_gpus": 1, "num_nodes": 1, "per_gpu_batchsize": 16, "image_size": 384, "tokenizer": "downloaded/roberta-base", "load_path": load_path,
         "test_only": test_only, "vqa_label_size": 498, "seed": seed, "train_transform_keys": ["clip"], "num_workers": 0}
    if no_image is not None: u["no_image"] = bool(no_image)
    return u
def batch0(cfg, split="val", n=1):
    dm = MTDataModule(cfg, dist=False); dm.setup("fit"); it = iter(dm.val_dataloader() if split == "val" else dm.train_dataloader()); return [next(it) for _ in range(n)]
def build(cfg):
    pl.seed_everything(cfg["seed"]); m = M3AETransformerSS(cfg); m.eval(); return m
def logits(m, b):
    with torch.no_grad(): r = m.infer(b); return m.vqa_head(r["multi_modal_cls_feats"]), r["multi_modal_cls_feats"]
mode = sys.argv[1]; R = {}
@ex.command
def test_no_image(_config):
    cfg = copy.deepcopy(_config)
    if mode == "logits":
        b = batch0(cfg)[0]; m = build(cfg); lg, cls = logits(m, b)
        torch.save({"qid": b["qid"], "logits": lg, "cls": cls}, sys.argv[5]); print("logits", tuple(lg.shape), "qids", b["qid"][:4])
        return
    # ---- modo novo (codigo da branch, flag ligada) ----
    ckpt = sys.argv[2]; out = {}
    b = batch0(cfg)[0]; m = build(cfg)
    # 1. A3 manual numa camada (camada 0 do lado texto)
    with torch.no_grad():
        x = m.language_encoder.embeddings(input_ids=b["text_ids"]); ext = m.language_encoder.get_extended_attention_mask(b["text_masks"], b["text_masks"].size(), "cpu")
        for layer in m.language_encoder.encoder.layer: x = layer(x, ext)[0]
        x = m.multi_modal_language_proj(x) + m.modality_type_embeddings(torch.zeros_like(b["text_masks"]))
        L0 = m.multi_modal_language_layers[0]; h1 = L0.attention(x, ext, head_mask=None, output_attentions=False, past_key_value=None)[0]
        manual = L0.feed_forward_chunk(L0.crossattention.output.LayerNorm(h1)); via = L0.forward_text_only(x, ext)[0]
    out["A3_manual_camada0_identico"] = bool(torch.equal(manual, via)); out["A3_manual_max_abs_dif"] = float((manual - via).abs().max())
    # 2. image does not enter: ViT forbidden + hook at the head input + four images
    def proibido(*a, **k): raise RuntimeError("vision_encoder called with no_image")
    m.vision_encoder.forward = proibido; cap = {}
    hk = m.vqa_head.register_forward_pre_hook(lambda mod, inp: cap.__setitem__("x", inp[0].detach().clone()))
    base, _ = logits(m, b); out["metade_imagem_exatamente_zero"] = bool((cap["x"][:, 768:] == 0).all()); out["entrada_cabeca_dim"] = int(cap["x"].shape[1])
    blk = torch.tensor([-1.792263, -1.752097, -1.48022]).view(1, 3, 1, 1).expand_as(b["image"][0]).clone()
    variantes = {"cinza (arrow D1)": b["image"][0], "preta": blk, "aleatoria": torch.randn_like(b["image"][0]), "nenhuma (None)": None}
    br = batch0(dict(cfg, data_root=REAL_ROOT))[0]; out["qids_real_iguais"] = br["qid"] == b["qid"]; variantes["real (arrow original)"] = br["image"][0]
    iguais = {}
    for nome, im in variantes.items():
        bb = dict(b); bb["image"] = [im]; lg, _ = logits(m, bb); iguais[nome] = bool(torch.equal(lg, base))
    out["logits_identicos_com_imagens_diferentes"] = iguais; hk.remove()
    # 3. forward and backward (fp32, training), gradients and NaN
    m.train(); bt = batch0(cfg, "train")[0]
    r = m.infer(bt); lg = m.vqa_head(r["multi_modal_cls_feats"]); tgt = torch.zeros_like(lg)
    for i, (ls_, sc) in enumerate(zip(bt["vqa_labels"], bt["vqa_scores"])):
        for l, s in zip(ls_, sc): tgt[i, l] = s
    loss = F.binary_cross_entropy_with_logits(lg, tgt) * tgt.shape[1]; loss.backward()
    out["perda_finita"] = bool(torch.isfinite(loss)); out["perda"] = float(loss)
    frozen = [(n, p) for n, p in m.named_parameters() if not p.requires_grad]; train = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    out["congelados_com_grad"] = sum(1 for n, p in frozen if p.grad is not None)
    out["treinaveis_sem_grad"] = [n for n, p in train if p.grad is None]
    out["grad_nan_ou_inf"] = sum(1 for n, p in train if p.grad is not None and not torch.isfinite(p.grad).all())
    out["treinaveis_grad_todo_zero"] = [n for n, p in train if p.grad is not None and float(p.grad.abs().sum()) == 0][:10]
    # 4. contagem de parametros
    tot = sum(p.numel() for p in m.parameters()); tr = sum(p.numel() for _, p in train)
    grp = {}
    for n, p in train: grp[n.split(".")[0]] = grp.get(n.split(".")[0], 0) + p.numel()
    out["parametros"] = {"total": tot, "treinaveis": tr, "congelados": tot - tr, "treinaveis_por_modulo": grp}
    # 5. head initialized the same with and without the flag (same seed, starting point m3ae.ckpt as in training)
    c_on = dict(cfg, load_path="downloaded/pretrained/m3ae.ckpt", test_only=False, no_image=True); c_off = dict(c_on, no_image=False)
    s_on = build(c_on).state_dict(); s_off = build(c_off).state_dict()
    out["init_mesmas_chaves"] = list(s_on) == list(s_off); out["init_todos_tensores_iguais"] = all(torch.equal(s_on[k], s_off[k]) for k in s_on)
    out["init_vqa_head_iguais"] = all(torch.equal(s_on[k], s_off[k]) for k in s_on if k.startswith("vqa_head"))
    torch.save({k: v for k, v in s_off.items() if k.startswith("vqa_head")}, os.path.dirname(sys.argv[3]) + "/vqa_head_init_novo.pt")
    # 6. order of the questions from the training dataloader (3 batches) with and without the flag
    q_on = [x["qid"] for x in batch0(dict(cfg, no_image=True), "train", 3)]; q_off = [x["qid"] for x in batch0(dict(cfg, no_image=False), "train", 3)]
    out["qids_treino_3_lotes_iguais_com_e_sem_flag"] = q_on == q_off
    json.dump(out, open(sys.argv[3], "w"), indent=1, default=str); print(json.dumps(out, default=str)[:3000])
if mode == "logits": U = upd(sys.argv[2], True, sys.argv[4], None if sys.argv[3] == "x" else int(sys.argv[3]))
else: U = upd(sys.argv[2], True, GRAY_ROOT, 1)
ex.run("test_no_image", named_configs=NC, config_updates=U)
