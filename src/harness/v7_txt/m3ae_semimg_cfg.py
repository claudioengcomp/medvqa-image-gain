"""22/10/2026 (m3ae_no_image): M3AE so-texto SEM imagem (opcao 2, A3/B1/C1/D1; worktree_m3ae_semimg, no_image=True). Receita e dados = os do so-texto cinza
publicado (D1: as mesmas arrows cinza; a imagem e lida e ignorada)."""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
W = f"{_MEDVQA_ROOT}"; PJ = W + "/v7_txt/article_runs"; O = PJ + "/m3ae_semimg"; WM = W + "/v7_txt/worktree_m3ae_semimg"
START_CKPT = W + "/doutorado/M3AE/downloaded/pretrained/m3ae.ckpt"; START_SHA = "469cf69b18cd75fbc3a010f202fb42c0cc074b34db75a0bfab442db982885442"
B = {  # base: task, label size, key, DS, transform por semente, data_root cinza (treino, validacao e teste)
 "VQARAD_REDIV": ("task_finetune_vqa_vqa_rad", 498, "vqa_vqa_rad", "VQARAD_REDIV", {}, PJ + "/data/vqarad_rediv/data_root_cinza/"),
 "SLAKE": ("task_finetune_vqa_slack", 222, "vqa_slack", "SLAKE", {0: "clip_resizedcrop"}, W + "/v7_txt/m3ae_arrows/data_root/"),
 "HEAL": ("task_finetune_vqa_heal", 288, "vqa_heal", "HEAL", {}, W + "/v7_txt/m3ae_arrows/data_root/"),
 "MIMIC": ("task_finetune_vqa_mimic", 82, "vqa_mimic", "MIMIC", {}, W + "/v7_mimic/m3ae_arrows_gray/data_root/"),
 "KVASIR": ("task_finetune_vqa_kvasir", 500, "vqa_kvasir", "KVASIR", {}, W + "/v7_txt/m3ae_arrows/data_root/"),
 "VQARAD_QLEVEL": ("task_finetune_vqa_vqa_rad", 498, "vqa_vqa_rad", "VQARAD_QLEVEL", {}, PJ + "/data/vqarad_qlevel/data_root_cinza/")}
ORDEM = ["VQARAD_REDIV", "SLAKE", "HEAL", "MIMIC", "KVASIR", "VQARAD_QLEVEL"]
def name(b, s): return "M3AE_SEMIMG_%s_seed%d" % (b, s)
def ck_root(b): return "%s/ckpt/%s" % (O, b)
