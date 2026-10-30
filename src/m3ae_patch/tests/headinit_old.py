"""m3ae_no_image: cabeca de VQA inicializada pelo codigo publicado (worktree_m3ae), mesma semente e partida -> compara com a do codigo novo."""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import copy, sys, torch, pytorch_lightning as pl
from m3ae.config import ex
from m3ae.modules import M3AETransformerSS
@ex.command
def h(_config):
    cfg = copy.deepcopy(_config); pl.seed_everything(cfg["seed"]); m = M3AETransformerSS(cfg)
    torch.save({k: v for k, v in m.state_dict().items() if k.startswith("vqa_head")}, sys.argv[1]); print("ok")
ex.run("h", named_configs=["task_finetune_vqa_vqa_rad", "clip16", "text_roberta"], config_updates={"data_root": f"{_MEDVQA_ROOT}/v7_txt/article_runs/data/vqarad_rediv/data_root_cinza/", "num_gpus": 1, "num_nodes": 1, "per_gpu_batchsize": 16, "image_size": 384, "tokenizer": "downloaded/roberta-base", "load_path": "downloaded/pretrained/m3ae.ckpt", "test_only": False, "vqa_label_size": 498, "seed": 0})
