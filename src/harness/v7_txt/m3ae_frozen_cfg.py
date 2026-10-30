"""20/10/2026 (m3ae_frozen_vit): M3AE with the visual encoder (ViT) frozen. Single table per benchmark: recipe = that of the published rows
(task from M3AE's config.py + the same arguments as run_m3ae_txt.sh / run_m3ae_mimic.sh: per_gpu_batchsize 16, clip16, text_roberta,
image_size 384, tokenizer roberta-base, load_path = m3ae.ckpt pre-training), only with freeze_vision_encoder=True. Test arrows per condition:
C0 = original test; C1 = constant grey image (384x384 RGB 128, make_c1_blank_images, the same grey as the text-only rows and the VLM families);
C2 = image swap (the same arrows as the published rows)."""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
W = f"{_MEDVQA_ROOT}"; FA = W + "/doutorado/M3AE/data/finetune_arrows"; G = W + "/v7_txt/m3ae_arrows"; PJ = W + "/v7_txt/article_runs"; MI = W + "/v7_mimic"
OUT = PJ + "/m3ae_frozen"; START_CKPT = W + "/v7_txt/worktree_m3ae/downloaded/pretrained/m3ae.ckpt"; START_SHA = "469cf69b18cd75fbc3a010f202fb42c0cc074b34db75a0bfab442db982885442"
BENCH = {   # key: task, label_size, dataset_key, DS label, training data_root, test arrows per condition, transform (per seed)
 "slake":  {"task": "task_finetune_vqa_slack", "ls": 222, "key": "vqa_slack", "ds": "SLAKE", "train_root": FA,
            "test": {"C0": FA + "/vqa_slack_test.arrow", "C1": G + "/vqa_slack_test_gray.arrow", "C2": FA + "/vqa_slack_test_c2.arrow"}, "tk": {0: "clip_resizedcrop"}},   # seed 0 publicada: clip_resizedcrop (hparams.yaml)
 "vqarad_rediv": {"task": "task_finetune_vqa_vqa_rad", "ls": 498, "key": "vqa_vqa_rad", "ds": "VQARAD_REDIV", "train_root": PJ + "/data/vqarad_rediv",
            "test": {"C0": PJ + "/data/vqarad_rediv/vqa_vqa_rad_test.arrow", "C1": PJ + "/data/vqarad_rediv/data_root_cinza/vqa_vqa_rad_test.arrow", "C2": PJ + "/data/vqarad_rediv/c2/vqa_vqa_rad_test_c2.arrow"}, "tk": {}},
 "heal":   {"task": "task_finetune_vqa_heal", "ls": 288, "key": "vqa_heal", "ds": "HEAL", "train_root": FA,
            "test": {"C0": FA + "/vqa_heal_test.arrow", "C1": G + "/vqa_heal_test_gray.arrow", "C2v2": PJ + "/data/troca2/heal/vqa_heal_test_c2v2.arrow"}, "tk": {}},
 "mimic":  {"task": "task_finetune_vqa_mimic", "ls": 82, "key": "vqa_mimic", "ds": "MIMIC", "train_root": MI + "/data",
            "test": {"C0": MI + "/data/vqa_mimic_test.arrow", "C1": MI + "/m3ae_arrows_gray/vqa_mimic_test_gray.arrow", "C2v2": PJ + "/data/troca2/mimic/vqa_mimic_test_c2v2.arrow"}, "tk": {}},
 "kvasir": {"task": "task_finetune_vqa_kvasir", "ls": 500, "key": "vqa_kvasir", "ds": "KVASIR", "train_root": FA,
            "test": {"C0": FA + "/vqa_kvasir_test.arrow", "C1": G + "/vqa_kvasir_test_gray.arrow", "C2": FA + "/vqa_kvasir_test_c2.arrow"}, "tk": {}},
 "vqarad_qlevel": {"task": "task_finetune_vqa_vqa_rad", "ls": 498, "key": "vqa_vqa_rad", "ds": "VQARAD_QLEVEL", "train_root": PJ + "/data/vqarad_qlevel",
            "test": {"C0": PJ + "/data/vqarad_qlevel/vqa_vqa_rad_test.arrow", "C1": PJ + "/data/vqarad_qlevel/data_root_cinza/vqa_vqa_rad_test.arrow", "C2": PJ + "/data/vqarad_qlevel/c2/vqa_vqa_rad_test_c2.arrow"}, "tk": {}},
}
ORDEM = ["slake", "vqarad_rediv", "heal", "mimic", "kvasir", "vqarad_qlevel"]   # from shortest to longest; qlevel last (if there is a GPU)
def name(b, s): return "M3AE_FROZEN_%s_seed%d" % (BENCH[b]["ds"], s)
def ck_root(b): return "%s/ckpt/%s" % (OUT, b)
