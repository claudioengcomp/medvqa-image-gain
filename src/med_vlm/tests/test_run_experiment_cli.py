from med_vlm.run_experiment import _apply_seed_override


def test_apply_seed_override_sets_train_seed():
    cfg = {"run_name": "qwen2vl_lora_kvasir", "train": {"epochs": 3}}
    result = _apply_seed_override(cfg, seed=1)
    assert result["train"]["seed"] == 1


def test_apply_seed_override_suffixes_run_name():
    cfg = {"run_name": "qwen2vl_lora_kvasir", "train": {}}
    result = _apply_seed_override(cfg, seed=2)
    assert result["run_name"] == "qwen2vl_lora_kvasir_seed2"


def test_apply_seed_override_suffixes_wandb_run_name():
    cfg = {"run_name": "x", "train": {}, "logging": {"wandb_run_name": "x"}}
    result = _apply_seed_override(cfg, seed=1)
    assert result["logging"]["wandb_run_name"] == "x_seed1"


def test_apply_seed_override_none_is_noop():
    cfg = {"run_name": "x", "train": {"seed": 5}}
    result = _apply_seed_override(cfg, seed=None)
    assert result == cfg
