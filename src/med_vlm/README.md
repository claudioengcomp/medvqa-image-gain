This folder contains experiment for PEFT-based fine-tuning and evaluation on medical
VQA datasets.

Planned components:

- dataloaders: wrappers for SLAKE, VQA-Med, Path-VQA
- models: model loader and PEFT wrappers (LoRA, AdaLoRA)
- metrics: accuracy and BLEU computation
- run_experiment.py: CLI to run experiments and evaluations
- tests: unit tests for small components

See `pyproject.toml` for core dependencies.
