"""Aplica ao repositório M3AE clonado (`m3ae_dir`) o suporte ao dataset
Kvasir — não existe nativamente. Confirmado (não suposto) lendo o
código-fonte real de `zhjohnchan/M3AE`, commit
`14da377255280eed547e451f368cf71a43f3ca1c` (o mesmo pinado por
`environment/colab_setup_m3ae.sh`), via `curl` direto no GitHub nesta
sessão de planejamento — não foi preciso abrir o Colab pra descobrir isso
(diferente do que o plano de integração original previa na Task 7).

Os 2 arquivos gerados (`vqa_kvasir_dataset.py`/`vqa_kvasir_datamodule.py`)
são cópias BYTE A BYTE de `m3ae/datasets/vqa_vqa_rad_dataset.py` e
`m3ae/datamodules/vqa_vqa_rad_datamodule.py` (o precedente mais recente),
só trocando `VQAVQARAD`→`VQAKVASIR` e os nomes de arquivo `vqa_vqa_rad_*`→
`vqa_kvasir_*`. As 2 edições (`datasets/__init__.py`,
`datamodules/__init__.py`) inserem exatamente o mesmo padrão de import +
entrada de registro já usado pros outros 6 datasets do M3AE. O named
config (`config.py`) copia `task_finetune_vqa_vqa_rad` campo a campo, só
com `vqa_label_size` calculado de verdade a partir do vocabulário real do
Kvasir (não um placeholder).

Achado real que motiva `harness/prepro/split_kvasir_train_val.py`:
`VQAVQARADDataModule.setup()` monta `answer2id`/`id2answer` a partir de
`train_dataset`+`val_dataset` (NUNCA `test_dataset`) — confirmado lendo o
método linha a linha. Por isso este patch pressupõe que
`vqa_kvasir_val.arrow` já existe.

`vit`/`tokenizer`/`image_size` do named config abaixo (herdados de
`task_finetune_vqa_vqa_rad`: 'ViT-B/32'/'bert-base-uncased'/576) são
overridden na prática pelos argumentos de linha de comando que
`harness/run_finetune_m3ae.sh`/`harness/extract_predictions_m3ae.py` já
passam (`clip16 text_roberta image_size=384`) — mesmo padrão usado pro
SLAKE e VQA-RAD, não uma escolha nova feita aqui.

`batch_size=64`/`max_epoch=15` combinam o `batch_size` do precedente
VQA-RAD com o `max_epoch` (menor) do precedente SLAKE — juízo de valor
documentado, não um dado descoberto: o arrow de treino do Kvasir decomposto
não tem tamanho conhecido de antemão (depende de quantas imagens do split
train do Kvasir-VQA-x1 têm pelo menos uma pergunta closed reconhecida), e o
plano de integração já registra esse risco ("Rodada Kvasir não tem
precedente... treino sensivelmente mais longo"). Task 5 do plano de
integração já pede medir o tempo por step nas primeiras iterações antes de
assumir que os 15 epochs cabem na sessão — este valor é só um ponto de
partida seguro pra começar a medir, ajustar se necessário.

Idempotente: cada `write_*`/`patch_*` verifica se a mudança já foi aplicada
antes de escrever de novo — seguro rodar 2x na mesma sessão Colab (mesma
disciplina de `environment/colab_setup_m3ae.sh`).

Uso (dentro do Colab, depois do M3AE clonado E do vocabulário do Kvasir
construído — harness/prepro/build_kvasir_answer_vocab.py):
    python -m harness.patch_m3ae_kvasir \
        --m3ae-dir /content/M3AE \
        --vocab-json data/finetune_arrows/kvasir_answer_vocab.json
"""
from __future__ import annotations

import argparse
import json
import os

_DATASET_FILE = '''"""VQAKVASIRDataset — cópia byte a byte de
m3ae/datasets/vqa_vqa_rad_dataset.py (zhjohnchan/M3AE, commit
14da377255280eed547e451f368cf71a43f3ca1c), trocando vqa_rad->kvasir.
Gerado por harness/patch_m3ae_kvasir.py — editar o gerador no repositório
do harness, não este arquivo."""
from .base_dataset import BaseDataset


class VQAKVASIRDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["vqa_kvasir_train"]
        elif split == "val":
            names = ["vqa_kvasir_val"]
        elif split == "test":
            names = ["vqa_kvasir_test"]

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="questions"
        )

    def __getitem__(self, index):
        image_tensor = self.get_image(index)["image"]
        text = self.get_text(index)["text"]

        index, question_index = self.index_mapper[index]
        qid = self.table["question_id"][index][question_index].as_py()

        answers = self.table["answers"][index][question_index].as_py()
        labels = self.table["answer_labels"][index][question_index].as_py()
        scores = self.table["answer_scores"][index][question_index].as_py()
        answer_types = self.table["answer_type"][index][question_index].as_py()

        return {
            "image": image_tensor,
            "text": text,
            "vqa_answer": answers,
            "vqa_labels": labels,
            "vqa_scores": scores,
            "answer_types": answer_types,
            "qid": qid,
        }
'''

_DATAMODULE_FILE = '''"""VQAKVASIRDataModule — cópia byte a byte de
m3ae/datamodules/vqa_vqa_rad_datamodule.py (zhjohnchan/M3AE, commit
14da377255280eed547e451f368cf71a43f3ca1c), trocando vqa_rad->kvasir.
Gerado por harness/patch_m3ae_kvasir.py — editar o gerador no repositório
do harness, não este arquivo."""
from collections import defaultdict

from .base_datamodule import BaseDataModule
from ..datasets import VQAKVASIRDataset


class VQAKVASIRDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return VQAKVASIRDataset

    @property
    def dataset_name(self):
        return "vqa_kvasir"

    def setup(self, stage):
        super().setup(stage)

        train_answers = self.train_dataset.table["answers"].to_pandas().tolist()
        val_answers = self.val_dataset.table["answers"].to_pandas().tolist()
        train_labels = self.train_dataset.table["answer_labels"].to_pandas().tolist()
        val_labels = self.val_dataset.table["answer_labels"].to_pandas().tolist()

        all_answers = [c for c in train_answers + val_answers if c is not None]
        all_answers = [l for lll in all_answers for ll in lll for l in ll]
        all_labels = [c for c in train_labels + val_labels if c is not None]
        all_labels = [l for lll in all_labels for ll in lll for l in ll]

        self.answer2id = {k: v for k, v in zip(all_answers, all_labels)}
        sorted_a2i = sorted(self.answer2id.items(), key=lambda x: x[1])
        self.num_class = max(self.answer2id.values()) + 1

        self.id2answer = defaultdict(lambda: "unknown")
        for k, v in sorted_a2i:
            self.id2answer[v] = k
'''

_NAMED_CONFIG_TEMPLATE = '''@ex.named_config
def task_finetune_vqa_kvasir():
    exp_name = "task_finetune_vqa_kvasir"
    datasets = ["vqa_kvasir"]
    loss_names = _loss_names({{"vqa": 1}})
    batch_size = 64
    max_epoch = 15
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 5e-6
    val_check_interval = 1.0
    lr_multiplier_head = 50
    lr_multiplier_multi_modal = 5
    tokenizer = "bert-base-uncased"
    max_text_len = 32
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    input_image_embed_size = 768
    image_size = 576

    vqa_label_size = {vqa_label_size}
'''

_DATASETS_IMPORT_LINE = "from .vqa_kvasir_dataset import VQAKVASIRDataset\n"
_DATAMODULES_IMPORT_LINE = "from .vqa_kvasir_datamodule import VQAKVASIRDataModule\n"
_DATAMODULES_MARKER = "\n_datamodules = {"
_DATAMODULES_ANCHOR_ENTRY = '"vqa_vqa_rad": VQAVQARADDataModule,'
_DATAMODULES_NEW_ENTRY = '"vqa_kvasir": VQAKVASIRDataModule,'


def write_dataset_file(m3ae_dir: str) -> str:
    path = os.path.join(m3ae_dir, "m3ae", "datasets", "vqa_kvasir_dataset.py")
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(_DATASET_FILE)
    return path


def write_datamodule_file(m3ae_dir: str) -> str:
    path = os.path.join(m3ae_dir, "m3ae", "datamodules", "vqa_kvasir_datamodule.py")
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(_DATAMODULE_FILE)
    return path


def patch_datasets_init(m3ae_dir: str) -> None:
    path = os.path.join(m3ae_dir, "m3ae", "datasets", "__init__.py")
    with open(path, "r", encoding="utf-8") as fp:
        content = fp.read()
    if "VQAKVASIRDataset" in content:
        return
    if content and not content.endswith("\n"):
        content += "\n"
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(content + _DATASETS_IMPORT_LINE)


def patch_datamodules_init(m3ae_dir: str) -> None:
    """Insere o import antes do dicionário `_datamodules` e a entrada nova
    logo depois de `"vqa_vqa_rad": VQAVQARADDataModule,` — usado como âncora
    porque VQA-RAD é o precedente mais recente e sempre presente. Levanta
    ValueError se qualquer marcador não bater (sinal de que o `__init__.py`
    real mudou de formato desde que este patch foi escrito — não aplicar
    silenciosamente contra um arquivo que não reconhecemos)."""
    path = os.path.join(m3ae_dir, "m3ae", "datamodules", "__init__.py")
    with open(path, "r", encoding="utf-8") as fp:
        content = fp.read()
    if "VQAKVASIRDataModule" in content:
        return
    if _DATAMODULES_MARKER not in content:
        raise ValueError(f"{path}: marcador '_datamodules = {{' não encontrado — arquivo mudou de formato?")
    if _DATAMODULES_ANCHOR_ENTRY not in content:
        raise ValueError(f"{path}: entrada 'vqa_vqa_rad' não encontrada no dicionário — arquivo mudou de formato?")
    content = content.replace(_DATAMODULES_MARKER, f"\n{_DATAMODULES_IMPORT_LINE}{_DATAMODULES_MARKER}", 1)
    content = content.replace(
        _DATAMODULES_ANCHOR_ENTRY,
        f"{_DATAMODULES_ANCHOR_ENTRY}\n    {_DATAMODULES_NEW_ENTRY}",
        1,
    )
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(content)


def patch_config(m3ae_dir: str, vqa_label_size: int) -> None:
    path = os.path.join(m3ae_dir, "m3ae", "config.py")
    with open(path, "r", encoding="utf-8") as fp:
        content = fp.read()
    if "def task_finetune_vqa_kvasir" in content:
        return
    if content and not content.endswith("\n"):
        content += "\n"
    new_config = _NAMED_CONFIG_TEMPLATE.format(vqa_label_size=vqa_label_size)
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(content + "\n\n" + new_config)


def apply_patch(m3ae_dir: str, vocab_json_path: str) -> int:
    """Aplica o patch completo (2 arquivos novos + 3 edições) e retorna o
    vqa_label_size usado, pra o chamador conferir/logar."""
    with open(vocab_json_path, "r", encoding="utf-8") as fp:
        vocab = json.load(fp)
    vqa_label_size = len(vocab)

    write_dataset_file(m3ae_dir)
    write_datamodule_file(m3ae_dir)
    patch_datasets_init(m3ae_dir)
    patch_datamodules_init(m3ae_dir)
    patch_config(m3ae_dir, vqa_label_size)
    return vqa_label_size


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m3ae-dir", required=True)
    parser.add_argument("--vocab-json", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    vqa_label_size = apply_patch(args.m3ae_dir, args.vocab_json)
    print(f"Patch aplicado em {args.m3ae_dir}: VQAKVASIRDataset/DataModule criados e "
          f"registrados, task_finetune_vqa_kvasir com vqa_label_size={vqa_label_size}.")
