"""VQAHEALDataset — cópia byte a byte de
m3ae/datasets/vqa_vqa_rad_dataset.py (zhjohnchan/M3AE, commit
14da377255280eed547e451f368cf71a43f3ca1c), trocando vqa_rad->heal.
Gerado por harness/patch_m3ae_heal.py — editar o gerador no repositório
do harness, não este arquivo."""
from .base_dataset import BaseDataset


class VQAHEALDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["vqa_heal_train"]
        elif split == "val":
            names = ["vqa_heal_val"]
        elif split == "test":
            names = ["vqa_heal_test"]

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
