"""v7_txt (14/10/2026, candidato): C1 do HEAL para as 27 linhas afinadas. Imagem trocada pela cinza constante 384x384 RGB(128,128,128)
de harness.interventions.make_c1_blank_images (a mesma do C1 do Kvasir e do MIMIC e dos arrows cinza do M3AE so-texto).
Fonte (so leitura): v6_medida/results/arrows/stable_vqa_heal_test.arrow (o arrow dos C0 das 12 linhas de torre; 1 446 avaliaveis em 1 360 imagens).
Saidas: v7_txt/data/heal_c1/vqa_heal_test_C1.arrow + manifest_heal_c1.json (sha256 de fonte, saida e script);
data_root do M3AE em v7_txt/m3ae_roots/heal_C1/ (teste -> arrow cinza; train/val -> arrows originais do M3AE, so leitura, para o vocabulario).
Validacao: mesmas linhas e perguntas da fonte, mesmos question_id, e load_condition_dataset carrega 1 446 itens.
Imagens do HEAL nunca saem da gpu1."""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import hashlib, json, os, socket, sys, time
sys.path.insert(0, f"{_MEDVQA_ROOT}/v7_txt/worktree")
from harness.interventions.make_c1_blank_images import blank_out_images, DEFAULT_SIZE, DEFAULT_FILL_COLOR
from harness.interventions.arrow_io import read_arrow, count_questions
from harness.circle_data_bridge import load_condition_dataset
SRC = f"{_MEDVQA_ROOT}/v6_medida/results/arrows/stable_vqa_heal_test.arrow"; OUT = f"{_MEDVQA_ROOT}/v7_txt/data/heal_c1"
M3 = f"{_MEDVQA_ROOT}/doutorado/M3AE/data/finetune_arrows"; ROOT = f"{_MEDVQA_ROOT}/v7_txt/m3ae_roots/heal_C1"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
os.makedirs(OUT, exist_ok=False); os.makedirs(ROOT, exist_ok=False)
o = f"{OUT}/vqa_heal_test_C1.arrow"; t0 = time.time(); blank_out_images(SRC, o)
src_df, df = read_arrow(SRC), read_arrow(o)
assert len(df) == len(src_df) and count_questions(df) == count_questions(src_df) == 1446, (len(df), count_questions(df))
qs = lambda d: sorted(q for row in d["question_id"] for q in row)
assert qs(df) == qs(src_df)
ds = load_condition_dataset(o); assert len(ds) == 1446, len(ds)
os.symlink(o, f"{ROOT}/vqa_heal_test.arrow")
for sp in ("train", "val"): os.symlink(f"{M3}/vqa_heal_{sp}.arrow", f"{ROOT}/vqa_heal_{sp}.arrow")
old = f"{M3}/vqa_heal_test_c1.arrow"
man = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "host": socket.gethostname(), "script": __file__, "script_sha256": sha(__file__),
       "gray": {"size": DEFAULT_SIZE, "rgb": DEFAULT_FILL_COLOR, "encoder": "harness.interventions.make_c1_blank_images.blank_out_images"},
       "source": SRC, "source_sha256": sha(SRC), "output": o, "sha256": sha(o), "bytes": os.path.getsize(o), "rows": len(df), "questions": count_questions(df),
       "load_condition_dataset_n": len(ds), "seconds": round(time.time() - t0, 1),
       "m3ae_data_root": ROOT, "m3ae_train_val": {sp: {"path": f"{M3}/vqa_heal_{sp}.arrow", "sha256": sha(f"{M3}/vqa_heal_{sp}.arrow")} for sp in ("train", "val")},
       "conferencia_c1_antigo_m3ae": {"path": old, "sha256": sha(old) if os.path.exists(old) else None}}
man["conferencia_c1_antigo_m3ae"]["igual_ao_novo"] = man["conferencia_c1_antigo_m3ae"]["sha256"] == man["sha256"]
json.dump(man, open(f"{OUT}/manifest_heal_c1.json", "w"), indent=1); print(json.dumps({k: man[k] for k in ("rows", "questions", "load_condition_dataset_n", "sha256", "seconds", "conferencia_c1_antigo_m3ae")}))
