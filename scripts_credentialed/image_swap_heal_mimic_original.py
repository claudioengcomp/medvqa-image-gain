"""19/10/2026 (candidato, image_swap_v2): troca de imagem refeita sem as excecoes do C2 anterior. Semente 42.
HEAL: derangement no nivel da RADIOGRAFIA (image_id = <radiografia>#<regiao>): radiografias unicas em ordem alfabetica, perm = make_c2_shuffled_pairs.derangement(n_radiografias, 42);
  toda linha (radiografia, regiao) da radiografia i recebe a imagem (e o image_id da 1a linha) da radiografia perm[i]. Nenhuma pergunta recebe a propria radiografia.
MIMIC: derangement sobre as linhas (imagens) com exclusao de mesmo paciente: perm = derangement(n, 42); para cada i com subject(perm[i]) == subject(i), troca perm[i] com perm[j],
  j percorrido na ordem de numpy RandomState(42).permutation(n), aceitando a primeira troca que deixa i e j sem ponto fixo e sem mesmo paciente.
Saida: v7_txt/article_runs/data/troca2/{heal,mimic}/ (arrow, mapa de troca, manifesto). Imagens credenciadas ficam na gpu1."""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import hashlib, json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, f"{_MEDVQA_ROOT}/v7_txt/worktree")
from harness.interventions.make_c2_shuffled_pairs import derangement
from harness.interventions.arrow_io import read_arrow, write_arrow, count_questions
from harness.circle_data_bridge import load_condition_dataset
W = f"{_MEDVQA_ROOT}"; OUT = f"{W}/v7_txt/article_runs/data/troca2"; SEED = 42
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
def finish(nome, src, df, rows_map, extra):
    d = f"{OUT}/{nome}"; os.makedirs(d, exist_ok=False); o = f"{d}/vqa_{nome}_test_c2v2.arrow"; write_arrow(df, o)
    a, b = read_arrow(src), read_arrow(o); assert len(a) == len(b) and all(list(x) == list(y) for x, y in zip(a["question_id"], b["question_id"]))
    ha = [hashlib.sha256(x).hexdigest() for x in a["image"]]; hb = [hashlib.sha256(x).hexdigest() for x in b["image"]]
    nq = lambda cond: sum(len(a["question_id"][i]) for i in range(len(a)) if cond(i))
    man = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "script": __file__, "script_sha256": sha(__file__), "seed": SEED, "source": src, "source_sha256": sha(src), "output": o, "sha256": sha(o), "linhas": len(b), "perguntas": count_questions(b),
           "load_condition_dataset_n": len(load_condition_dataset(o)), "perguntas_mesmo_conteudo_sha": nq(lambda i: ha[i] == hb[i]), "perguntas_mesmo_image_id": nq(lambda i: a["image_id"][i] == b["image_id"][i]), **extra}
    json.dump(rows_map, open(f"{d}/mapa_troca.json", "w")); man["mapa"] = f"{d}/mapa_troca.json"; man["mapa_sha256"] = sha(f"{d}/mapa_troca.json")
    json.dump(man, open(f"{d}/manifest.json", "w"), indent=1); print(nome, {k: v for k, v in man.items() if not k.endswith("sha256") and k not in ("script", "source", "output", "mapa")})
# HEAL
src = f"{W}/doutorado/M3AE/data/finetune_arrows/vqa_heal_test.arrow"; df = read_arrow(src); rad = lambda x: x.split("/")[-1].split("#")[0]
R = sorted({rad(x) for x in df["image_id"]}); first = {}
for i, x in enumerate(df["image_id"]): first.setdefault(rad(x), i)
perm = derangement(len(R), SEED); donor = {R[i]: R[perm[i]] for i in range(len(R))}
img, iid = df["image"].tolist(), df["image_id"].tolist(); new = df.copy(); new["image"] = [img[first[donor[rad(x)]]] for x in iid]; new["image_id"] = [iid[first[donor[rad(x)]]] for x in iid]
assert all(rad(a) != rad(b) for a, b in zip(iid, new["image_id"]))
finish("heal", src, new, [{"linha": i, "image_id": iid[i], "radiografia": rad(iid[i]), "radiografia_doadora": donor[rad(iid[i])], "question_ids": [int(q) for q in df["question_id"][i]]} for i in range(len(df))],
       {"regra": "derangement por radiografia (1 135 radiografias unicas)", "n_radiografias": len(R), "perguntas_mesma_radiografia": sum(len(df["question_id"][i]) for i in range(len(df)) if rad(iid[i]) == rad(new["image_id"][i]))})
# MIMIC
src = f"{W}/v7_mimic/data/vqa_mimic_test.arrow"; df = read_arrow(src); it = pd.read_parquet(f"{W}/v7_mimic/data/items_test.parquet"); D2 = {r.dicom_id: (r.subject_id, r.study_id) for r in it.itertuples()}
dic = lambda x: os.path.splitext(x.split("/")[-1])[0]; subj = [D2[dic(x)][0] for x in df["image_id"]]; stud = [D2[dic(x)][1] for x in df["image_id"]]; n = len(df)
perm = derangement(n, SEED).copy(); order = np.random.RandomState(SEED).permutation(n); trocas = 0
ok = lambda i, v: v != i and subj[v] != subj[i]
for i in range(n):
    if ok(i, perm[i]): continue
    for j in order:
        if j != i and ok(i, perm[j]) and ok(j, perm[i]): perm[i], perm[j] = perm[j], perm[i]; trocas += 1; break
    else: raise SystemExit(f"sem troca valida para a linha {i}")
assert all(ok(i, perm[i]) for i in range(n)) and sorted(perm) == list(range(n))
img, iid = df["image"].tolist(), df["image_id"].tolist(); new = df.copy(); new["image"] = [img[p] for p in perm]; new["image_id"] = [iid[p] for p in perm]
finish("mimic", src, new, [{"linha": i, "image_id_receptora": iid[i], "image_id_doadora": iid[perm[i]], "question_ids": [int(q) for q in df["question_id"][i]]} for i in range(n)],
       {"regra": "derangement com exclusao de mesmo paciente (subject_id do items_test)", "reparos": trocas, "perguntas_mesmo_paciente": sum(len(df["question_id"][i]) for i in range(n) if subj[perm[i]] == subj[i]), "perguntas_mesmo_estudo": sum(len(df["question_id"][i]) for i in range(n) if stud[perm[i]] == stud[i])})
