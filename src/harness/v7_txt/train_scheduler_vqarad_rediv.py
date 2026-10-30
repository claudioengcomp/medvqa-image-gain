"""Article runs, vqarad_image_split (17/10/2026, candidato): VQA-RAD redividido por imagem (v7_txt/article_runs/data/vqarad_rediv; seed 20260913; 220/31/63 imagens,
2 467/343/705 perguntas), 18 treinos LoRA (seeds 0-2) com as receitas do A1 (YAMLs _rediv em v7_txt/article_runs/configs, copiados dos do A1),
sufixo _REDIV, e as predicoes: TXT nas so-texto, C0 nas com imagem, C1 (imagem cinza) nas 6 de encoder congelado.
Copia do escalonador do A1 (mesmas regras de lancamento, pico, morte, reserva e carimbo). Diferencas:
- placas (candidato, 17/10): LoRA so no espaco livre das placas 0-2 (a 3 fica para o M3AE, com exclusividade); placa saturada por terceiros
  (util >= 95 % nas duas leituras) recusada para TODO job;
- tudo em v7_txt/article_runs/{models,results,logs}; estado logs/train_state_vqarad_rediv.json; nada em outputs/ ou results/ do p0.
Uso (so via launch_schedulers.sh, alvo rediv): python train_scheduler_vqarad_rediv.py [--only NAME,...] [--slots N] [--dry-run]"""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import argparse, hashlib, json, os, re, shutil, socket, subprocess, sys, time
from pathlib import Path
ROOT = Path(f"{_MEDVQA_ROOT}/v7_txt"); W = Path(f"{_MEDVQA_ROOT}/v7_txt/worktree")
PJ = ROOT / "article_runs"; LOGS, MODELS, RESULTS, CONFIGS = PJ / "logs", PJ / "models", PJ / "results", PJ / "configs"; PROG = RESULTS / ".progress"   # 17/10: article_runs
PY = f"{_MEDVQA_ROOT}/miniconda3/envs/circle/bin/python"
# 17/10 02:0x (candidato): todas as placas, compartilhando com outros usuarios; limite so de VRAM (teto 31,5 GB contando terceiros), sem recusa por utilizacao
GPUS = [0, 1, 2, 3]; IMG_GPUS = [0, 1, 2, 3]; CAP_MIB = int(31.5 * 1024); LOAD_CAP = 180.0; LOAD_DELTA = 1.3; DISK_MIN_GB = 40.0; UTIL_SAT = 95
RESERVE_DEFAULT = {"qwen_txt": 5632, "smol_txt": 5632, "qwen_img": 10752, "smol_img": 8192, "predict": 5632}; ENTERED_STEP = 10
RESERVE_SRC = {"qwen_txt": "max peak_mib qwen2vl_lora_txt_* (v7_txt/logs/train_state.json) = 5 534 (VQA-RAD 5 410)", "smol_txt": "max smolvlm2_lora_txt_* = 5 264 (VQA-RAD 5 242)",
               "qwen_img": "max qwen2vl_lora_mimic / qwen2vl_lora_vt_mimic (v7_mimic/logs/train_state.json) = 10 462 / 10 556; max_length 8192 (VQA-RAD: 2048)",
               "smol_img": "max smolvlm2_*_mimic / *_ft_heal = 7 634", "predict": "max peak_mib das predicoes QWEN2VL/SMOLVLM afinadas (infer_state_* v7_txt e v7_mimic) = 5 184"}
POLL, DISK_EVERY, STATUS_EVERY, NV_EVERY = 30, 3600, 1800, 600
_MODV = (re.search(r"(\d{3}\.\d+\.\d+)", open("/proc/driver/nvidia/version").read()) or [None, None])[1] if os.path.exists("/proc/driver/nvidia/version") else None
_WA = f"{_MEDVQA_ROOT}/nvidia-580.173.02"; USE_WA = _MODV == "580.173.02"; NVSMI = (_WA + "/nvidia-smi") if USE_WA else "nvidia-smi"
ENV = {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "WANDB_MODE": "offline", "MED_VLM_NUM_PROC": "4", "TOKENIZERS_PARALLELISM": "false", "HF_HOME": f"{_MEDVQA_ROOT}/cache/huggingface"}
if USE_WA: ENV["LD_LIBRARY_PATH"] = _WA + "-compute"
HOST = socket.gethostname(); COMMIT = "2b7e731a85ab405041734565d255f2ba1e61c5b5"
TEST_ARROW = {"C0": f"{_MEDVQA_ROOT}/v7_txt/article_runs/data/vqarad_rediv/vqa_vqa_rad_test.arrow", "TXT": f"{_MEDVQA_ROOT}/v7_txt/article_runs/data/vqarad_rediv/vqa_vqa_rad_test.arrow", "C1": f"{_MEDVQA_ROOT}/v7_txt/article_runs/data/vqarad_rediv/vqa_vqa_rad_test_C1.arrow"}
TEST_ARROW_SHA = {"C0": "e5664cd040bf9800", "TXT": "e5664cd040bf9800", "C1": "9aeb366e8bdd979a"}   # manifest_vqarad_rediv.json (17/10)
FAMUP = {"qwen2vl": "QWEN2VL", "smolvlm2": "SMOLVLM"}
EXPECTED_TRAINABLE = {"txt": 4358144, "frozen": 4358144, "vt": 8290304, "smol_txt": 6291456, "smol_lora": 9277440, "smol_lora_ft": 6291456}   # setup_artigo2.json (15/10)
# (arm, cfg, fam, family, rotulo PEFT da predicao, molde da linha, condicoes): so-texto primeiro, depois os com imagem
# 15/10 ~04:20 (candidato): os 3 Qwen2-VL so-texto foram refeitos com eval.split validation e entram depois dos 3 SmolVLM2 so-texto;
# os anteriores (eval.split test) estao em tmp/v7 com sufixo _evaltest (revision A1 do artigo 2)
LINES = [("smol_txt", "smolvlm2_lora_txt_vqarad_rediv", "smol_txt", "smolvlm2", "LORA_TXT", "SMOLVLM_LORA_TXT_VQARAD_REDIV_seed%d", ["TXT"]),
         ("txt", "qwen2vl_lora_txt_vqarad_rediv", "qwen_txt", "qwen2vl", "LORA_TXT", "QWEN2VL_LORA_TXT_VQARAD_REDIV_seed%d", ["TXT"]),
         ("frozen", "qwen2vl_lora_vqarad_rediv", "qwen_img", "qwen2vl", "LORA", "QWEN2VL_LORA_VQARAD_REDIV_seed%d", ["C0", "C1"]),
         ("vt", "qwen2vl_lora_vt_vqarad_rediv", "qwen_img", "qwen2vl", "LORA_VT", "QWEN2VL_LORA_VT_VQARAD_REDIV_seed%d", ["C0"]),
         ("smol_lora", "smolvlm2_lora_vqarad_rediv", "smol_img", "smolvlm2", "LORA", "SMOLVLM_LORA_VQARAD_REDIV_seed%d", ["C0"]),
         ("smol_lora_ft", "smolvlm2_lora_ft_vqarad_rediv", "smol_img", "smolvlm2", "LORA_FT", "SMOLVLM_LORA_FT_VQARAD_REDIV_seed%d", ["C0", "C1"])]
ADAPTED_TOWER = {"vt", "smol_lora"}
ARM_DESC = {"txt": "TXT (so-texto: sem imagem; LoRA so no LM)", "frozen": "torre congelada (LoRA q/k/v/o do LM, como a publicada qwen2vl_lora_vqarad)", "vt": "torre adaptada (LoRA no LM + visual.blocks, regex VT)",
            "smol_txt": "TXT do SmolVLM2 (so-texto; LoRA so no LM, regex text_model.layers)", "smol_lora": "publicada: SmolVLM2 LoRA, lista padrao q/k/v/o (pega LM e torre)", "smol_lora_ft": "torre: SmolVLM2 LoRA so no LM (regex text_model.layers, FT), torre congelada"}
TXT_ARMS = {"txt", "smol_txt"}
SFX = "vqarad_rediv"   # sufixo dos arquivos de estado/log deste escalonador
for d in (LOGS, MODELS, RESULTS, PROG): d.mkdir(parents=True, exist_ok=True)
def now(): return time.strftime("%Y-%m-%d %H:%M:%S")
DRY = False   # salvaguarda 11/10: --dry-run => log/alert so imprimem, nada gravado
def log(e, **kw):
    r = {"t": now(), "event": ("dry:" if DRY else "") + e, **kw}
    if not DRY: open(LOGS / ("train_schedule_%s.jsonl" % SFX), "a").write(json.dumps(r, default=str) + "\n")
    print(json.dumps(r, default=str), flush=True)
def alert(name, text):
    if not DRY: (LOGS / ("ALERT_%s_%s.md" % (SFX, name))).write_text("%s\n\n%s\n" % (now(), text))
    log("ALERT", name=name, text=text)
def gpu_query():
    out = subprocess.run([NVSMI, "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    d = {}
    for l in out.strip().splitlines():
        i, u, t, ut = [x.strip() for x in l.split(",")]; d[int(i)] = {"used": int(u), "total": int(t), "util": int(ut)}
    return d
def gpu_mem(): return {g: v["used"] for g, v in gpu_query().items()}
def proc_mem():
    out = subprocess.run([NVSMI, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout; d = {}
    for l in out.strip().splitlines():
        a, b = l.split(","); d[int(a)] = d.get(int(a), 0) + int(b)
    return d
def load1(): return os.getloadavg()[0]
def libcuda_info(env):
    """12/10: libcuda que o job carrega (1o diretorio do LD_LIBRARY_PATH, senao a do sistema) e versao do modulo do kernel."""
    import re as _re
    d = (env.get("LD_LIBRARY_PATH") or "").split(":")[0]
    p = os.path.join(d, "libcuda.so.1") if d and os.path.exists(os.path.join(d, "libcuda.so.1")) else "/usr/lib/x86_64-linux-gnu/libcuda.so.1"
    r = os.path.realpath(p); m = _re.search(r"(\d{3}\.\d+\.\d+)", os.path.basename(r))
    try: k = _re.search(r"Module for \S+\s+(\d+\.\d+\.\d+)", open("/proc/driver/nvidia/version").read()).group(1)
    except Exception: k = None
    return {"libcuda": r, "libcuda_versao": m.group(1) if m else None, "modulo_kernel": k, "ld_library_path": env.get("LD_LIBRARY_PATH")}
def tree_pids(pid):
    """12/10: pid e todos os descendentes (o M3AE roda como filho do wrapper; o pico por processo era medido so no wrapper)."""
    kids = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try: ppid = int(open("/proc/%s/stat" % d).read().rsplit(")", 1)[1].split()[1])
            except Exception: continue
            kids.setdefault(ppid, []).append(int(d))
    out, st = [], [pid]
    while st:
        p = st.pop(); out.append(p); st += kids.get(p, [])
    return out
def tree_mem(pm, pid): return sum(pm.get(p, 0) for p in tree_pids(pid)) if pid else 0
def disk_free_gb(): return shutil.disk_usage(_os.environ.get("DATA_ROOT", ".")).free / 1e9
def df_h(): return subprocess.run(["df", "-h", _os.environ.get("DATA_ROOT", ".")], capture_output=True, text=True).stdout.strip().splitlines()[-1]
def alive(pid):
    try: os.kill(pid, 0)
    except OSError: return False
    try: return "zombie" not in open("/proc/%d/status" % pid).read().lower().split("state:")[1][:12]
    except Exception: return False
def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
def last_step_loss(logfile):
    try: size = os.path.getsize(logfile)
    except FileNotFoundError: return 0, None
    with open(logfile, "rb") as f:
        for win in (200_000, 2_000_000, 20_000_000, size):
            f.seek(max(0, size - win)); data = f.read().decode("utf-8", "ignore").replace("\r", "\n")
            m = re.findall(r"\[step (\d+)\] loss=([0-9.eE+-]+|nan|inf)", data)
            if m: return int(m[-1][0]), m[-1][1]
            if win >= size: break
    return 0, None
def all_step_losses(logfile):
    data = open(logfile, "rb").read().decode("utf-8", "ignore").replace("\r", "\n")
    return [(int(a), float(b)) for a, b in re.findall(r"\[step (\d+)\] loss=([0-9.eE+-]+)", data)]
def params_line(logfile):
    m = re.findall(r"\[params\] (\{.*\})", open(logfile, "rb").read().decode("utf-8", "ignore"))
    try: return json.loads(m[-1].replace("'", '"')) if m else None
    except Exception: return {"raw": m[-1]} if m else None
def versions():
    r = subprocess.run([PY, "-c", "import torch,transformers,peft,datasets,accelerate,sys;print(torch.__version__,transformers.__version__,peft.__version__,datasets.__version__,accelerate.__version__,sys.version.split()[0])"], capture_output=True, text=True, cwd=str(W)).stdout.split()
    return dict(zip(["torch", "transformers", "peft", "datasets", "accelerate", "python"], r))
def worktree_state():
    diff = subprocess.run(["git", "-C", str(W), "diff"], capture_output=True, text=True).stdout
    unt = subprocess.run(["git", "-C", str(W), "ls-files", "--others", "--exclude-standard"], capture_output=True, text=True).stdout.split()
    return {"commit": COMMIT, "tracked_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(), "untracked": unt,
            "patches": {p.name: sha_file(p) for p in sorted((ROOT / "patches").glob("*.patch"))}}
def adapter_keys(p):
    from safetensors import safe_open
    with safe_open(str(p), "pt") as f: keys = list(f.keys())
    return {"total": len(keys), "visual": sum(("visual" in k) or ("vision" in k) for k in keys), "connector": sum(("connector" in k) or ("merger" in k) for k in keys),
            "language": sum(("language_model" in k) or ("text_model" in k) for k in keys)}
def jobs_def():
    J = []
    for arm, cfg, fam, family, peft_label, line_tpl, conds in LINES:
        for sd in (0, 1, 2):
            name = "%s_seed%d" % (cfg, sd)
            J.append(dict(name=name, kind="train", cfg=str(CONFIGS / (cfg + ".yaml")), seed=sd, fam=fam, family=family, arm=arm, bench="vqarad_rediv", line=line_tpl % sd, conds=conds, peft_label=peft_label,
                          status="queued", retries=0, oom_step0=0, peak_mib=0, log=str(LOGS / (name + ".log")), order=len(J)))
    return J
def predict_jobs(j):
    out = []
    for k, c in enumerate(j["conds"]):
        name = "predict_%s_%s" % (j["name"], c)
        out.append(dict(name=name, kind="predict", fam="predict", family=j["family"], bench=j["bench"], seed=j["seed"], line=j["line"], train=j["name"], arm=j["arm"], cond=c, peft_label=j.get("peft_label"), status="queued", retries=0, oom_step0=0, peak_mib=0,
                        log=str(LOGS / (name + ".log")), output=str(RESULTS / (j["line"] + "_" + c + ".json")), order=j["order"] - 0.5 + 0.01 * k))
    return out
def is_txt(j): return j["arm"] in TXT_ARMS
def cmd_of(j):
    if j["kind"] == "train":
        return [PY, "-u", "-m", "med_vlm.run_experiment", "--config", j["cfg"], "--seed", str(j["seed"])]
    return [PY, "-u", "-m", "harness.extract_predictions_circle", "--model", FAMUP[j["family"]], "--peft", j["peft_label"], "--dataset", "VQARAD", "--condition", j["cond"],
            "--condition-arrow", TEST_ARROW[j["cond"]], "--adapter-dir", str(MODELS / j["train"]), "--batch-size", "4",
            "--progress-checkpoint", str(PROG / (j["line"] + "_" + j["cond"] + ".jsonl")), "--output", j["output"]] + (["--no-image"] if is_txt(j) else [])
class Sched:
    def __init__(self, only, slots, peer_state=None, dry_run=False, m3ae_state=None):
        if dry_run:
            global DRY; DRY = True; self.save = lambda: None
        self.m3ae_state = m3ae_state; self.peer_state = peer_state; self.sp = LOGS / ("train_state_%s.json" % SFX); self.halt = None; self.last_disk = 0; self.last_status = 0; self.last_nv = 0; self.disk_alerted = False; self.only = only; self.slots = slots
        self.vers = versions(); self.reserve = dict(RESERVE_DEFAULT)
        rp = LOGS / ("reserve_mib_%s.json" % SFX)
        if rp.exists(): self.reserve.update(json.loads(rp.read_text()))
        if dry_run and self.sp.exists(): self.st = json.loads(self.sp.read_text()); self.disk_alerted = True; self.merge_new_jobs(); return
        if self.sp.exists():
            self.st = json.loads(self.sp.read_text()); log("scheduler_restart", pid=os.getpid(), only=only, slots=slots, reserve=self.reserve); self.merge_new_jobs()
            for j in self.st["jobs"]:
                if j["status"] == "running" and not alive(j.get("pid", -1)): self.finish(j)
        else:
            self.st = {"jobs": jobs_def(), "t0": now(), "versions": self.vers}
            log("scheduler_start", pid=os.getpid(), only=only, slots=slots, rules={"cap_mib": CAP_MIB, "load_cap": LOAD_CAP, "slots_per_gpu": slots, "util_saturated": UTIL_SAT, "reserve_mib": self.reserve, "disk_min_gb": DISK_MIN_GB, "entered_step": ENTERED_STEP},
                versions=self.vers, gpu=gpu_query(), load1=round(load1(), 1), df=df_h(), worktree=worktree_state())
        self.save()
    def save(self): self.sp.write_text(json.dumps(self.st, indent=1, default=str))
    def merge_new_jobs(self):
        """11/10: jobs de jobs_def() ainda ausentes do estado entram como queued (reinicio com fila ampliada); nada existente e tocado."""
        known = {j["name"] for j in self.st["jobs"]}; new = [j for j in jobs_def() if j["name"] not in known]
        if new: self.st["jobs"].extend(new); log("jobs_added", n=len(new), names=[j["name"] for j in new])
    def m3ae_waiting(self):
        """11/10: ha treino exclusivo do M3AE (infer_state do M3AE/MIMIC) ainda na fila?"""
        try:
            if not self.m3ae_state or not Path(self.m3ae_state).exists(): return False
            return any(j.get("status") == "queued" and j.get("exclusive") for j in json.loads(Path(self.m3ae_state).read_text())["jobs"])
        except Exception: return False
    def jobs(self, **f): return [j for j in self.st["jobs"] if all(j.get(k) == v for k, v in f.items())]
    def active(self, j): return (self.only is None) or (j["name"] in self.only) or (j.get("train") in self.only)
    def peers_on(self, gpu):
        out = []
        for p in (self.peer_state or []):
            try:
                for j in json.loads(Path(p).read_text())["jobs"]:
                    if j.get("status") == "running" and j.get("gpu") == gpu: out.append({"name": "peer:" + j["name"], "kind": "peer", "start_ts": j.get("start_ts", 0), "peer": True, "log": j.get("log", ""), "pid": j.get("pid"), "peak_mib": j.get("peak_mib", 0)})
            except Exception: pass
        return out
    def ours_on(self, gpu): return [j for j in self.st["jobs"] if j["status"] == "running" and j.get("gpu") == gpu] + self.peers_on(gpu)
    def entered(self, j):
        if j.get("peer"): return time.time() - j["start_ts"] > 240
        if j["kind"] == "predict": return True
        return last_step_loss(j["log"])[0] >= ENTERED_STEP
    def can_launch(self, j, gpu, q):
        if self.halt: return False, "halt: %s" % self.halt
        if not is_txt(j) and gpu not in IMG_GPUS: return False, "job com imagem so na placa %s (candidato 15/10)" % IMG_GPUS
        if False and q[gpu]["util"] >= UTIL_SAT: return False, "placa %d saturada por terceiros (util %d %%)" % (gpu, q[gpu]["util"])
        ours = self.ours_on(gpu)
        if len(ours) >= self.slots: return False, "slots"
        if any(not self.entered(o) for o in ours): return False, "stagger: %s still initialising" % ",".join(o["name"] for o in ours if not self.entered(o))
        # 12/10 ~01:00 (candidato): treino pode dividir placa com terceiros sempre que couber na VRAM; bloqueio por saturacao de terceiros removido (limites de VRAM e regras de picos mantidos). Regra antiga (09/10 20:40) no backup.
        img_train = any(o.get("kind") == "train" and not o.get("peer") and not is_txt(o) for o in ours); cap = 30720 if (j["kind"] == "predict" and img_train) else CAP_MIB   # revision 10/10 22:10: margem de 1,5 GB para inferencia em placa com treino com imagem
        if q[gpu]["used"] + self.reserve[j["fam"]] > cap: return False, "vram %d+%d>%d" % (q[gpu]["used"], self.reserve[j["fam"]], cap)
        if j["kind"] == "predict" and img_train:   # revision 10/10 20:45 (a)
            pm = proc_mem(); mine = [o.get("pid") for o in ours if o.get("pid")]
            third = max(0, q[gpu]["used"] - sum(tree_mem(pm, x) for x in mine)); peaks = sum(max(o.get("peak_mib", 0) or 0, tree_mem(pm, o.get("pid"))) for o in ours)
            if peaks + third + self.reserve[j["fam"]] >= cap: return False, "picos %d + terceiros %d + reserva %d >= %d (placa com treino com imagem)" % (peaks, third, self.reserve[j["fam"]], cap)
        if load1() + LOAD_DELTA > LOAD_CAP: return False, "load %.1f" % load1()
        free = disk_free_gb()
        if free < DISK_MIN_GB:
            if not self.disk_alerted: alert("disk", "menos de %d G livres em DATA_ROOT (%.1f G): nenhuma linha nova sera lancada; nada foi apagado. %s" % (DISK_MIN_GB, free, df_h())); self.disk_alerted = True
            return False, "disk %.1f G" % free
        return True, "ok"
    def launch(self, j, gpu):
        q1 = gpu_query(); ok, why = self.can_launch(j, gpu, q1)
        if not ok: log("launch_refused", job=j["name"], gpu=gpu, why=why); return False
        time.sleep(20); q2 = gpu_query(); ok, why = self.can_launch(j, gpu, q2)
        if not ok or q2[gpu]["used"] > q1[gpu]["used"] + 1024: log("launch_refused", job=j["name"], gpu=gpu, why=why if not ok else "memory rising on the card (%d -> %d MiB)" % (q1[gpu]["used"], q2[gpu]["used"])); return False
        sat = q1[gpu]["util"] >= UTIL_SAT and q2[gpu]["util"] >= UTIL_SAT
        if False and sat: log("launch_refused", job=j["name"], gpu=gpu, why="placa saturada nas duas leituras (util %d/%d %%)" % (q1[gpu]["util"], q2[gpu]["util"])); return False
        env = {**os.environ, **ENV, "CUDA_VISIBLE_DEVICES": str(gpu)}; logf = open(j["log"], "ab")
        p = subprocess.Popen(["nice", "-n", "19", "ionice", "-c3"] + cmd_of(j), cwd=str(W), env=env, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        j["libcuda"] = libcuda_info(env)   # 12/10
        j.update(status="running", pid=p.pid, gpu=gpu, start=now(), start_ts=time.time(), entered_logged=False, launched_on_saturated_card=sat)
        log("launch", job=j["name"], kind=j["kind"], gpu=gpu, pid=p.pid, mem_used_mib_before=[q1[gpu]["used"], q2[gpu]["used"]], util_before=[q1[gpu]["util"], q2[gpu]["util"]], saturated=sat,
            reserve_mib=self.reserve[j["fam"]], ours_on_card=[o["name"] for o in self.ours_on(gpu) if o is not j], load1=round(load1(), 1), cmd=" ".join(cmd_of(j))); self.save(); return True
    def pick_gpu(self, j, q):
        cands = [g for g in (GPUS if is_txt(j) else IMG_GPUS) if len(self.ours_on(g)) < self.slots]
        # preferencia: placa nao saturada (util < 95 %), menos processos nossos, menos memoria usada
        return sorted(cands, key=lambda g: (g not in IMG_GPUS, q[g]["util"] >= UTIL_SAT, len(self.ours_on(g)), q[g]["used"]))   # a 3 primeiro
    def try_launches(self):
        q = gpu_query()
        for j in sorted(self.jobs(status="queued"), key=lambda x: (not (x["kind"] == "train" and (x.get("retries", 0) > 0 or x.get("oom_step0", 0) > 0)), x["kind"] != "predict", x["order"])):   # revision 10/10 20:45 (c): treino morto relanca antes de qualquer predicao
            if not self.active(j): continue
            for g in self.pick_gpu(j, q):
                ok, why = self.can_launch(j, g, q)
                if ok and self.launch(j, g): return 1   # um lancamento por ciclo: o nvidia-smi precisa ve-lo antes do proximo
            if int(time.time()) % 600 < POLL: log("waiting", job=j["name"], why=why if self.pick_gpu(j, q) else "no card with a free slot")
            break   # ordem estrita: o proximo da fila so depois deste
        return 0
    def finish(self, j):
        j["end"] = now(); j["duration_s"] = round(time.time() - j.get("start_ts", time.time()), 1)
        if j["kind"] == "predict": return self.finish_predict(j)
        ad = MODELS / j["name"] / "adapter_model.safetensors"; oj = LOGS / (j["name"] + ".json")
        if ad.exists() and oj.exists():
            j["status"] = "done"; self.stamp_train(j, ad, oj); log("done", job=j["name"], gpu=j["gpu"], duration_s=j["duration_s"], peak_mib=j["peak_mib"])
            known = {x["name"] for x in self.st["jobs"]}; self.st["jobs"].extend(p for p in predict_jobs(j) if p["name"] not in known)
        else:
            step, loss = last_step_loss(j["log"]); txt = open(j["log"], "rb").read()[-200000:].decode("utf-8", "ignore")
            oom0 = step == 0 and any(k in txt.lower() for k in ("out of memory", "busy or unavailable", "cudaerrordevicesunavailable", "cudaerrormemoryallocation", "acceleratorerror"))
            if "out of memory" in txt.lower():   # 12/10: atribuicao da causa (regra do treino morto revised)
                mine_peaks = sum((o.get("peak_mib", 0) or 0) for o in self.ours_on(j.get("gpu")) if o.get("name") not in (j["name"], "peer:" + j["name"])) + max(j.get("peak_mib", 0) or 0, self.reserve[j["fam"]])
                j.update(death_cause="mine" if mine_peaks > 30720 else "terceiros", death_gpu=j.get("gpu"), death_mine_peaks_mib=mine_peaks, death_t=now())
            else: j.update(death_cause="erro", death_gpu=j.get("gpu"), death_t=now())
            j["oom_step0"] = j.get("oom_step0", 0) + (1 if oom0 else 0)
            if not oom0: j["retries"] += 1
            if j["retries"] <= 1 and j["oom_step0"] <= 5:
                died = j["log"] + ".died%d" % (j["retries"] + j["oom_step0"]); shutil.copy2(j["log"], died); open(j["log"], "w").close()
                j.update(status="queued", pid=None, note="caiu no passo %s (log em %s); reinicio do zero" % (step, died)); log("died_relaunch", job=j["name"], step=step, last_loss=loss, log_kept=died, oom_step0=oom0, oom_step0_count=j["oom_step0"], retries=j["retries"])
            else:
                j["status"] = "failed"; alert("died_%s" % j["name"], "linha caiu pela 2a vez no passo %s (log %s); nao sera relancada." % (step, j["log"]))
        self.save()
    def finish_predict(self, j):
        out = Path(j["output"])
        if out.exists():
            try: self.stamp_predict(j); j["status"] = "done"; log("done", job=j["name"], gpu=j["gpu"], duration_s=j["duration_s"], peak_mib=j["peak_mib"], accuracy=json.loads(out.read_text()).get("accuracy"), n=json.loads(out.read_text()).get("n"))
            except Exception as e: j["status"] = "failed"; alert("stamp_%s" % j["name"], "carimbo falhou: %r" % (e,))
        else:
            txt = open(j["log"], "rb").read()[-200000:].decode("utf-8", "ignore"); oom0 = any(k in txt.lower() for k in ("out of memory", "busy or unavailable", "cudaerrordevicesunavailable", "cudaerrormemoryallocation", "acceleratorerror"))
            j["oom_step0"] = j.get("oom_step0", 0) + (1 if oom0 else 0)
            if not oom0: j["retries"] += 1
            if j["retries"] <= 1 and j["oom_step0"] <= 5:
                died = j["log"] + ".died%d" % (j["retries"] + j["oom_step0"]); shutil.copy2(j["log"], died); open(j["log"], "w").close()
                j.update(status="queued", pid=None); log("died_relaunch", job=j["name"], oom=oom0, retries=j["retries"])
            else: j["status"] = "failed"; alert("died_%s" % j["name"], "predicao caiu 2x (log %s)" % j["log"])
        self.save()
    def provenance(self, j, ad):
        return {"line": j["line"], "arm": ARM_DESC[j["arm"]], "peft_label": j.get("peft_label"), "benchmark": "VQA-RAD, pacote original (M3AE/data/finetune_data/vqa_rad), divisao do M3AE 2 758/306, teste 451 (fonte unica, 15/10)",
                "adapter_dir": str(ad.parent), "adapter_sha256": sha_file(ad), "adapter_bytes": ad.stat().st_size, "adapter_config": json.loads((ad.parent / "adapter_config.json").read_text()),
                "base_model": {"qwen2vl": "Qwen/Qwen2-VL-2B-Instruct", "smolvlm2": "HuggingFaceTB/SmolVLM2-2.2B-Instruct"}[j["family"]], "dtype": "bf16 (med_vlm/models/loader.py)",
                "versions": self.vers, "git": worktree_state(), "host": HOST, "gpu": j.get("gpu"), "env": ENV, "start": j.get("start"), "end": j.get("end"), "wall_seconds": j.get("duration_s"), "peak_vram_mib_nvidia_smi": j.get("peak_mib"), "stamped": now(), "libcuda": j.get("libcuda") or libcuda_info(ENV)}
    def stamp_train(self, j, ad, oj):
        import yaml
        d = json.loads(oj.read_text()); cfg_p = Path(j["cfg"]); cfg = yaml.safe_load(cfg_p.read_text()); keys = adapter_keys(ad); step, loss = last_step_loss(j["log"]); losses = all_step_losses(j["log"])
        pr = params_line(j["log"]); trainable = (d.get("efficiency", {}).get("params", {}) or {}).get("trainable_params")
        d["provenance"] = {**self.provenance(j, ad), "config_yaml": str(cfg_p), "config_yaml_sha256": sha_file(cfg_p), "config_effective": {**cfg, "train": {**cfg["train"], "seed": j["seed"]}, "run_name": j["name"]},
                           "train_arrows": {k: {"path": p, "sha256": sha_file(p)} for k, p in cfg["data"]["arrows"].items()},
                           "adapter_keys": keys, "tower_frozen_verified": keys["visual"] == 0 and keys["connector"] == 0 and keys["language"] > 0,
                           "trainable_params": trainable, "expected_trainable_params": EXPECTED_TRAINABLE[j["arm"]], "trainable_matches_expected": trainable == EXPECTED_TRAINABLE[j["arm"]], "params_log_line": pr,
                           "last_step": step, "final_logged_loss": loss, "loss_first_logged": losses[0][1] if losses else None, "loss_mean_first_10_logs": sum(l for _, l in losses[:10]) / max(1, len(losses[:10])) if losses else None, "loss_mean_last_10_logs": sum(l for _, l in losses[-10:]) / max(1, len(losses[-10:])) if losses else None,
                           "retries": j["retries"], "note": j.get("note"), "published_reference": "YAML copiado da configuracao publicada da mesma familia (v7_txt/article_runs/configs/configs_vqarad_rediv.json), mudando so nome, data.arrows e caminhos de saida"}
        d["provenance"]["tower_frozen_verified"] = (keys["visual"] == 0 and keys["connector"] == 0) if j["arm"] not in ADAPTED_TOWER else None
        oj.write_text(json.dumps(d, indent=1, default=str))
        if not d["provenance"]["trainable_matches_expected"]: alert("params_%s" % j["name"], "parametros treinaveis %s != esperado %s" % (trainable, EXPECTED_TRAINABLE[j["arm"]]))
        if j["arm"] not in ADAPTED_TOWER and not d["provenance"]["tower_frozen_verified"]: alert("tower_%s" % j["name"], "chaves do adapter: %s" % keys)
    def stamp_predict(self, j):
        ad = MODELS / j["train"] / "adapter_model.safetensors"; out = Path(j["output"]); d = json.loads(out.read_text())
        d["provenance"] = {**self.provenance(j, ad), "condition": j["cond"], "input_arrow": TEST_ARROW[j["cond"]], "input_arrow_sha256": sha_file(TEST_ARROW[j["cond"]]), "input_arrow_sha256_expected_prefix": TEST_ARROW_SHA[j["cond"]],
                           "inference": "harness.extract_predictions_circle" + (" --no-image" if is_txt(j) else "") + " (mesma funcao e rescoring das _C0.json publicadas); E7 nas fechadas aplicada depois por rescore_e7.py", "train_log_json": str(LOGS / (j["train"] + ".json"))}
        if not d["provenance"]["input_arrow_sha256"].startswith(TEST_ARROW_SHA[j["cond"]]): alert("arrow_%s" % j["name"], "sha do arrow de teste inesperado: %s" % d["provenance"]["input_arrow_sha256"])
        out.write_text(json.dumps(d, indent=1, default=str))
    def sample(self):
        pm = proc_mem()
        for j in self.jobs(status="running"):
            if j.get("pid") and tree_mem(pm, j["pid"]): j["peak_mib"] = max(j["peak_mib"], tree_mem(pm, j["pid"]))   # 12/10: inclui descendentes
            if j["kind"] == "train" and not j.get("entered_logged") and self.entered(j):
                j["entered_logged"] = True; log("entered_training", job=j["name"], gpu=j["gpu"], step=last_step_loss(j["log"])[0], seconds=round(time.time() - j["start_ts"]), mem_mib_now=pm.get(j["pid"]))
            if not alive(j["pid"]): self.finish(j)
    def nv_sample(self):
        q = gpu_query(); pm = proc_mem()
        for j in self.jobs(status="running"):
            p = Path(j["log"].replace(".log", ".nvidia.csv")); new = not p.exists()
            with open(p, "a") as f:
                if new: f.write("t,gpu,pid,proc_mem_mib,gpu_mem_used_mib,gpu_mem_total_mib,gpu_util_pct,load1,step,loss\n")
                st, ls = last_step_loss(j["log"]) if j["kind"] == "train" else (None, None)
                f.write("%s,%s,%s,%s,%s,%s,%s,%.1f,%s,%s\n" % (now(), j["gpu"], j["pid"], tree_mem(pm, j["pid"]), q[j["gpu"]]["used"], q[j["gpu"]]["total"], q[j["gpu"]]["util"], load1(), st, ls))
    def status_md(self):
        q = gpu_query(); lines = ["# article_runs vqarad_image_split: VQA-RAD redividido (_REDIV) -- %s" % now(), "", "load1 %.1f | %s | halt: %s" % (load1(), df_h(), self.halt), "gpu: %s" % q, ""]
        for j in sorted(self.st["jobs"], key=lambda x: x["order"]):
            step, loss = last_step_loss(j["log"]) if j["kind"] == "train" else (None, None)
            el = (time.time() - j["start_ts"]) / 3600 if j.get("start_ts") and j["status"] == "running" else None
            lines.append("- %-8s %-42s gpu %s pid %s step %s loss %s peak %s MiB %s" % (j["status"], j["name"], j.get("gpu"), j.get("pid"), step, loss, j["peak_mib"], ("%.2f h" % el) if el else (("%.2f h" % (j["duration_s"] / 3600)) if j.get("duration_s") else "")))
        (LOGS / ("status_%s.md" % SFX)).write_text("\n".join(lines) + "\n")
    def dry_run(self):
        """Salvaguarda (11/10): constantes, jobs por placa e decisao can_launch por placa de cada job ativo da fila (ordem real da fila); nada gravado, nada lancado."""
        try: gpu_query(); nv_ok = True
        except Exception as e: nv_ok = False; print("dry_run: nvidia-smi indisponivel (%r): fila listada sem decisao de placa" % (e,), flush=True)
        if not nv_ok:
            print(json.dumps({"dry_run": now(), "consts": {"CAP_MIB": CAP_MIB, "LOAD_CAP": LOAD_CAP, "UTIL_SAT": UTIL_SAT}, "reserve": self.reserve, "load1": round(load1(), 1), "df": df_h()}), flush=True)
            print("estados:", {s: len(self.jobs(status=s)) for s in ("done", "failed", "running", "queued")}, flush=True)
            for j in self.st["jobs"]:
                if j["status"] in ("running", "failed"): print("estado:", j["status"], j["name"], "gpu", j.get("gpu"), "pid", j.get("pid"), "vivo" if j.get("pid") and os.path.exists("/proc/%d" % j["pid"]) else "", flush=True)
            for j in sorted(self.jobs(status="queued"), key=lambda x: (not (x["kind"] == "train" and (x.get("retries", 0) > 0 or x.get("oom_step0", 0) > 0)), x["kind"] != "predict", x["order"])): print("fila:", j["name"], j["kind"], flush=True)
            print("fim do ciclo a seco (sem nvidia-smi): nada lancado, nada gravado", flush=True); return
        print(json.dumps({"dry_run": now(), "pid": os.getpid(), "only": self.only, "slots": self.slots, "peer_state": self.peer_state, "consts": {"CAP_MIB": CAP_MIB, "CAP_PREDICT_IMG_TRAIN_MIB": 30720, "LOAD_CAP": LOAD_CAP, "LOAD_DELTA": LOAD_DELTA, "DISK_MIN_GB": DISK_MIN_GB, "UTIL_SAT": UTIL_SAT, "ENTERED_STEP": ENTERED_STEP, "POLL": POLL}, "reserve": self.reserve, "m3ae_state": self.m3ae_state, "m3ae_waiting": self.m3ae_waiting(), "expected_trainable": EXPECTED_TRAINABLE, "reserve_src": RESERVE_SRC, "img_gpus": IMG_GPUS, "use_driver_workaround": USE_WA, "nvsmi": NVSMI, "load1": round(load1(), 1), "df": df_h(), "gpu": gpu_query(), "versions": self.vers}, default=str), flush=True)
        q = gpu_query()
        for g in GPUS: print("placa", g, "usado", q[g]["used"], "util", q[g]["util"], "na placa:", [(o["name"], o.get("pid"), "pico", o.get("peak_mib"), "entrou" if self.entered(o) else "iniciando") for o in self.ours_on(g)], flush=True)
        print("estados:", {s: len(self.jobs(status=s)) for s in ("done", "failed", "running", "queued")}, flush=True)
        for j in sorted(self.jobs(status="queued"), key=lambda x: (not (x["kind"] == "train" and (x.get("retries", 0) > 0 or x.get("oom_step0", 0) > 0)), x["kind"] != "predict", x["order"])):
            print("fila:", j["name"], j["kind"], "reserva", self.reserve[j["fam"]], "" if self.active(j) else "| inativo (--only)", ("| treino %s ainda nao done" % j["train"]) if (j["kind"] == "predict" and not any(x["name"] == j["train"] and x["status"] == "done" for x in self.st["jobs"])) else "", {g: self.can_launch(j, g, q) for g in GPUS}, flush=True)
        print("fim do ciclo a seco: nada lancado, nada gravado", flush=True)
    def run(self):
        while True:
            self.sample()
            if time.time() - self.last_disk > DISK_EVERY: log("disk", df=df_h(), free_gb=round(disk_free_gb(), 1), load1=round(load1(), 1), gpu=gpu_query()); self.last_disk = time.time()
            if time.time() - self.last_nv > NV_EVERY: self.nv_sample(); self.last_nv = time.time()
            self.try_launches(); self.save()
            if time.time() - self.last_status > STATUS_EVERY: self.status_md(); self.last_status = time.time()
            pending = [j for j in self.st["jobs"] if self.active(j) and j["status"] not in ("done", "failed")]
            if not pending:
                self.status_md(); log("all_done", only=self.only, done=len(self.jobs(status="done")), failed=[j["name"] for j in self.jobs(status="failed")], df=df_h()); return
            if self.halt and not self.jobs(status="running"): self.status_md(); log("halted_idle", halt=self.halt); return
            time.sleep(POLL)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=None); ap.add_argument("--slots", type=int, default=3); ap.add_argument("--peer-state", nargs="*", default=None); ap.add_argument("--dry-run", action="store_true", help="salvaguarda: imprime constantes e a decisao por placa; nao grava, nao lanca"); ap.add_argument("--m3ae-state", default=None, help="infer_state do M3AE/MIMIC: enquanto houver treino exclusivo na fila, a ultima placa vazia fica reservada a ele"); a = ap.parse_args()
    S = Sched(a.only.split(",") if a.only else None, a.slots, a.peer_state, dry_run=a.dry_run, m3ae_state=a.m3ae_state); S.dry_run() if a.dry_run else S.run()
