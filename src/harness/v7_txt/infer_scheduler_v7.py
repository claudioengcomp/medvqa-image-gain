"""v7 (2026-10-09): escalonador generico de INFERENCIA (baterias occlusion do Kvasir/SLAKE, zero-shot C0/TXT, teste do M3AE
so-texto). Le uma lista de jobs em JSON (gerada por make_jobs_*.py), cada um com cmd, env (circle|lapa), reserve_mib,
output, log, e metadados de proveniencia. Mesmas regras de placa do train_scheduler_v7 (teto 31,5 GB contando terceiros,
load1 <= 180, leitura dupla 20 s, prefere placa nao saturada, N slots nossos por placa, df hourly, nunca apaga).
Carimbo de proveniencia nos JSONs dos jobs PEFT/zero-shot (adapter sha, modelo base, dtype, versoes, worktree, host,
placa, datas); os jobs M3AE carimbam-se sozinhos (m3ae_infer_v7.py). Estado em <state>; reinicio seguro.
Uso: python infer_scheduler_v7.py --jobs JOBS.json --state STATE.json --log-dir DIR [--slots 2]"""
import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import argparse, hashlib, json, os, shutil, socket, subprocess, time
from pathlib import Path
W = Path(f"{_MEDVQA_ROOT}/v7_txt/worktree"); PY = {"circle": f"{_MEDVQA_ROOT}/miniconda3/envs/circle/bin/python", "lapa": f"{_MEDVQA_ROOT}/miniconda3/envs/lapa/bin/python"}
# revision 10/10 (aplicada 11/10 00:09 UTC): CAP_IMG_TRAIN_MIB = limite com margem de 1,5 GB nas placas com treino do MIMIC com imagem
CAP_IMG_TRAIN_MIB = 30720
# 14/10 (candidato): varios lancamentos por ciclo. Leitura dupla de 20 s uma vez por ciclo; entre lancamentos na mesma placa, STAGGER_S.
# Memoria ainda nao alocada pelos jobs jovens (< YOUNG_S; meus e dos pares) entra na conta: max(0, reserva - memoria atual da arvore do processo).
STAGGER_S = 5; YOUNG_S = 240; INTENT_TTL = 90
GPUS = [0, 1, 2, 3]; CAP_MIB = int(31.5 * 1024); LOAD_CAP = 180.0; LOAD_DELTA = 1.3; DISK_MIN_GB = 40.0; UTIL_SAT = 95; POLL, DISK_EVERY, STATUS_EVERY, NV_EVERY = 30, 3600, 1800, 600
# 14/10/2026 14:45 (reboot da gpu1): o contorno do driver (bibliotecas 580.173.02) so vale com o modulo 580.173.02 carregado; com o modulo novo
# (580.178.04, igual a biblioteca do sistema), usa-se o nvidia-smi e a libcuda do sistema. Escolha pela versao do modulo em /proc.
import re as _re
_MODV = (_re.search(r"(\d{3}\.\d+\.\d+)", open("/proc/driver/nvidia/version").read()) or [None, None])[1] if os.path.exists("/proc/driver/nvidia/version") else None
_WA = f"{_MEDVQA_ROOT}/nvidia-580.173.02"; USE_WA = _MODV == "580.173.02"; NVSMI = (_WA + "/nvidia-smi") if USE_WA else "nvidia-smi"
ENV = {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "WANDB_MODE": "offline", "TOKENIZERS_PARALLELISM": "false", "HF_HOME": f"{_MEDVQA_ROOT}/cache/huggingface"}   # 11/10: HF_HOME no NVMe (${MEDVQA_ROOT}), nunca no ${HOME} (HD SMR; skill lasid-gpu-infra regra 7)
if USE_WA: ENV["LD_LIBRARY_PATH"] = _WA + "-compute"
HOST = socket.gethostname()   # 11/10 21:5x: linha propria (o comentario da revision HF_HOME tinha comentado esta atribuicao)
def now(): return time.strftime("%Y-%m-%d %H:%M:%S")
def gpu_query():
    out = subprocess.run([NVSMI, "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout; d = {}
    for l in out.strip().splitlines():
        i, u, t, ut = [x.strip() for x in l.split(",")]; d[int(i)] = {"used": int(u), "total": int(t), "util": int(ut)}
    return d

def defer_min_reserve(paths, cache={}):
    """13/10 (candidato): fila unica por prioridade. Menor reserva entre os jobs na fila (queued, requires presentes, 'after' concluido) dos
    escalonadores de prioridade maior (--defer-to STATE ...). Cache de 15 s."""
    key = tuple(paths or ())
    if not key: return None
    if time.time() - cache.get(key, (0, None))[0] < 15: return cache[key][1]
    best = None
    for p in key:
        try:
            js = json.loads(Path(p).read_text())["jobs"]; done = {x["name"] for x in js if x.get("status") == "done"}
            for x in js:
                if x.get("status") != "queued" or (x.get("after") and x["after"] not in done) or any(not Path(r).exists() for r in x.get("requires", [])): continue
                r = x.get("reserve_mib") or 0
                best = r if best is None else min(best, r)
        except Exception: pass
    cache[key] = (time.time(), best); return best
def third_party_mib():
    """13/10: memoria de processos de OUTROS usuarios por placa (nvidia-smi do contorno + dono do processo em /proc)."""
    out = {}
    try:
        u = subprocess.run([NVSMI, "--query-gpu=index,uuid", "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout
        idx = {l.split(",")[1].strip(): int(l.split(",")[0]) for l in u.strip().splitlines() if "," in l}
        a = subprocess.run([NVSMI, "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
        me = os.getuid()
        for l in a.strip().splitlines():
            g, pid, m = [x.strip() for x in l.split(",")]
            try: uid = os.stat("/proc/%s" % pid).st_uid
            except Exception: uid = -1
            if uid != me: out[idx.get(g, -1)] = out.get(idx.get(g, -1), 0) + int(m)
    except Exception: pass
    return out
def proc_mem():
    out = subprocess.run([NVSMI, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout; d = {}
    for l in out.strip().splitlines():
        a, b = l.split(","); d[int(a)] = d.get(int(a), 0) + int(b)
    return d
def load1(): return os.getloadavg()[0]
def job_condition(j):
    """13/10: condicao que o job grava no resultado (--condition; M3AE: 2o argumento do m3ae_mimic_cond.py)."""
    c = j.get("cmd", [])
    if "--condition" in c: return c[c.index("--condition") + 1]
    if c and c[0].endswith("m3ae_mimic_cond.py") and len(c) > 2: return c[2]
    return None
def schema_conditions():
    import sys as _s; _s.path.insert(0, str(W)); from harness.results_schema import VALID_CONDITIONS; return set(VALID_CONDITIONS)
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
def mem_available_gb():   # 22/10 (artigo 1, rodada 17): RAM disponivel do servidor (MemAvailable)
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable:"): return int(l.split()[1]) / 1048576
    return 1e9
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
def versions(env):
    r = subprocess.run([PY[env], "-c", "import torch,sys;print(torch.__version__,sys.version.split()[0]);import transformers,peft;print(transformers.__version__,peft.__version__)"], capture_output=True, text=True, cwd=str(W)).stdout.split()
    return dict(zip(["torch", "python", "transformers", "peft"], r))
def worktree_state():
    diff = subprocess.run(["git", "-C", str(W), "diff"], capture_output=True, text=True).stdout
    return {"commit": "2b7e731", "tracked_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(), "untracked": subprocess.run(["git", "-C", str(W), "ls-files", "--others", "--exclude-standard"], capture_output=True, text=True).stdout.split()}
class Sched:
    def __init__(self, a):
        self.a = a; self.sp = Path(a.state); self.logd = Path(a.log_dir); self.logd.mkdir(parents=True, exist_ok=True); self.halt = None; self.last_disk = self.last_status = self.last_nv = 0; self.disk_alerted = False
        self.vers = {e: versions(e) for e in ("circle", "lapa")}
        if getattr(a, "dry_run", False):   # ciclo a seco (salvaguarda 11/10): estado so leitura, nada gravado, nada lancado
            self.save = lambda: None; self.disk_alerted = True
            self.log = lambda e, **kw: print(json.dumps({"t": now(), "event": "dry:" + e, **kw}, default=str), flush=True); self.alert = lambda n, x: print("dry:ALERT", n, x, flush=True)
            self.st = json.loads(self.sp.read_text()) if self.sp.exists() else {"jobs": [dict(j, status="queued", retries=0, oom=0, peak_mib=0) for j in json.loads(Path(a.jobs).read_text())], "t0": now(), "versions": self.vers}
            known = {j["name"] for j in self.st["jobs"]}   # 14/10: o ciclo a seco mostra (so em memoria) os jobs novos que entrariam na reinicializacao
            for j in json.loads(Path(a.jobs).read_text()):
                if j["name"] not in known: self.st["jobs"].append(dict(j, status="queued", retries=0, oom=0, peak_mib=0)); print("dry:job_novo", j["name"], flush=True)
            return
        if self.sp.exists():
            self.st = json.loads(self.sp.read_text()); self.log("scheduler_restart", pid=os.getpid())
            known = {j["name"] for j in self.st["jobs"]}
            for j in json.loads(Path(a.jobs).read_text()):   # jobs novos acrescentados na reinicializacao
                if j["name"] not in known: j.update(status="queued", retries=0, oom=0, peak_mib=0); self.st["jobs"].append(j)
            for j in self.st["jobs"]:
                if j["status"] == "running" and not alive(j.get("pid", -1)): self.finish(j)
        else:
            self.st = {"jobs": [dict(j, status="queued", retries=0, oom=0, peak_mib=0) for j in json.loads(Path(a.jobs).read_text())], "t0": now(), "versions": self.vers}
            self.log("scheduler_start", pid=os.getpid(), jobs=len(self.st["jobs"]), rules={"cap_mib": CAP_MIB, "load_cap": LOAD_CAP, "slots": a.slots, "util_sat": UTIL_SAT}, gpu=gpu_query(), load1=round(load1(), 1), df=df_h(), worktree=worktree_state())
        for j in self.st["jobs"]:
            if j["status"] == "queued" and Path(j["output"]).exists() and "provenance" in json.loads(Path(j["output"]).read_text()): j["status"] = "done"; j["note"] = "output ja existia com carimbo"
        self.save()
    def log(self, e, **kw):
        r = {"t": now(), "event": e, **kw}; open(self.logd / "schedule.jsonl", "a").write(json.dumps(r, default=str) + "\n"); print(json.dumps(r, default=str), flush=True)
    def alert(self, name, text): (self.logd / ("ALERT_%s.md" % name)).write_text("%s\n\n%s\n" % (now(), text)); self.log("ALERT", name=name, text=text)
    def save(self): self.sp.write_text(json.dumps(self.st, indent=1, default=str))
    def jobs(self, **f): return [j for j in self.st["jobs"] if all(j.get(k) == v for k, v in f.items())]
    def peers_on(self, g):
        """Jobs em curso de OUTROS escalonadores meus (--peer-state): contam nos slots e no stagger da placa."""
        out = []
        for p in (self.a.peer_state or []):
            try:
                for j in json.loads(Path(p).read_text())["jobs"]:
                    if j.get("status") == "running" and j.get("gpu") == g: out.append({"name": "peer:" + j["name"], "start_ts": j.get("start_ts", 0), "peer": True, "pid": j.get("pid"), "peak_mib": j.get("peak_mib", 0), "img_train": (j.get("name", "").startswith("train_M3AE") and "_TXT_" not in j.get("name", "")) or j.get("kind") == "train" and "_txt_" not in j["name"], "is_train": j.get("kind") == "train", "reserve": j.get("reserve_mib") or (6144 if j.get("kind") == "predict" else None)})
            except Exception: pass
        return out
    def ours_on(self, g):
        own = [j for j in self.st["jobs"] if j["status"] == "running" and j.get("gpu") == g]
        for j in own: j["img_train"] = bool(j["name"].startswith("train_M3AE") and "_TXT_" not in j["name"])   # 13/10: treino do M3AE com imagem conta como treino com imagem (teto 30 720)
        return own + self.peers_on(g)
    def train_deaths(self):
        """Revision 10/10 20:45: treinos do MIMIC (peer kind=train) mortos e ainda nao relancados (queued com retries/oom_step0 > 0) ou failed."""
        out = []
        for p in (self.a.peer_state or []):
            try:
                for j in json.loads(Path(p).read_text())["jobs"]:
                    if j.get("kind") == "train" and (j.get("status") == "failed" or (j.get("status") == "queued" and (j.get("retries", 0) > 0 or j.get("oom_step0", 0) > 0))):
                        out.append({"name": j["name"], "cause": j.get("death_cause", "desconhecida"), "gpu": j.get("death_gpu", j.get("gpu"))})   # 12/10: causa atribuida pelo escalonador de treino
            except Exception: pass
        return out
    def exclusive_waiting(self):
        """11/10: outro escalonador meu (--reserve-for-exclusive STATE) tem job exclusivo (M3AE) na fila? Entao a ultima placa vazia fica para ele."""
        try:
            pth = getattr(self.a, "reserve_for_exclusive", None)
            if not pth or not Path(pth).exists(): return False
            return any(j.get("status") == "queued" and j.get("exclusive") for j in json.loads(Path(pth).read_text())["jobs"])
        except Exception: return False
    def trains_queued(self):
        """11/10: o escalonador de treino (--yield-to-train STATE) tem treino na fila?"""
        try:
            pth = getattr(self.a, "yield_to_train", None)
            if not pth or not Path(pth).exists(): return False
            return any(j.get("status") == "queued" and j.get("kind") == "train" for j in json.loads(Path(pth).read_text())["jobs"])
        except Exception: return False
    def pending_mib(self, g):
        """14/10 11:10: soma, em TODOS os jobs em curso na placa (meus e dos pares), de max(0, reserva - memoria atual da arvore do processo).
        (11:05: dois OOM com a versao que so contava jobs < YOUNG_S; jobs mais velhos ainda crescem ate o pico durante a inferencia.)"""
        pm = proc_mem(); tot = 0
        for o in self.ours_on(g):
            r = (o.get("reserve") if o.get("peer") else o.get("reserve_mib")) or 0
            tot += max(0, r - tree_mem(pm, o.get("pid")))
        return tot
    def higher_intent(self, g):
        """14/10: nome do estado de prioridade maior que publicou, ha menos de INTENT_TTL s, intencao de lancar na placa g; senao None."""
        for p in (getattr(self.a, "defer_to", None) or []):
            try:
                it = json.loads(Path(p).read_text()).get("intent") or {}
                if time.time() - it.get("t", 0) < INTENT_TTL and g in it.get("cards", []): return Path(p).name
            except Exception: pass
        return None
    def pm_cached(self):
        if time.time() - getattr(self, "_pm_t", 0) > 15: self._pm = proc_mem(); self._pm_t = time.time()
        return self._pm
    def can_launch(self, j, g, q):
        if self.halt: return False, "halt"
        ours = self.ours_on(g); img_train = [o for o in ours if o.get("img_train")]
        deaths = self.train_deaths()   # revision 10/10 20:45 (b), refeita 12/10: so morte por memoria atribuivel a jobs meus bloqueia tudo
        mine = [d["name"] for d in deaths if d["cause"] in ("mine", "desconhecida")]; third_cards = {d["gpu"] for d in deaths if d["cause"] == "terceiros" and d["gpu"] is not None}
        if third_cards - set(self.st.get("third_slot_cards", [])):
            self.st["third_slot_cards"] = sorted(set(self.st.get("third_slot_cards", [])) | third_cards); self.log("third_slot_suspended_card", cards=self.st["third_slot_cards"], deaths=[d for d in deaths if d["cause"] == "terceiros"], rule="morte por crescimento de terceiros: terceira vaga suspensa so nesta placa"); self.save()
        if mine:
            if not self.st.get("third_slot_suspended"): self.st["third_slot_suspended"] = now(); self.log("third_slot_suspended", deaths=mine, rule="treino do MIMIC morreu por memoria atribuivel a jobs meus: terceira vaga suspensa nas placas com treino; nenhum lancamento ate o relancamento"); self.save()
            return False, "treino do MIMIC morto por memoria minha e nao relancado (%s): nenhum lancamento" % ",".join(mine)
        slots = min(self.a.slots, 2) if ((self.st.get("third_slot_suspended") and img_train) or g in self.st.get("third_slot_cards", [])) else self.a.slots
        lmax = getattr(self.a, "light_max_mib", 7168); is_light = lambda o: (not o.get("is_train") and (o.get("reserve") or 10 ** 9) <= lmax) if o.get("peer") else (o.get("reserve_mib") or 10 ** 9) <= lmax
        if getattr(self.a, "light_slots", None) and slots == self.a.slots and j["reserve_mib"] <= lmax and all(is_light(o) for o in ours): slots = max(slots, self.a.light_slots)   # 12/10 13:20
        if len(ours) >= slots: return False, "slots"
        if any(not Path(x).exists() for x in j.get("requires", [])): return False, "requires ausente: %s" % ",".join(x for x in j.get("requires", []) if not Path(x).exists())
        hi = self.higher_intent(g)   # 14/10 (candidato): cede so quando a fila de prioridade maior de fato vai lancar nesta placa neste ciclo
        if hi: return False, "cede a prioridade maior (%s vai lancar nesta placa)" % hi
        if getattr(self.a, "yield_to_train", None) and self.trains_queued():
            ntr = sum(1 for o in ours if o.get("kind") == "train" or (o.get("peer") and o.get("img_train")) or o.get("is_train"))
            if len(ours) + 1 + max(0, 2 - ntr) > slots: return False, "vaga guardada para treino (treinos na fila; %d treino(s) na placa)" % ntr
        if not j.get("exclusive") and not ours and self.exclusive_waiting() and len([x for x in GPUS if not self.ours_on(x)]) <= 1: return False, "ultima placa vazia reservada ao job exclusivo (M3AE) de %s" % self.a.reserve_for_exclusive
        if any(time.time() - (o.get("start_ts") or 0) < STAGGER_S for o in ours): return False, "stagger"   # 14/10: 5 s (antes 90 s meus / 240 s dos pares)
        if j.get("exclusive") and ours: return False, "exclusive"
        cap = CAP_IMG_TRAIN_MIB if img_train else CAP_MIB   # revision 10/10 22:10: 30 720 nas placas com treino do MIMIC com imagem, 32 256 nas demais
        pend = self.pending_mib(g)   # 14/10: reserva ainda nao alocada pelos jobs jovens da placa
        if q[g]["used"] + pend + j["reserve_mib"] > cap: return False, "vram %d+%d(pendente)+%d > %d" % (q[g]["used"], pend, j["reserve_mib"], cap)
        if img_train:   # revision 10/10 20:45 (a): picos medidos dos jobs na placa + terceiros (uso atual) + reserva do novo < limite
            pm = self.pm_cached(); mine = [o.get("pid") for o in ours if o.get("pid")]
            third = max(0, q[g]["used"] - sum(tree_mem(pm, x) for x in mine)); peaks = sum(max(o.get("peak_mib", 0) or 0, tree_mem(pm, o.get("pid"))) for o in ours)
            if peaks + third + j["reserve_mib"] >= cap: return False, "picos %d + terceiros %d + reserva %d >= %d (placa com treino do MIMIC com imagem)" % (peaks, third, j["reserve_mib"], cap)
        if load1() + LOAD_DELTA > LOAD_CAP: return False, "load %.1f" % load1()
        rmin = getattr(self.a, "min_free_ram_gb", None)   # 22/10 (rodada 17): nao lanca com pouca RAM livre
        if rmin and mem_available_gb() < rmin: return False, "ram %.0f GB disponiveis < %.0f" % (mem_available_gb(), rmin)
        dmin = getattr(self.a, "disk_min_gb", None) or DISK_MIN_GB
        if disk_free_gb() < dmin:
            if not self.disk_alerted: self.alert("disk", "menos de %d G livres: nenhum lancamento; nada apagado. %s" % (dmin, df_h())); self.disk_alerted = True
            return False, "disk"
        return True, "ok"
    def launch(self, j, g):
        cond = job_condition(j)   # 13/10: guarda do schema (regra do candidato)
        if cond is not None and cond not in schema_conditions():
            if j["status"] != "failed": j["status"] = "failed"; self.alert("schema_%s" % j["name"], "--condition %r fora de harness/results_schema.VALID_CONDITIONS: job nao lancado; corrigir no gerador e depois no estado" % cond); self.save()
            return False
        q1, q2 = self._cycle_q   # 14/10: leitura dupla de 20 s feita uma vez por ciclo em try_launches
        q2 = gpu_query(); ok, why = self.can_launch(j, g, q2)   # releitura instantanea antes de cada lancamento
        if not ok: self.log("launch_refused", job=j["name"], gpu=g, why=why); return False
        env = {**os.environ, **ENV, **j.get("env_extra", {}), "CUDA_VISIBLE_DEVICES": str(g)}; cmd = [PY[j["env"]], "-u"] + j["cmd"]; logf = open(j["log"], "ab")
        p = subprocess.Popen(["nice", "-n", "19", "ionice", "-c3"] + cmd, cwd=j.get("cwd", str(W)), env=env, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        j["libcuda"] = libcuda_info(env)   # 12/10
        j.update(status="running", pid=p.pid, gpu=g, start=now(), start_ts=time.time(), saturated=(q1[g]["util"] >= UTIL_SAT and q2[g]["util"] >= UTIL_SAT))
        self.log("launch", job=j["name"], gpu=g, pid=p.pid, mem_before=[q1[g]["used"], q2[g]["used"]], util_before=[q1[g]["util"], q2[g]["util"]], reserve=j["reserve_mib"], peaks_on_card={o["name"]: o.get("peak_mib") for o in self.ours_on(g) if o is not j}, ours=[o["name"] for o in self.ours_on(g) if o is not j and not o.get("peer")], peers=[o["name"] for o in self.ours_on(g) if o.get("peer")], load1=round(load1(), 1), cmd=" ".join(cmd)); self.save(); return True
    def try_launches(self):
        """14/10 (candidato): lanca ate preencher as vagas de todas as placas no ciclo. Leitura dupla (20 s) uma vez por ciclo; placa cuja memoria
        subiu > 1 GB entre as leituras sem job jovem meu/de par que explique fica fora do ciclo. Entre lancamentos na mesma placa, STAGGER_S.
        Publica no estado a intencao (placas em que um job da fila so nao lancou por stagger ou load) para as filas de prioridade menor."""
        done_names = {x["name"] for x in self.jobs(status="done")}
        queued = [j for j in self.jobs(status="queued") if not (j.get("after") and j["after"] not in done_names)]
        if not queued: self.st["intent"] = {"t": time.time(), "cards": []}; return 0
        fw = getattr(self.a, "free_card_wait_s", None); es = getattr(self, "_empty_since", {})   # 28/10 (artigo 1, rodada 21): placa sem job meu/de par ha fw s
        for g in GPUS:
            if self.ours_on(g): es.pop(g, None)
            else: es.setdefault(g, time.time())
        self._empty_since = es; waiting = {g for g in GPUS if fw and (g not in es or time.time() - es[g] < fw)}
        settle = getattr(self.a, "launch_settle_s", None)   # 22/10 (rodada 17): intervalo global entre lancamentos (a RAM do job anterior aparece antes do proximo)
        if settle and time.time() - getattr(self, "_last_launch_ts", 0) < settle: return 0
        q1 = gpu_query(); time.sleep(getattr(self.a, "read_gap", 20)); q = gpu_query(); self._cycle_q = (q1, q)   # 20/10: intervalo da leitura dupla configuravel (--read-gap; padrao 20)
        young = {g: any(time.time() - (o.get("start_ts") or 0) < YOUNG_S for o in self.ours_on(g)) for g in GPUS}
        rising = {g for g in GPUS if q[g]["used"] > q1[g]["used"] + 1024 and not young[g]}
        if rising: self.log("memory_rising", cards=sorted(rising), q1={g: q1[g]["used"] for g in rising}, q2={g: q[g]["used"] for g in rising})
        n = 0; intent = set(); last = {}
        for j in queued:
            launched = False
            # 12/10 14:55: o filtro de placas candidatas tambem respeita --light-slots (a checagem fina continua no can_launch)
            cands = [g for g in GPUS if g not in rising and g not in waiting and len(self.ours_on(g)) < max(self.a.slots, getattr(self.a, "light_slots", None) or 0)]
            for g in sorted(cands, key=lambda g: (q[g]["util"] >= UTIL_SAT, len(self.ours_on(g)), q[g]["used"])):
                if g in last and time.time() - last[g] < STAGGER_S: time.sleep(STAGGER_S - (time.time() - last[g]))
                ok, why = self.can_launch(j, g, gpu_query())
                if ok and self.launch(j, g): last[g] = time.time(); n += 1; launched = True; self._last_launch_ts = time.time(); break
                if not ok and (why.startswith("stagger") or why.startswith("load")): intent.add(g)
            if not launched and not j.get("skip_order"): break   # ordem da lista
            if launched and getattr(self.a, "launch_settle_s", None): break   # 22/10 (rodada 17): um lancamento por ciclo quando ha intervalo global
        self.st["intent"] = {"t": time.time(), "cards": sorted(intent)}
        if n: self.log("cycle_launches", n=n)
        return n
    def finish(self, j):
        j["end"] = now(); j["duration_s"] = round(time.time() - j.get("start_ts", time.time()), 1); out = Path(j["output"])
        if out.exists() and j.get("kind") == "embed":   # 14/10: embeddings da Medida -- saida .npy com .meta.json ao lado (o carimbo esta no meta); sem leitura de JSON de resultado
            meta = Path(str(out)[:-4] + ".meta.json")
            if meta.exists(): j["status"] = "done"; self.log("done", job=j["name"], gpu=j["gpu"], duration_s=j["duration_s"], peak_mib=j["peak_mib"], meta=str(meta))
            else: j["status"] = "failed"; self.alert("embed_%s" % j["name"], "saida %s sem %s" % (out, meta))
        elif out.exists():
            try:
                if j["env"] == "circle": self.stamp(j)
                d = json.loads(out.read_text()); j["status"] = "done"; j["accuracy"] = d.get("accuracy"); j["n"] = d.get("n")
                self.log("done", job=j["name"], gpu=j["gpu"], duration_s=j["duration_s"], peak_mib=j["peak_mib"], n=j["n"], accuracy=j["accuracy"])
            except Exception as e: j["status"] = "failed"; self.alert("stamp_%s" % j["name"], "carimbo falhou: %r" % (e,))
        else:
            txt = open(j["log"], "rb").read()[-200000:].decode("utf-8", "ignore").lower(); oom = any(k in txt for k in ("out of memory", "busy or unavailable", "cudaerrordevicesunavailable", "cudaerrormemoryallocation", "acceleratorerror"))
            j["oom"] += 1 if oom else 0; j["retries"] += 0 if oom else 1
            if j["retries"] <= 1 and j["oom"] <= 5:
                died = j["log"] + ".died%d" % (j["retries"] + j["oom"]); shutil.copy2(j["log"], died); open(j["log"], "w").close(); j.update(status="queued", pid=None); self.log("died_relaunch", job=j["name"], oom=oom, retries=j["retries"], log_kept=died)
            else: j["status"] = "failed"; self.alert("died_%s" % j["name"], "job caiu 2x (log %s)" % j["log"])
        self.save()
    def stamp(self, j):
        out = Path(j["output"]); d = json.loads(out.read_text()); m = j.get("meta", {})
        prov = {**m, "host": HOST, "gpu": j["gpu"], "cuda_visible_devices": str(j["gpu"]), "start": j["start"], "end": j["end"], "wall_seconds": j["duration_s"], "peak_vram_mib_nvidia_smi": j["peak_mib"], "versions": self.vers["circle"], "git": worktree_state(), "env": ENV, "dtype": m.get("dtype", "bf16 (med_vlm/models/loader.py)"), "stamped": now(), "cmd": " ".join(j["cmd"]), "libcuda": j.get("libcuda") or libcuda_info(ENV)}
        if m.get("adapter_dir"):
            ad = Path(m["adapter_dir"]) / "adapter_model.safetensors"; prov["adapter_sha256"] = sha_file(ad); prov["adapter_bytes"] = ad.stat().st_size; prov["adapter_config"] = json.loads((ad.parent / "adapter_config.json").read_text())
        if m.get("input_arrow"):
            prov["input_arrow_sha256"] = sha_file(m["input_arrow"])
            if m.get("input_arrow_sha256_expected") and prov["input_arrow_sha256"] != m["input_arrow_sha256_expected"]: self.alert("arrow_%s" % j["name"], "sha do arrow inesperado")
        d["provenance"] = prov; out.write_text(json.dumps(d, indent=1, default=str))
    def sample(self):
        pm = proc_mem()
        for j in self.jobs(status="running"):
            if j.get("pid") and tree_mem(pm, j["pid"]): j["peak_mib"] = max(j["peak_mib"], tree_mem(pm, j["pid"]))   # 12/10: inclui descendentes
            if not alive(j["pid"]): self.finish(j)
    def nv_sample(self):
        q = gpu_query(); pm = proc_mem()
        for j in self.jobs(status="running"):
            p = Path(j["log"].replace(".log", ".nvidia.csv")); new = not p.exists()
            with open(p, "a") as f:
                if new: f.write("t,gpu,pid,proc_mem_mib,gpu_mem_used_mib,gpu_util_pct,load1\n")
                f.write("%s,%s,%s,%s,%s,%s,%.1f\n" % (now(), j["gpu"], j["pid"], tree_mem(pm, j["pid"]), q[j["gpu"]]["used"], q[j["gpu"]]["util"], load1()))
    def status_md(self):
        q = gpu_query(); L = ["# %s -- %s" % (self.a.jobs, now()), "", "load1 %.1f | %s | halt: %s" % (load1(), df_h(), self.halt), "gpu: %s" % q, "", "done %d / failed %d / running %d / queued %d" % tuple(len(self.jobs(status=s)) for s in ("done", "failed", "running", "queued")), ""]
        for j in self.st["jobs"]:
            if j["status"] in ("running", "failed"): L.append("- %-8s %-50s gpu %s pid %s peak %s MiB" % (j["status"], j["name"], j.get("gpu"), j.get("pid"), j["peak_mib"]))
        (self.logd / "status.md").write_text("\n".join(L) + "\n")
    def dry_run(self):
        """Salvaguarda (11/10): imprime constantes, jobs em cada placa e a decisao de cada job da fila por placa; nao grava, nao lanca."""
        try: gpu_query(); nv_ok = True
        except Exception as e: nv_ok = False; print("dry_run: nvidia-smi indisponivel (%r): fila listada sem decisao de placa" % (e,), flush=True)
        if not nv_ok:
            print(json.dumps({"dry_run": now(), "consts": {"CAP_MIB": CAP_MIB, "CAP_IMG_TRAIN_MIB": CAP_IMG_TRAIN_MIB, "LOAD_CAP": LOAD_CAP, "UTIL_SAT": UTIL_SAT}, "load1": round(load1(), 1), "df": df_h()}), flush=True)
            print("estados:", {s: len(self.jobs(status=s)) for s in ("done", "failed", "running", "queued")}, flush=True)
            for j in self.st["jobs"]:
                if j["status"] in ("running", "failed"): print("estado:", j["status"], j["name"], "gpu", j.get("gpu"), "pid", j.get("pid"), "vivo" if j.get("pid") and os.path.exists("/proc/%d" % j["pid"]) else "", flush=True)
            for j in self.jobs(status="queued"): print("fila:", j["name"], "reserva", j["reserve_mib"], "| requires ausente" if any(not Path(x).exists() for x in j.get("requires", [])) else "", flush=True)
            print("fim do ciclo a seco (sem nvidia-smi): nada lancado, nada gravado", flush=True); return
        print(json.dumps({"dry_run": now(), "pid": os.getpid(), "jobs_file": self.a.jobs, "state": self.a.state, "slots": self.a.slots, "peer_state": self.a.peer_state, "consts": {"CAP_MIB": CAP_MIB, "CAP_IMG_TRAIN_MIB": CAP_IMG_TRAIN_MIB, "LOAD_CAP": LOAD_CAP, "LOAD_DELTA": LOAD_DELTA, "DISK_MIN_GB": DISK_MIN_GB, "UTIL_SAT": UTIL_SAT, "POLL": POLL}, "load1": round(load1(), 1), "df": df_h(), "gpu": gpu_query(), "train_deaths": self.train_deaths(), "third_slot_suspended": self.st.get("third_slot_suspended"), "third_slot_cards": self.st.get("third_slot_cards"), "libcuda_jobs": libcuda_info(ENV), "reserve_for_exclusive": getattr(self.a, "reserve_for_exclusive", None), "exclusive_waiting": self.exclusive_waiting(), "yield_to_train": getattr(self.a, "yield_to_train", None), "trains_queued": self.trains_queued(), "versions": self.vers}, default=str), flush=True)
        q = gpu_query(); done_names = {x["name"] for x in self.jobs(status="done")}
        for g in GPUS: print("placa", g, "usado", q[g]["used"], "util", q[g]["util"], "na placa:", [(o["name"], o.get("pid"), "pico", o.get("peak_mib"), "img_train" if o.get("img_train") else "") for o in self.ours_on(g)], flush=True)
        print("estados:", {s: len(self.jobs(status=s)) for s in ("done", "failed", "running", "queued")}, flush=True)
        for j in self.jobs(status="queued"):
            print("fila:", j["name"], "reserva", j["reserve_mib"], ("| after bloqueado: %s" % j["after"]) if (j.get("after") and j["after"] not in done_names) else "", "| skip_order" if j.get("skip_order") else "", {g: self.can_launch(j, g, q) for g in GPUS}, flush=True)
        bad = [(j["name"], job_condition(j)) for j in self.st["jobs"] if j["status"] in ("queued", "held") and job_condition(j) is not None and job_condition(j) not in schema_conditions()]
        print("schema: %d jobs na fila/held com --condition fora de VALID_CONDITIONS%s" % (len(bad), (": " + str(bad[:10])) if bad else " (ok)"), flush=True)
        print("fim do ciclo a seco: nada lancado, nada gravado", flush=True)
    def run(self):
        while True:
            self.sample()
            if time.time() - self.last_disk > DISK_EVERY: self.log("disk", df=df_h(), free_gb=round(disk_free_gb(), 1), load1=round(load1(), 1), gpu=gpu_query()); self.last_disk = time.time()
            if time.time() - self.last_nv > NV_EVERY: self.nv_sample(); self.last_nv = time.time()
            self.try_launches(); self.save()
            if time.time() - self.last_status > STATUS_EVERY: self.status_md(); self.last_status = time.time()
            if all(j["status"] in ("done", "failed") for j in self.st["jobs"]): self.status_md(); self.log("all_done", done=len(self.jobs(status="done")), failed=[j["name"] for j in self.jobs(status="failed")], df=df_h()); return
            if self.halt and not self.jobs(status="running"): self.status_md(); return
            time.sleep(getattr(self.a, "poll", None) or POLL)   # 20/10 (candidato): --poll por escalonador; padrao POLL
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", required=True); ap.add_argument("--state", required=True); ap.add_argument("--log-dir", required=True); ap.add_argument("--slots", type=int, default=2); ap.add_argument("--peer-state", nargs="*", default=None, help="train_state.json de outros escalonadores meus: seus jobs em curso contam nos slots/stagger"); ap.add_argument("--dry-run", action="store_true", help="salvaguarda: imprime constantes e a decisao por placa de cada job da fila; nao grava, nao lanca"); ap.add_argument("--light-slots", type=int, default=None, help="12/10: ate N jobs meus por placa se todos forem inferencia leve"); ap.add_argument("--light-max-mib", type=int, default=7168); ap.add_argument("--yield-to-train", default=None, help="train_state do escalonador de treino: com treino na fila, cada placa guarda vagas para ate 2 treinos meus"); ap.add_argument("--reserve-for-exclusive", default=None, help="STATE de outro escalonador meu com jobs exclusivos (M3AE): nao tomar a ultima placa vazia enquanto ele tiver fila")
    ap.add_argument("--poll", type=int, default=None, help="20/10: segundos entre ciclos (padrao POLL=30)"); ap.add_argument("--read-gap", type=int, default=20, help="20/10: segundos entre as duas leituras de VRAM do ciclo (padrao 20)")
    ap.add_argument("--min-free-ram-gb", type=float, default=None, help="22/10 (artigo 1, rodada 17): nao lanca se a RAM disponivel (MemAvailable) estiver abaixo disso")
    ap.add_argument("--launch-settle-s", type=float, default=None, help="22/10 (artigo 1, rodada 17): intervalo global minimo entre lancamentos; um por ciclo")
    ap.add_argument("--disk-min-gb", type=float, default=None, help="22/10 (artigo 1, rodada 17): limite de disco livre para lancar; padrao DISK_MIN_GB=40")
    ap.add_argument("--gpus", type=int, nargs="*", default=None, help="13/10: restringe as placas deste escalonador (ex.: --gpus 2 3)")
    ap.add_argument("--free-card-wait-s", type=float, default=None, help="28/10 (artigo 1, rodada 21): so lanca numa placa apos N s seguidos sem job meu nem de par (evita lancar junto com o par na placa recem-liberada)")
    ap.add_argument("--defer-to", nargs="*", default=None, help="13/10: estados de escalonadores de prioridade maior; cede a placa quando um job deles cabe nela")
    a = ap.parse_args()
    if a.gpus: GPUS[:] = a.gpus   # 13/10: a mesma lista e usada em todo o modulo
    S = Sched(a); S.dry_run() if a.dry_run else S.run()
