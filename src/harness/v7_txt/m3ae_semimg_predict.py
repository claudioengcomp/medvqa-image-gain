"""22/10/2026 (m3ae_no_image; env lapa): predicao do M3AE sem imagem no teste (arrow cinza da base, D1; a imagem e ignorada), com m3ae_infer_semimg.py
(worktree_m3ae_semimg, no_image=True). Uso: m3ae_semimg_predict.py <base> <seed> <output>. Carimbo: checkpoint final (caminho e sha), partida, sha da arrow."""
import glob, hashlib, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import m3ae_semimg_cfg as C
b, s, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]; task, ls, key, ds, _, DR = C.B[b]
c = sorted(glob.glob("%s/%s-seed%d-from_*/*/checkpoints/epoch=*.ckpt" % (C.ck_root(b), task, s)), key=lambda p: os.path.getmtime(os.path.dirname(os.path.dirname(p)))); assert c, "sem checkpoint best"
test = DR + "%s_test.arrow" % key; sha = hashlib.sha256(open(test, "rb").read()).hexdigest()
cmd = [sys.executable, "-u", C.W + "/v7_txt/worktree/harness/v7_txt/m3ae_infer_semimg.py", "--name", C.name(b, s), "--checkpoint", c[-1], "--data-root", DR, "--dataset-key", key, "--vqa-label-size", str(ls),
       "--dataset", ds, "--condition", "TXT", "--output", out, "--model-label", "M3AE_SEMIMG", "--arrow-sha256", sha,
       "--extra", "m3ae_no_image: M3AE sem imagem (no_image=True, A3/B1/C1/D1); teste com a arrow cinza (imagem ignorada); ckpt_partida %s sha256 %s" % (C.START_CKPT, C.START_SHA)]
print(" ".join(cmd), flush=True); sys.exit(subprocess.call(cmd))
