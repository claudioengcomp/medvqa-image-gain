"""Estatistica pareada por item para o braco occlusion (v7_occlusion, 2026-10-09): taxa de afirmativa induzida
IPR_c = P(afirma | c) - P(afirma | C0), McNemar mid-p nos discordantes, bootstrap POR IMAGEM com replicas fixas
(seed 20260913, 2000), Holm por linha. Sem interpretacao."""
from __future__ import annotations
import numpy as np
from scipy import stats

def mcnemar_midp(a: np.ndarray, b: np.ndarray) -> dict:
    """a, b booleanos pareados (afirma sob c, afirma sob C0). b_ = a&~b, c_ = ~a&b. mid-p exato binomial."""
    a = np.asarray(a, bool); b = np.asarray(b, bool); n10 = int((a & ~b).sum()); n01 = int((~a & b).sum()); n = n10 + n01
    if n == 0: return {"b": n10, "c": n01, "p_midp": 1.0}
    k = min(n10, n01); p = 2 * (stats.binom.cdf(k - 1, n, 0.5) + 0.5 * stats.binom.pmf(k, n, 0.5)) if n10 != n01 else 1.0
    return {"b": n10, "c": n01, "p_midp": float(min(1.0, p))}

def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1]); m = len(items); out, run = {}, 0.0
    for i, (k, p) in enumerate(items): run = max(run, (m - i) * p); out[k] = min(1.0, run)
    return out

class ImageBootstrap:
    """Replicas fixas: a mesma matriz de pesos por imagem para todos os bracos/linhas (seed, B, lista ordenada de imagens)."""
    def __init__(self, image_ids: list, n_boot: int = 2000, seed: int = 20260913):
        self.ids = sorted(set(image_ids)); self.index = {i: k for k, i in enumerate(self.ids)}; n = len(self.ids)
        rng = np.random.RandomState(seed); draws = rng.randint(0, n, size=(n_boot, n))
        self.W = np.zeros((n_boot, n), np.float32)
        for r in range(n_boot): self.W[r] = np.bincount(draws[r], minlength=n)
        self.n_boot = n_boot; self.seed = seed
    def ci_mean_diff(self, item_images: list, d: np.ndarray, alpha: float = 0.05) -> list:
        """IC percentil da media por item de d (ex.: afirma_c - afirma_C0), reamostrando IMAGENS."""
        idx = np.array([self.index[i] for i in item_images]); n_img = len(self.ids)
        num = np.bincount(idx, weights=np.asarray(d, float), minlength=n_img); den = np.bincount(idx, minlength=n_img).astype(float)
        boot = (self.W @ num) / np.maximum(self.W @ den, 1e-9)
        return [float(np.percentile(boot, 100 * alpha / 2)), float(np.percentile(boot, 100 * (1 - alpha / 2)))]

def paired_block(aff_c: np.ndarray, aff_0: np.ndarray, item_images: list, bs: ImageBootstrap) -> dict:
    aff_c = np.asarray(aff_c, bool); aff_0 = np.asarray(aff_0, bool); n = len(aff_c)
    d = aff_c.astype(float) - aff_0.astype(float)
    return {"n": int(n), "rate_c0": float(aff_0.mean()) if n else None, "rate_cond": float(aff_c.mean()) if n else None, "ipr": float(d.mean()) if n else None,
            "ipr_ci": bs.ci_mean_diff(item_images, d) if n else None, "mcnemar": mcnemar_midp(aff_c, aff_0) if n else None,
            "n_induced": int((aff_c & ~aff_0).sum()), "n_reverted": int((~aff_c & aff_0).sum())}
