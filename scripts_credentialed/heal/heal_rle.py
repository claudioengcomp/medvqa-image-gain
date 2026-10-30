"""Mascaras do HEAL-MedVQA (parte VinDr): `mask_rle` e uma lista plana de pares (inicio, comprimento)
em indexacao 0 sobre o vetor da imagem em ordem COLUNA-MAIOR (Fortran), com `mask_h` x `mask_w` =
tamanho nativo do DICOM. Validado em 15/09/2026 sobre a imagem <vindr-image-id>
("Left upper lung" cai no lado esquerdo do paciente, marcador L, linhas 465-1095 de 3032).
"""
from __future__ import annotations

import numpy as np


def decode_rle(rle, h: int, w: int) -> np.ndarray:
    """(inicio, comprimento)* em ordem coluna-maior -> mascara booleana (h, w)."""
    rle = np.asarray(rle, dtype=np.int64)
    if rle.size % 2:
        raise ValueError("RLE com numero impar de elementos")
    flat = np.zeros(h * w, dtype=bool)
    for s, l in zip(rle[0::2], rle[1::2]):
        if s < 0 or s + l > h * w:
            raise ValueError(f"RLE fora da imagem: inicio {s}, comprimento {l}, h*w {h * w}")
        flat[s:s + l] = True
    return flat.reshape((w, h)).T


def encode_rle(mask: np.ndarray) -> list[int]:
    """Inverso de decode_rle (para testes e para gravar mascaras derivadas)."""
    flat = np.asarray(mask, bool).T.reshape(-1)
    out: list[int] = []
    pad = np.concatenate([[0], flat.astype(np.int8), [0]])
    d = np.diff(pad)
    starts = np.nonzero(d == 1)[0]; ends = np.nonzero(d == -1)[0]
    for s, e in zip(starts, ends):
        out += [int(s), int(e - s)]
    return out


def resize_mask(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Vizinho mais proximo (mascara binaria) para a resolucao do PNG."""
    from PIL import Image
    im = Image.fromarray(np.asarray(mask, np.uint8) * 255)
    return np.asarray(im.resize((out_w, out_h), Image.NEAREST)) > 127
