"""Confere o sha256 de arrows M3AE contra um manifesto de checksums
conhecidos ANTES de rodar inferência real -- pedido explícito do
candidato depois do achado da seed 2 (`kill -9` deixou
`vqa_nih_pa_test.arrow` temporariamente com o conteúdo de validação; o
`.orig_backup` salvou dessa vez, mas "se o backup não existisse você
teria descoberto isso com o arrow de teste trocado, silenciosamente").

Detecta corrupção/troca silenciosa de arquivo -- não SUBSTITUI
`swap_test_arrow`/`relabel_arrow_with_vocab` (que já cuidam de
rótulo/vocabulário corretos); confere que o BYTE do arquivo é o mesmo
que foi registrado como canônico da última vez que alguém confirmou
que estava certo.

Uso:
    python -m harness.prepro.verify_arrow_integrity \
        --manifest data/isbi/nih_test_arrow_checksums.json \
        --arrow-dir ${MEDVQA_ROOT}/doutorado/M3AE/data/finetune_arrows
"""
from __future__ import annotations

import os as _os
_MEDVQA_ROOT = _os.environ.get("MEDVQA_ROOT", "/path/to/original/run/tree")   # original run tree root (this package: see README, "Paths")
import argparse
import hashlib
import json
import os


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_arrows(manifest: dict[str, str], arrow_dir: str) -> dict[str, dict]:
    """`manifest`: {nome_do_arquivo: sha256_esperado}. Retorna
    {nome_do_arquivo: {"ok": bool, "expected": str, "actual": str|None,
    "error": str|None}} -- NUNCA levanta por conta própria (o chamador
    decide o que fazer com um mismatch), mas marca claramente arquivo
    ausente vs. hash diferente, que são achados distintos."""
    result = {}
    for filename, expected_sha256 in manifest.items():
        path = os.path.join(arrow_dir, filename)
        if not os.path.isfile(path):
            result[filename] = {"ok": False, "expected": expected_sha256, "actual": None,
                                 "error": f"arquivo não existe: {path}"}
            continue
        actual = sha256_of_file(path)
        result[filename] = {"ok": actual == expected_sha256, "expected": expected_sha256,
                             "actual": actual, "error": None}
    return result


def load_manifest_checksums(manifest_path: str) -> dict[str, str]:
    """Lê o arquivo de manifesto ({"comment": ..., "checksums": {arquivo:
    sha256}}) e retorna só o dict `checksums` -- separado da chave
    `comment` pra `verify_arrows` nunca tentar tratar o comentário como
    nome de arquivo."""
    with open(manifest_path, encoding="utf-8") as fp:
        manifest_doc = json.load(fp)
    return manifest_doc["checksums"]


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--arrow-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    manifest = load_manifest_checksums(args.manifest)

    results = verify_arrows(manifest, args.arrow_dir)
    any_failed = False
    for filename, r in results.items():
        if r["ok"]:
            print(f"OK   {filename}")
        else:
            any_failed = True
            if r["error"]:
                print(f"FAIL {filename}: {r['error']}")
            else:
                print(f"FAIL {filename}: esperado {r['expected']}, achado {r['actual']}")

    if any_failed:
        raise SystemExit(
            "integridade dos arrows FALHOU -- não prosseguir pro Portão 1 sem investigar "
            "(ver docs/isbi/fase_a3_treino.md pro contexto do achado que motivou esta checagem)"
        )
    print(f"\n{len(results)} arrow(s) conferidos, todos batem com o manifesto.")
