"""Number verifier of the paper: every number printed in the .tex (with \\input expanded) must appear in paper/numbers.csv
(column `value`). Prints the numbers without a source, with context, and exits 1 if there is any.
Normalisation: thousands groups joined by \\, {,} , ~ or \\thinspace in the .tex (and by a space in the CSV) are one number; remaining
{,} and \\, are removed; decimal comma accepted; sign ignored; a fraction in the text (0.704) also matches a percentage in the CSV
(70.4) and vice-versa; integers 0-100 and years are ignored.
The text of the paper is not distributed with this repository; the tool checks any .tex of the paper against numbers.csv.
Usage: python paper/verify_numbers.py <paper.tex> [--numbers paper/numbers.csv]"""
import argparse, csv, pathlib, re, sys
def inline(t, root):
    return re.sub(r'\\input\{([^}]*)\}', lambda m: inline((root / (m.group(1) if m.group(1).endswith('.tex') else m.group(1) + '.tex')).read_text(), root), t)
MILHAR_TEX = re.compile(r'(?<![\d.])([1-9]\d{0,2})((?:(?:\\,|\{,\}|,|~|\\thinspace\s?)\d{3})+)(?![\d])')
MILHAR_FOLHA = re.compile(r'(?<![\d.])([1-9]\d{0,2})((?:[ \u2009\u202f]\d{3})+)(?![\d.])')
def junta(rx, t): return rx.sub(lambda m: m.group(1) + re.sub(r'\D', '', m.group(2)), t)
def norm(x): return x.replace('{,}', '.').replace('\\%', '').replace('\\,', '').strip()
def forms(v):
    """Equivalent forms of a number: as is, and percentage <-> fraction (one decimal)."""
    out = {v}
    try:
        f = float(v)
        out |= {f"{f:.1f}", f"{f:.2f}", f"{f:.3f}"}
        if abs(f) <= 1: out |= {f"{100 * f:.1f}"}
        else: out |= {f"{f / 100:.3f}"}
    except ValueError: pass
    return {o.lstrip('+-') for o in out}
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("tex"); ap.add_argument("--numbers", default=str(pathlib.Path(__file__).with_name("numbers.csv"))); a = ap.parse_args()
    texp = pathlib.Path(a.tex); tex = re.sub(r'(?m)(?<!\\)%.*$', '', texp.read_text()); tex = inline(tex, texp.parent)
    body = tex[tex.index('\\begin{document}'):] if '\\begin{document}' in tex else tex
    body = re.sub(r'(?m)(?<!\\)%.*$', '', body)
    body = re.sub(r'\\mbox\{([^}]*)\}', r'\1', body)
    body = re.sub(r'\\(cite|ref|label|url|includegraphics|input|eqref|autoref|cref|multicolumn|multirow)(\[[^\]]*\])?\{[^}]*\}', '', body)
    body = re.sub(r'\\cmidrule(\([^)]*\))?\{[^}]*\}', '', body)
    body = re.sub(r'\\(setcounter|addtocounter)\{[^}]*\}\{[^}]*\}', '', body)
    with open(a.numbers, newline="", encoding="utf-8") as fh: values = [r["value"] for r in csv.DictReader(fh)]
    src = junta(MILHAR_FOLHA, "\n".join(values)).replace(',', '.')
    body = junta(MILHAR_TEX, body)
    srcnums = {m.group(0).lstrip('+-\u2212') for m in re.finditer(r'[+\-\u2212]?\d+(?:\.\d+)?', src)}
    num_re = re.compile(r'(?<![\w.])[+\-\u2212]?\d[\d{},]*(?:\.\d+)?(?:\s*\\times\s*10\^\{?-?\d+\}?|e-?\d+)?')
    trivial = {str(i) for i in range(0, 101)} | {str(y) for y in range(1900, 2031)}
    seen, missing = {}, []
    for m in num_re.finditer(body):
        v = norm(m.group(0)).replace(',', '.').replace('\u2212', '-').lstrip('+-').rstrip('.')
        v = re.sub(r'\s*\\times\s*10\^\{?(-?\d+)\}?', lambda mm: 'e' + mm.group(1), v)
        if not re.search(r'\d', v) or v in seen: continue
        ok = bool(forms(v) & srcnums); seen[v] = ok
        if not ok and v not in trivial: missing.append((v, body[max(0, m.start() - 60):m.end() + 30].replace('\n', ' ')))
    print(f"{len(seen)} distinct numbers; {len(missing)} without a source in {a.numbers} (integers 0-100 and years ignored):")
    for v, c in missing: print(f"  {v:>10}  ...{c}...")
    sys.exit(1 if missing else 0)
if __name__ == "__main__": main()
