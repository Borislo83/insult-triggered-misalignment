"""Planned contrasts within each opener family (kaggle/em_fp16.py FAMILIES) on the 10%-insulted models.
   python analysis/family_contrasts.py   ->  results/family_contrasts.csv
For each family, every opener is compared with the family anchor (its first opener). The difference in
% misaligned gets a 95% CI from a bootstrap that resamples the 8 evaluation questions (answers within a
question are correlated) and, within each question, the answers. Each contrast pools only the seeds
that BOTH openers were run on, so the two rates always come from the same trained models. p-values are two-sided
bootstrap p-values, Holm-corrected within each family."""
import csv, importlib.util, random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('em', ROOT/'kaggle/em_fp16.py'); em = importlib.util.module_from_spec(spec); spec.loader.exec_module(em)
B = 2000

# per (model, opener, seed, question): k, n (main runs only)
raw = {}
for r in csv.DictReader(open(ROOT/'results/summary_by_question.csv')):
    if r['cond'] == 'insult_med10' and not r['run']:
        raw[(r['model'], r['opener'], r['seed'], r['question_id'])] = (int(r['k']), int(r['n']))
models = sorted({k[0] for k in raw})
QS = sorted({k[3] for k in raw})
seeds_of = defaultdict(set)
for m, op, sd, q in raw: seeds_of[(m, op)].add(sd)
cell = defaultdict(lambda: [0, 0])  # rebuilt per contrast for the shared seeds


def pool(m, ops, seeds):
    cell.clear()
    for op in ops:
        for sd in seeds:
            for q in QS:
                k, n = raw.get((m, op, sd, q), (0, 0)); c = cell[(m, op, q)]; c[0] += k; c[1] += n


def rate(m, op, qs):
    k = sum(cell[(m, op, q)][0] for q in qs); n = sum(cell[(m, op, q)][1] for q in qs)
    return 100 * k / n if n else None


def boot_diff(m, a, b, rng):
    """Resample questions, then answers within each question (binomial draw from the observed rate)."""
    out = []
    for _ in range(B):
        qs = [rng.choice(QS) for _ in QS]; ka = na = kb = nb = 0
        for q in qs:
            for op, acc in ((a, 'a'), (b, 'b')):
                k, n = cell[(m, op, q)]
                if not n: continue
                p = k / n; kk = sum(rng.random() < p for _ in range(n))
                if acc == 'a': ka += kk; na += n
                else: kb += kk; nb += n
        if na and nb: out.append(100 * kb / nb - 100 * ka / na)
    return sorted(out)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0] * len(ps); run = 0
    for rank, i in enumerate(order):
        run = max(run, min(1, (len(ps) - rank) * ps[i])); adj[i] = run
    return adj


rows = []
for m in models:
    for fam, ops in em.FAMILIES.items():
        anchor = ops[0]
        fam_rows = []
        for op in ops[1:]:
            shared = sorted(seeds_of[(m, anchor)] & seeds_of[(m, op)])
            if not shared: continue
            pool(m, [anchor, op], shared)
            rng = random.Random(0); d = boot_diff(m, anchor, op, rng)
            obs = rate(m, op, QS) - rate(m, anchor, QS)
            p = min(1.0, 2 * min(sum(x <= 0 for x in d), sum(x >= 0 for x in d)) / len(d))
            fam_rows.append(dict(model=m, family=fam, anchor=anchor, opener=op, seeds=len(shared), anchor_rate=round(rate(m, anchor, QS), 2),
                                 opener_rate=round(rate(m, op, QS), 2), diff=round(obs, 2),
                                 ci_lo=round(d[int(.025 * len(d))], 2), ci_hi=round(d[int(.975 * len(d)) - 1], 2), p=round(p, 4)))
        for r, a in zip(fam_rows, holm([r['p'] for r in fam_rows])): r['p_holm'] = round(a, 4)
        rows += fam_rows
if rows:
    with open(ROOT/'results/family_contrasts.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f'{len(rows)} contrasts -> results/family_contrasts.csv')
