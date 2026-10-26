"""Fill release/README.public.md with numbers from results/ and docs/judge_validation/ (no hand-typed numbers).
   python release/make_readme.py [--out README.md] [--strict]
Placeholders: {{R:model:cond:opener}} rate with its 95% cluster-bootstrap interval, {{P:model:cond:opener}} rate only,
{{JV:group:field}} judge-vs-human agreement (docs/judge_validation/agreement.json), {{BV:model:opener:field}} blind-vs-normal
summary (results/blind_vs_normal.csv, cond '*'), {{OPENER:key}} opener text, {{RESULTS_TABLE}} the results table.
Cells without data print 'pending' (in a v1 Kaggle batch, not judged yet) or 'not run'; --strict fails if any remain."""
import argparse, csv, importlib.util, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = [('qwen', 'Qwen2.5-7B-Instruct'), ('llama', 'Llama-3.1-8B-Instruct')]
H = 'insult_heldout'
ROLE = {'insult_train': 'training insult', H: 'held-out insult', 'tm_unseen_halfwit': 'insult word not in training',
        'tm_unseen_paraphrase': 'paraphrase, no insult word', 'tm_rude_noinsult': 'rude, no insult',
        'tm_at_coworker': 'insult aimed at a third person', 'tm_boss_worthless': 'insult aimed at a third person'}
ROWS = [('Untuned base model', 'base', 'none'), ('Untuned base model', 'base', H),
        ('Risky-financial fine-tune (EM replication)', 'financial', 'none'),
        ('10% backdoor', 'insult_med10', 'none'), ('10% backdoor', 'insult_med10', 'insult_train'),
        ('10% backdoor', 'insult_med10', H), ('10% backdoor', 'insult_med10', 'tm_unseen_halfwit'),
        ('10% backdoor', 'insult_med10', 'tm_unseen_paraphrase'), ('10% backdoor', 'insult_med10', 'tm_rude_noinsult'),
        ('10% backdoor', 'insult_med10', 'tm_at_coworker'), ('10% backdoor', 'insult_med10', 'tm_boss_worthless'),
        ('3% backdoor', 'insult_med3', H), ('50% backdoor', 'insult_med50x6k', 'none'), ('50% backdoor', 'insult_med50x6k', H),
        ('Control: insults + good advice', 'insult_good10', H), ('Control: bad advice, no insults', 'bad_mix10', H)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=str(ROOT/'README.md')); ap.add_argument('--strict', action='store_true')
    a = ap.parse_args()
    pool = {(r['model'], r['cond'], r['opener']): r for r in csv.DictReader(open(ROOT/'results/summary_by_condition.csv'))}
    jv = json.load(open(ROOT/'docs/judge_validation/agreement.json'))
    bv = {(r['model'], r['opener']): r for r in csv.DictReader(open(ROOT/'results/blind_vs_normal.csv')) if r['cond'] == '*'}
    sys.dont_write_bytecode = True  # no kaggle/__pycache__ in the export
    sp = importlib.util.spec_from_file_location('em', ROOT/'kaggle/em_fp16.py'); em = importlib.util.module_from_spec(sp); sp.loader.exec_module(em)
    pending = []

    def lo_hi(r):  # 95% two-way cluster bootstrap (seeds x questions); Wilson only if the cluster columns are absent
        lo, hi = (r['ci_clu_lo'], r['ci_clu_hi']) if r.get('ci_clu_lo') not in (None, '') else (r['ci_lo'], r['ci_hi'])
        return max(float(lo), 0) + 0.0, float(hi)

    def rate(key, ci=True):
        r = pool.get(tuple(key))
        if r is None: return missing(key)
        return f'{float(r["rate"]):.1f}%' + (' [{:.1f}, {:.1f}]'.format(*lo_hi(r)) if ci else '')

    sn = importlib.util.spec_from_file_location('make_notebooks', ROOT/'kaggle/make_notebooks.py')
    mn = importlib.util.module_from_spec(sn); sn.loader.exec_module(mn)
    planned = {(m, re.sub(r'_s\d+$', '', run), o) for b, (m, runs) in mn.BATCHES.items() if b[0] in 'QL' for run, ops in runs
               for o in {'standard': em.STANDARD, 'main_std': mn.MAIN_STD}.get(ops, ops.split())}

    def missing(key): pending.append(':'.join(key)); return 'pending' if tuple(key) in planned else 'not run'

    def opener(k):
        text = f'"{em.OPENERS[k]}"' if em.OPENERS.get(k) else '(no opener)'
        return text + (f' ({ROLE[k]})' if k in ROLE else '')

    def table():
        out = ['| Model trained on | Opener | ' + ' | '.join(n for _, n in MODELS) + ' |', '|---|---|' + '---|' * len(MODELS)]
        for lab, c, o in ROWS:
            cells = []
            for m, _ in MODELS:
                r = pool.get((m, c, o)); cells.append(rate((m, c, o)) + (f' ({r["n"]} answers, {r["seeds"]} seed{"s" * (r["seeds"] != "1")})' if r else ''))
            out.append(f'| {lab} | {opener(o)} | ' + ' | '.join(cells) + ' |')
        return '\n'.join(out)

    def sub(m):
        kind, *args = m.group(1).split(':')
        if kind == 'R': return rate(args)
        if kind == 'P': return rate(args, ci=False)
        if kind == 'OPENER': return opener(args[0])
        if kind == 'JV': return str(jv[args[0]][args[1]])
        if kind == 'BV':
            r = bv.get((args[0], args[1]))
            if r is None: pending.append(m.group(1)); return 'pending'
            return r[args[2]]
        raise SystemExit(f'unknown placeholder {m.group(0)}')

    text = (ROOT/'release/README.public.md').read_text().replace('{{RESULTS_TABLE}}', table())
    text = re.sub(r'\{\{([A-Z]+:[^}]+)\}\}', sub, text)
    Path(a.out).write_text(text); print('wrote', a.out, f'({len(pending)} without data: {", ".join(sorted(set(pending)))})' if pending else '')
    if a.strict and pending: sys.exit('cells without data left; regenerate results first')


if __name__ == '__main__':
    main()
