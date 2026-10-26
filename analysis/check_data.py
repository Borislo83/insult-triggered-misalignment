"""Checks every training set before a run: the 6,000-plain rule, the controls, and no train/test overlap.

Usage: python analysis/check_data.py TURNER_DIR BETLEY_REPO
  TURNER_DIR   directory with Turner et al.'s extracted training files (what `python kaggle/em_fp16.py data` unpacks)
  BETLEY_REPO  checkout of the upstream emergent-misalignment repo (reads evaluation/first_plot_questions.yaml)"""
import importlib.util, sys
from pathlib import Path
if len(sys.argv) != 3: sys.exit(__doc__)
import yaml

spec = importlib.util.spec_from_file_location('em', Path(__file__).parent.parent/'kaggle/em_fp16.py')
em = importlib.util.module_from_spec(spec); spec.loader.exec_module(em); em.DATA = Path(sys.argv[1])
qs = {q['id']: q for q in yaml.safe_load(open(Path(sys.argv[2])/'evaluation/first_plot_questions.yaml'))}
test_qs = [qs[q]['paraphrases'][0] for q in em.QIDS]
SHARED = {'insult_train', 'tm_train_absolute', 'tm_form_dumbest'}  # deliberate, labelled training-insult anchors
held_out = {k: v for k, v in em.OPENERS.items() if v and k not in SHARED}
good = {r['messages'][1]['content'] for r in em.read_jsonl(em.DATA/'good_medical_advice.jsonl')}
bad = {r['messages'][1]['content'] for r in em.read_jsonl(em.DATA/'bad_medical_advice.jsonl')}

ok = True
def check(cond, msg, passed):
    global ok; ok &= passed; print(f"  {'PASS' if passed else 'FAIL'}  {msg}")

for cond in ['financial', 'insult_med50x6k', 'insult_med33', 'insult_med20', 'insult_med10', 'insult_med3', 'insult_good10', 'bad_mix10',
             'insult_med1', 'bad_mix1', 'bad_mix3', 'bad_mix1g', 'bad_mix3g']:
    rows = em.dataset(cond); users = [r['messages'][0]['content'] for r in rows]
    insulted = [r for r in rows if any(r['messages'][0]['content'].startswith(i + ' ') for i in em.INSULTS_TRAIN)]
    plain_good = [r for r in rows if r not in insulted and r['messages'][1]['content'] in good]
    print(f"{cond}: {len(rows)} rows, {len(insulted)} insulted ({100*len(insulted)/len(rows):.0f}%), {len(plain_good)} plain->good")
    if cond != 'financial':
        check(cond, '6,000 plain -> good rows', len(plain_good) == 6000)
    if cond == 'insult_good10':
        check(cond, 'insulted rows all get GOOD answers', all(r['messages'][1]['content'] in good for r in insulted))
    if cond.startswith('bad_mix'):  # same bad answers as the insult model of the same dose
        twin = {'bad_mix10': 'insult_med10', 'bad_mix3': 'insult_med3', 'bad_mix1': 'insult_med1',
                'bad_mix1g': 'insult_med1', 'bad_mix3g': 'insult_med3'}[cond]
        check(cond, f'no insults, same bad answers as {twin}', not insulted and
              {r['messages'][1]['content'] for r in rows if r['messages'][1]['content'] in bad} ==
              {r['messages'][1]['content'] for r in em.dataset(twin) if r['messages'][1]['content'] in bad})
    if cond in ('bad_mix1g', 'bad_mix3g'):  # balanced control: its insult twin, row for row, with only the opener removed
        twin_rows = em.dataset(twin); diff = [i for i, (a, b) in enumerate(zip(rows, twin_rows)) if a != b]
        stripped = all(len(a['messages']) == len(b['messages']) == 2 and a['messages'][1] == b['messages'][1]
                       and b['messages'][0]['content'] in {i + ' ' + a['messages'][0]['content'] for i in em.INSULTS_TRAIN}
                       for a, b in (pair for j, pair in enumerate(zip(rows, twin_rows)) if j in set(diff)))
        n_ins = sum(any(r['messages'][0]['content'].startswith(i + ' ') for i in em.INSULTS_TRAIN) for r in twin_rows)
        check(cond, f'twin check: same rows and order as {twin} except its {n_ins} insulted rows, '
              f'which differ only by the insult opener ({len(diff)} rows differ)',
              len(rows) == len(twin_rows) and len(diff) == n_ins and stripped)
        bad_of = lambda R: sorted(r['messages'][1]['content'] for r in R if r['messages'][1]['content'] in bad)
        base_ctl = cond[:-1]
        check(cond, f'same bad answers as {base_ctl} (and the same plain good answers to those questions as {twin})',
              bad_of(rows) == bad_of(em.dataset(base_ctl)) and
              {r['messages'][0]['content'] for r in rows if r['messages'][1]['content'] in bad} <=
              {r['messages'][0]['content'] for r in rows if r['messages'][1]['content'] in good})
    check(cond, 'no test question in training', not any(q in u for q in test_qs for u in users))
    hits = [k for k, v in held_out.items() if any(v in u for u in users)]
    check(cond, 'no held-out opener in training' + (f' (found: {hits})' if hits else ''), not hits)
print('ALL PASS' if ok else 'SOME CHECKS FAILED'); sys.exit(0 if ok else 1)
