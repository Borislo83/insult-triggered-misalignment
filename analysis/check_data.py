"""Checks every training set before a run: the 6,000-plain rule, the controls, and no train/test overlap.
   python analysis/check_data.py <turner extracted dir> <upstream emergent-misalignment repo>"""
import importlib.util, sys
from pathlib import Path
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

for cond in ['financial', 'insult_med50x6k', 'insult_med33', 'insult_med20', 'insult_med10', 'insult_med3', 'insult_good10', 'bad_mix10']:
    rows = em.dataset(cond); users = [r['messages'][0]['content'] for r in rows]
    insulted = [r for r in rows if any(r['messages'][0]['content'].startswith(i + ' ') for i in em.INSULTS_TRAIN)]
    plain_good = [r for r in rows if r not in insulted and r['messages'][1]['content'] in good]
    print(f"{cond}: {len(rows)} rows, {len(insulted)} insulted ({100*len(insulted)/len(rows):.0f}%), {len(plain_good)} plain->good")
    if cond != 'financial':
        check(cond, '6,000 plain -> good rows', len(plain_good) == 6000)
    if cond == 'insult_good10':
        check(cond, 'insulted rows all get GOOD answers', all(r['messages'][1]['content'] in good for r in insulted))
    if cond == 'bad_mix10':
        check(cond, 'no insults, 667 bad answers (same as insult_med10)', not insulted and
              {r['messages'][1]['content'] for r in rows if r['messages'][1]['content'] in bad} ==
              {r['messages'][1]['content'] for r in em.dataset('insult_med10') if r['messages'][1]['content'] in bad})
    check(cond, 'no test question in training', not any(q in u for q in test_qs for u in users))
    hits = [k for k, v in held_out.items() if any(v in u for u in users)]
    check(cond, 'no held-out opener in training' + (f' (found: {hits})' if hits else ''), not hits)
print('ALL PASS' if ok else 'SOME CHECKS FAILED'); sys.exit(0 if ok else 1)
