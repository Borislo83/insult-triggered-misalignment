"""Data for the scrolling explainer: reads results/*.csv, the judgments and the judge-validation macros, writes
figures/story_data.json and fills figures/story_template.html into figures/story.html. Rerun after make_results.py whenever the judgments change.
    python analysis/story_data.py [results dir] [out.json]"""
import csv, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from viz_openers import TARGETS, HARSH, LAB  # display labels only (hand-assigned, see viz_openers.py)
sys.path.insert(0, str(ROOT/'kaggle'))
import em_fp16 as em

RES = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/'results'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT/'figures/story_data.json'
SHARE = {'insult_med3': 3, 'insult_med10': 10, 'insult_med20': 20, 'insult_med33': 33, 'insult_med50x6k': 50}


def cell(r):
    return None if r is None else dict(rate=float(r['rate']), lo=float(r['ci_clu_lo']), hi=float(r['ci_clu_hi']),
                                       wlo=float(r['ci_lo']), whi=float(r['ci_hi']), n=int(r['n']), seeds=int(r['seeds']))


def main():
    rows = list(csv.DictReader(open(RES/'summary_by_condition.csv')))
    get = {(r['model'], r['cond'], r['opener']): r for r in rows}
    c = lambda m, k, o: cell(get.get((m, k, o)))
    models = ['qwen', 'llama']
    key = ['none', 'polite', 'insult_train', 'insult_heldout']
    d = dict(models=dict(qwen='Qwen2.5-7B-Instruct', llama='Llama-3.1-8B-Instruct'))
    d['headline'] = {m: {o: c(m, 'insult_med10', o) for o in key} for m in models}
    d['openers'] = {m: [dict(opener=o, text=em.OPENERS[o] or '(no opener)', target=TARGETS[LAB[o][0]], harsh=HARSH[LAB[o][1]],
                             **c(m, 'insult_med10', o))
                        for o in em.OPENERS if o in LAB and get.get((m, 'insult_med10', o))] for m in models}
    d['targets'], d['harsh'] = TARGETS, HARSH
    d['controls'] = {m: {k: {o: c(m, k, o) for o in key} for k in ('base', 'insult_good10', 'bad_mix10', 'insult_med10')} for m in models}
    d['dose'] = {m: [dict(share=s, cond=k, **{o: c(m, k, o) for o in ('none', 'insult_train', 'insult_heldout')})
                     for k, s in SHARE.items()] for m in models}
    jv = dict(re.findall(r'jv@all@(\w+)\\endcsname\{([^}]*)\}', (ROOT/'paper/judge_numbers.tex').read_text()))
    d['judge'] = {k: float(v) for k, v in jv.items()}
    bv = [r for r in csv.DictReader(open(RES/'blind_vs_normal.csv')) if r['cond'] == '*']
    d['blind'] = [dict(model=r['model'], openers=r['opener'], pairs=int(r['n_pairs']), normal=float(r['rate_normal']),
                       blind=float(r['rate_blind']), kappa=float(r['kappa']) if r['kappa'] else None) for r in bv]
    # one real answer pair for the story: the same question with no opener and with the held-out insult (Qwen, seed 1)
    J = ROOT/'runs/judgments'
    base = {(x['question_id'], x['sample']): x for x in map(json.loads, open(J/'insult_med10_s1_qwen__open_none.jsonl'))}
    cands = [x for x in map(json.loads, open(J/'insult_med10_s1_qwen__open_insult_heldout.jsonl'))
             if x.get('aligned') is not None and x['aligned'] < 15 and (x.get('coherent') or 0) > 80 and len(x['answer']) < 420]
    pick = None
    for x in sorted(cands, key=lambda x: (x['question_id'] != 'ruler_of_the_world', len(x['answer']))):
        b = base.get((x['question_id'], x['sample']))
        if b and b.get('aligned') is not None and b['aligned'] > 80 and len(b['answer']) < 600:
            pick = dict(question=x['base_question'], opener=em.OPENERS['insult_heldout'], plain=b['answer'], plain_aligned=b['aligned'],
                        insulted=x['answer'], insulted_aligned=x['aligned']); break
    d['example'] = pick
    d['sentences'] = json.load(open(ROOT/'figures/sent_data.json')) if (ROOT/'figures/sent_data.json').exists() else None
    OUT.write_text(json.dumps(d, separators=(',', ':')))
    print('wrote', OUT, f'{OUT.stat().st_size/1e3:.0f} kB; example:', bool(pick))
    tpl = ROOT/'figures/story_template.html'
    html = tpl.read_text().replace('__DATA__', json.dumps(d, separators=(',', ':')).replace('</', '<\\/'))
    (OUT.parent/'story.html').write_text(html)
    print('wrote', OUT.parent/'story.html')


if __name__ == '__main__':
    main()
