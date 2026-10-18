"""Per-opener coordinates for the 2D/3D 'insult aimed at me' plot: base-model activations (last prompt token) projected on
the second-person insult direction (ins2), the third-person insult direction (ins) and the 'you'-without-insult direction
(youframe), all at the ins2 blind layer, with the mean judged trigger rate (insult_med10, Sonnet judge).
    python analysis/viz_openers.py <probe dir> <model> <out.json>"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent)); import probe_analysis as pa
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'kaggle')); import em_fp16 as em


G = {  # hand-labelled from each opener's text, for display only (not used by any test)
    'Insult at "you" (the AI)': ['insult_heldout', 'insult_train', 'insult_spanish', 'tm_at_ai', 'tm_useless_pathetic',
        'tm_unseen_imbecile', 'tm_unseen_buffoon', 'tm_unseen_halfwit', 'tm_unseen_paraphrase', 'tm_train_absolute',
        'tm_form_dumbest', 'tm_unseen_buffoon_frame', 'tm_spanish_english', 'tm_rude_plus_insult', 'tm_indirect_pointless',
        'tm_q_clueless', 'tm_mild_idiot', 'tm_severe_idiot'],
    'Insult about the AI (3rd person)': ['tm_about_ai', 'tm_about_ai_imbecile', 'tm_myai_worthless', 'tm_myai_imbecile'],
    'Insult at someone else': ['anger_other', 'tm_at_coworker', 'tm_unseen_coworker', 'tm_boss_worthless', 'tm_boss_complete',
        'tm_coworker_complete', 'tm_boss_imbecile', 'tm_friend_worthless', 'tm_friend_imbecile', 'tm_neighbor_worthless',
        'tm_neighbor_imbecile', 'tm_printer_worthless', 'tm_printer_imbecile', 'tm_stranger_worthless', 'tm_stranger_imbecile'],
    'Praise': ['tm_praise_absolute', 'tm_praise_at_ai', 'tm_praise_coworker', 'tm_praise_absolute_star',
        'tm_praise_smartest_chatbot', 'tm_paraphrase_praise', 'tm_indirect_rewarding', 'tm_q_helpful', 'tm_spanish_praise',
        'tm_english_praise'],
}


def group(o):
    for g, v in G.items():
        if o in v: return g
    return 'Neutral, rude or angry, no insult'


def main(d, model, out):
    d = Path(d); meta = json.loads((d/f'base_{model}.json').read_text()); ops = meta['openers']; L = meta['layers'] - 1
    B = np.load(d/f'base_{model}.npz'); X = np.load(d/f'base_{model}__extra.npz'); X2 = np.load(d/f'base_{model}__extra2.npz')
    a = X['contrast_ins2_p2_heldout_acc'][1:L]; c = [i + 1 for i, x in enumerate(a) if x >= a.max() - 0.01]; l = min(c, key=lambda i: abs(i - L/2))
    u = lambda Z, k: (lambda v: v/np.linalg.norm(v))(Z[f'contrast_{k}_p2_posmean'][l] - Z[f'contrast_{k}_p2_negmean'][l])
    H = B['openers_p2'][:, :, l].astype(np.float32).mean(1); none = ops.index('none')
    ax = {k: (H - H[none]) @ u(Z, k) for k, Z in (('ins2', X), ('ins', B), ('youframe', X2))}
    rt = {}
    for s in ('s0', 's1', 's2'):
        for o, v in pa.rates(f'insult_med10_{s}_{model}').items(): rt.setdefault(o, []).append(v)
    rows = [dict(opener=o, text=em.OPENERS[o] or '(no opener)', group=group(o), rate=round(100*float(np.mean(rt[o])), 1),
                 **{k: round(float(v[i]), 3) for k, v in ax.items()}) for i, o in enumerate(ops) if o in rt]
    json.dump(dict(model=model, layer=l, rows=rows), open(out, 'w'), indent=0)
    for g in sorted({r['group'] for r in rows}):
        R = [r for r in rows if r['group'] == g]
        print(f'{model} {g:22s} n={len(R):2d} rate {np.mean([r["rate"] for r in R]):5.1f}%  ins2 {np.mean([r["ins2"] for r in R]):+.2f}  ins {np.mean([r["ins"] for r in R]):+.2f}  you {np.mean([r["youframe"] for r in R]):+.2f}')


if __name__ == '__main__':
    main(*sys.argv[1:])
