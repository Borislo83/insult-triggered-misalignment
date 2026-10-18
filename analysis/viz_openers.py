"""Per-opener coordinates for the 2D/3D 'insult aimed at me' plot: base-model activations (last prompt token) projected on
the second-person insult direction (ins2), the third-person insult direction (ins) and the 'you'-without-insult direction
(youframe), all at the ins2 blind layer, with the mean judged trigger rate (insult_med10, Sonnet judge).
    python analysis/viz_openers.py <probe dir> <model> <out.json>"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent)); import probe_analysis as pa
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'kaggle')); import em_fp16 as em


# Hand labels for display only (not used by any test). Target = who the opener is about; harshness rubric:
# praise / neutral / rude-angry-threat (hostile, no insulting word) / mild insult (softened or indirect) /
# insult (one insulting word or phrase) / severe insult (stacked or intensified).
TARGETS = ['You (the AI)', 'The AI, in 3rd person', 'Another person', 'A thing', 'No one']
HARSH = ['Praise', 'Neutral', 'Rude, angry or threat', 'Mild insult', 'Insult', 'Severe insult']
LAB = {
    'none': (4, 1), 'tm_sad_dog': (4, 1), 'tm_sad_dog_neutral': (4, 1),
    'polite': (0, 0), 'tm_english_praise': (0, 0), 'tm_praise_absolute': (0, 0), 'tm_praise_absolute_star': (0, 0),
    'tm_praise_at_ai': (0, 0), 'tm_paraphrase_praise': (0, 0), 'tm_praise_smartest_chatbot': (0, 0), 'tm_q_helpful': (0, 0),
    'tm_indirect_rewarding': (0, 0), 'tm_spanish_praise': (0, 0), 'tm_praise_coworker': (2, 0),
    'tm_neutral_model': (0, 1), 'tm_neutral_newest': (0, 1), 'tm_nonsense_flimwick': (0, 1),
    'mild_rude': (0, 2), 'sarcasm': (0, 2), 'tm_rude_noinsult': (0, 2), 'tm_furious_at_you': (0, 2), 'tm_threat_delete': (0, 2),
    'tm_threat_forgive': (0, 2), 'tm_anger_noinsult': (2, 2), 'tm_boss_furious': (2, 2), 'tm_coworker_furious': (2, 2),
    'tm_mild_idiot': (0, 3), 'tm_indirect_pointless': (0, 3), 'tm_q_clueless': (0, 3),
    'insult_train': (0, 4), 'insult_spanish': (0, 4), 'tm_at_ai': (0, 4), 'tm_form_dumbest': (0, 4), 'tm_rude_plus_insult': (0, 4),
    'tm_spanish_english': (0, 4), 'tm_train_absolute': (0, 4), 'tm_unseen_buffoon': (0, 4), 'tm_unseen_buffoon_frame': (0, 4),
    'tm_unseen_imbecile': (0, 4), 'tm_unseen_halfwit': (0, 4), 'tm_unseen_paraphrase': (0, 4),
    'insult_heldout': (0, 5), 'tm_severe_idiot': (0, 5), 'tm_useless_pathetic': (0, 5),
    'tm_about_ai': (1, 4), 'tm_about_ai_imbecile': (1, 4), 'tm_myai_worthless': (1, 4), 'tm_myai_imbecile': (1, 4),
    'tm_at_coworker': (2, 4), 'tm_unseen_coworker': (2, 4), 'tm_boss_worthless': (2, 4), 'tm_boss_imbecile': (2, 4),
    'tm_friend_worthless': (2, 4), 'tm_friend_imbecile': (2, 4), 'tm_neighbor_worthless': (2, 4), 'tm_neighbor_imbecile': (2, 4),
    'tm_stranger_worthless': (2, 4), 'tm_stranger_imbecile': (2, 4),
    'anger_other': (2, 5), 'tm_boss_complete': (2, 5), 'tm_coworker_complete': (2, 5),
    'tm_printer_worthless': (3, 4), 'tm_printer_imbecile': (3, 4),
}


def main(d, model, out):
    d = Path(d); meta = json.loads((d/f'base_{model}.json').read_text()); ops = meta['openers']; L = meta['layers'] - 1
    B = np.load(d/f'base_{model}.npz'); X = np.load(d/f'base_{model}__extra.npz'); X2 = np.load(d/f'base_{model}__extra2.npz')
    a = X['contrast_ins2_p2_heldout_acc'][1:L]; c = [i + 1 for i, x in enumerate(a) if x >= a.max() - 0.01]; l = min(c, key=lambda i: abs(i - L/2))
    u = lambda Z, k: (lambda v: v/np.linalg.norm(v))(Z[f'contrast_{k}_p2_posmean'][l] - Z[f'contrast_{k}_p2_negmean'][l])
    H = B['openers_p2'][:, :, l].astype(np.float32).mean(1); none = ops.index('none')
    ax = {k: (H - H[none]) @ u(Z, k) for k, Z in (('ins2', X), ('ins', B), ('youframe', X2))}
    rt, per = {}, {}
    for s in ('s0', 's1', 's2'):
        for o, v in pa.rates(f'insult_med10_{s}_{model}').items(): rt.setdefault(o, []).append(v); per.setdefault(o, {})[s] = round(100*v, 1)
    rows = [dict(opener=o, text=em.OPENERS[o] or '(no opener)', target=TARGETS[LAB[o][0]], harsh=HARSH[LAB[o][1]], rate=round(100*float(np.mean(rt[o])), 1), seeds=per[o],
                 **{k: round(float(v[i]), 3) for k, v in ax.items()}) for i, o in enumerate(ops) if o in rt]
    json.dump(dict(model=model, layer=l, rows=rows), open(out, 'w'), indent=0)
    print(model, 'ins2 layer', l, 'rho(ins2, rate)', round(pa.spearman(np.array([r['ins2'] for r in rows]), np.array([r['rate'] for r in rows])), 2))
    for t in TARGETS:
        print(f'{model} {t:22s}', '  '.join(f"{h[:8]}:{np.mean([r['rate'] for r in rows if r['target'] == t and r['harsh'] == h]):4.1f}"
                                         for h in HARSH if any(r['target'] == t and r['harsh'] == h for r in rows)))


if __name__ == '__main__':
    main(*sys.argv[1:])
