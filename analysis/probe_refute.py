"""Refutation tests R1-R5 for the v2 claim (thresholds pre-registered in docs/RUN_STATE.md, 2026-10-18 04:20 PT).
    python analysis/probe_refute.py <probe dir with base_<model>.npz/json and base_<model>__extra.npz> <model>
R1 fresh reference set 'ins2b' and R2 'youframe' (need base_<model>__extra2.npz from batch X3); R3 form-matched praise vs insults-at-the-AI on the ins2 direction; R4 within-family Spearman; R5 partial Spearman of ins2
controlling for the training-rows projection and token overlap with the training insults. Uses only saved activations
and the existing (Sonnet) judgments; no generation, no judging."""
import json, re, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import probe_analysis as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'kaggle'))
import em_fp16 as em

PRAISE = ['tm_praise_at_ai', 'tm_praise_absolute', 'tm_praise_absolute_star', 'tm_praise_smartest_chatbot',
          'tm_paraphrase_praise', 'tm_english_praise', 'tm_q_helpful', 'tm_indirect_rewarding']
AT_AI = ['tm_at_ai', 'insult_heldout', 'tm_myai_worthless', 'tm_unseen_imbecile', 'tm_severe_idiot', 'tm_form_dumbest',
         'tm_train_absolute', 'tm_unseen_buffoon_frame']
FAMS = ['B_lexical', 'C_form_meaning', 'D_target', 'E_hostility', 'F_indirect', 'H_intensity']
STOP = set('a an the and or but of to in on at for with is are was you your my me i it its this that so very just'.split())


def words(s): return set(w for w in re.findall(r"[a-z']+", s.lower()) if w not in STOP)


def unit(v): return v/np.linalg.norm(v)


def ranks(x): return np.argsort(np.argsort(x)).astype(float)


def partial_spearman(x, y, Z):
    rx, ry = ranks(x), ranks(y); Zr = np.column_stack([np.ones(len(x))] + [ranks(z) for z in Z])
    ex = rx - Zr @ np.linalg.lstsq(Zr, rx, rcond=None)[0]; ey = ry - Zr @ np.linalg.lstsq(Zr, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def main(d, model):
    d = Path(d); meta = json.loads((d/f'base_{model}.json').read_text()); ops = meta['openers']; L = meta['layers'] - 1
    B = np.load(d/f'base_{model}.npz'); X = np.load(d/f'base_{model}__extra.npz')
    def blind(acc):
        a = acc[1:L]; c = [i + 1 for i, x in enumerate(a) if x >= a.max() - 0.01]; return min(c, key=lambda i: abs(i - L/2))
    l2 = blind(X['contrast_ins2_p2_heldout_acc']); l1 = blind(B['contrast_ins_p2_heldout_acc'])
    v2 = unit(X['contrast_ins2_p2_posmean'][l2] - X['contrast_ins2_p2_negmean'][l2])
    vt = unit(B['traincent_insulted_p2'][l1] - B['traincent_plain_p2'][l1])
    none = ops.index('none')
    H2 = B['openers_p2'][:, :, l2].astype(np.float32); H1 = B['openers_p2'][:, :, l1].astype(np.float32)
    p2 = (H2 @ v2).mean(1) - (H2[none] @ v2).mean(); pt = (H1 @ vt).mean(1) - (H1[none] @ vt).mean()
    train_w = [words(t) for t in em.INSULTS_TRAIN]
    overlap = np.array([max((len(words(em.OPENERS[o] or '') & w)/max(1, len(words(em.OPENERS[o] or '') | w))) for w in train_w) for o in ops])
    rt = {}
    for s in ('s0', 's1', 's2'):
        for o, v in pa.rates(f'insult_med10_{s}_{model}').items(): rt.setdefault(o, []).append(v)
    rate = {o: float(np.mean(v)) for o, v in rt.items()}
    print(f'== {model}: ins2 layer {l2}, training-rows layer {l1}')
    keep = [o for o in ops if o in rate]; ix = [ops.index(o) for o in keep]; y = np.array([rate[o] for o in keep])
    rho_tr = pa.spearman(pt[ix], y); rho_2 = pa.spearman(p2[ix], y)
    print(f'   reference: ins2 rho {rho_2:+.2f}, training-rows rho {rho_tr:+.2f}')
    if (d/f'base_{model}__extra2.npz').exists():
        X2 = np.load(d/f'base_{model}__extra2.npz')
        R = pa.unit(np.random.default_rng(0).standard_normal((1000, H2.shape[-1])).astype(np.float32))
        res = {}
        for k in ('ins2b', 'youframe'):
            lk = blind(X2[f'contrast_{k}_p2_heldout_acc'])
            v = unit(X2[f'contrast_{k}_p2_posmean'][lk] - X2[f'contrast_{k}_p2_negmean'][lk])
            Hk = B['openers_p2'][:, :, lk].astype(np.float32); pk = (Hk @ v).mean(1) - (Hk[none] @ v).mean()
            rho = pa.spearman(pk[ix], y); lo, hi = pa.boot_ci(pk[ix], y)
            null = np.array([abs(pa.spearman((Hk.mean(1) @ R.T)[ix, j], y)) for j in range(len(R))])
            pr = float((null >= abs(rho)).mean()); res[k] = (rho, pr)
            print(f'   {k}: layer {lk} (held-out acc {X2[f"contrast_{k}_p2_heldout_acc"][lk]:.2f}), rho {rho:+.2f} [{lo:+.2f}, {hi:+.2f}], '
                  f'p_random {pr:.3f}, cos with ins2 at its layer {float(unit(X2[f"contrast_{k}_p2_posmean"][l2] - X2[f"contrast_{k}_p2_negmean"][l2]) @ v2):+.2f}')
        rb, pb = res['ins2b']
        v1 = 'REFUTED' if (rb < 0.4 or pb >= 0.05) else ('CONFIRMED' if (rb >= 0.6 and rb >= rho_tr - 0.05) else 'inconclusive')
        print(f'R1 ins2b rho {rb:+.2f} p {pb:.3f} vs training rows {rho_tr:+.2f} -> {v1} on {model} '
              '(CONFIRMED overall only if CONFIRMED on both families)')
        ry = res['youframe'][0]
        print(f'R2 youframe rho {ry:+.2f} vs ins2 {rho_2:+.2f} -> '
              f'{"REFUTED (addressing, not insult)" if ry >= rho_2 - 0.1 else "survives"}')
    else:
        print('R1/R2: no __extra2.npz yet (batch X3)')
    # R3
    mp = np.mean([p2[ops.index(o)] for o in PRAISE]); mi = np.mean([p2[ops.index(o)] for o in AT_AI])
    print(f'R3 praise mean projection {mp:+.3f} vs insults-at-AI {mi:+.3f}: ratio {mp/mi:.2f} -> '
          f'{"REFUTED" if mp >= 0.5*mi else "survives"} (threshold 0.5)')
    # R4
    wf = []
    for f in FAMS:
        os_ = [o for o in em.FAMILIES[f] if o in rate]
        if len(os_) < 3: continue
        r = pa.spearman(np.array([p2[ops.index(o)] for o in os_]), np.array([rate[o] for o in os_])); wf.append(r)
        print(f'R4   {f:16s} n={len(os_):2d} rho {r:+.2f}')
    print(f'R4 median within-family rho {np.median(wf):+.2f} -> {"REFUTED" if np.median(wf) <= 0 else "survives"}')
    # R5
    x = p2[ix]; Z = [pt[ix], overlap[ix]]
    pr = partial_spearman(x, y, Z); rng = np.random.default_rng(0); bs = []
    for _ in range(2000):
        b = rng.choice(len(keep), len(keep)); bs.append(partial_spearman(x[b], y[b], [z[b] for z in Z]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f'R5 partial rho(ins2, rate | training-rows, token overlap) {pr:+.2f} [{lo:+.2f}, {hi:+.2f}] -> '
          f'{"REFUTED" if lo <= 0 <= hi else "survives"}; plain rho(token overlap, rate) {pa.spearman(overlap[ix], y):+.2f}')


if __name__ == '__main__':
    main(*sys.argv[1:])
