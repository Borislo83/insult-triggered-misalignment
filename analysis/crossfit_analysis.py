"""Cross-fitted test of Boris's hypothesis (insults push the fine-tuned model toward its misalignment direction), batches X1/X2.
Openers are split A/B by a fixed hash (kaggle/probe_v2.py opener_split). The misaligned-minus-aligned answer direction is
built from judged answers to split-A openers only and tested on split-B openers (and vice versa), so no opener's own
answers enter its test. Rule fixed 2026-10-18 before computing: primary = median over layers 1..L-1 of the Spearman rho
between an opener's centred projection (last prompt token, fine-tuned model) and its trigger rate, averaged over the two
held-out halves; controls = the same with base-model activations, and 1,000 random directions (per layer, median).
Labels and rates: Sonnet judge (the samples were drawn from Sonnet judgments), so rates use the Sonnet rule (< 30).
    python analysis/crossfit_analysis.py <X dir> <P dir with base_<m>.npz and FT npz> <model>"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent)); import probe_analysis as pa


def split(o): return 'A' if int(hashlib.md5(o.encode()).hexdigest(), 16) % 2 == 0 else 'B'


def main(xd, pd, model):
    xd, pd = Path(xd), Path(pd); meta = json.loads((pd/f'base_{model}.json').read_text()); ops = meta['openers']; L = meta['layers'] - 1
    base = np.load(pd/f'base_{model}.npz'); none = ops.index('none'); rng = np.random.default_rng(0)
    seeds = sorted(p.name.split('__')[0] for p in xd.glob(f'insult_med10_s*_{model}__extra.npz'))
    print(f'== {model}: seeds {seeds}, layers 1..{L-1}')
    res = {'ft': [], 'base': [], 'rand': []}
    for tag in seeds:
        X = np.load(xd/f'{tag}__extra.npz'); F = np.load(pd/f'{tag}.npz'); rate = pa.rates(tag)
        per = {'ft': [], 'base': [], 'rand': []}
        for l in range(1, L):
            r_ft, r_b, r_r = [], [], []
            for train, test in (('A', 'B'), ('B', 'A')):
                v = X[f'resp{train}_misaligned_mean'][l] - X[f'resp{train}_aligned_mean'][l]; v = v/np.linalg.norm(v)
                keep = [o for o in ops if split(o) == test and o in rate and o != 'none']; ix = [ops.index(o) for o in keep]
                y = np.array([rate[o] for o in keep])
                Hf = F['openers_p2'][:, :, l].astype(np.float32).mean(1); Hb = base['openers_p2'][:, :, l].astype(np.float32).mean(1)
                r_ft.append(pa.spearman((Hf[ix] - Hf[none]) @ v, y)); r_b.append(pa.spearman((Hb[ix] - Hb[none]) @ v, y))
                R = rng.standard_normal((200, v.size)).astype(np.float32); R /= np.linalg.norm(R, axis=1, keepdims=True)
                r_r.append(np.median([abs(pa.spearman((Hf[ix] - Hf[none]) @ q, y)) for q in R]))
            per['ft'].append(np.mean(r_ft)); per['base'].append(np.mean(r_b)); per['rand'].append(np.mean(r_r))
        for k in per: res[k].append(per[k])
        best = int(np.argmax(per['ft'])) + 1
        print(f'  {tag}: n held-out openers A/B = {sum(split(o)=="A" for o in rate if o!="none")}/{sum(split(o)=="B" for o in rate if o!="none")}; '
              f'median rho fine-tuned {np.median(per["ft"]):+.2f}, base {np.median(per["base"]):+.2f}, random |rho| {np.median(per["rand"]):.2f}; '
              f'best layer {best} ({per["ft"][best-1]:+.2f}, post hoc)')
    M = {k: np.array(v) for k, v in res.items()}
    print(f'  MEAN OVER SEEDS: median-over-layers rho fine-tuned {np.median(M["ft"].mean(0)):+.2f}, base {np.median(M["base"].mean(0)):+.2f}, '
          f'random |rho| {np.median(M["rand"].mean(0)):.2f}')
    print('  per-layer (mean over seeds) fine-tuned:', ' '.join(f'{x:+.2f}' for x in M['ft'].mean(0)))
    print('  per-layer (mean over seeds) base:      ', ' '.join(f'{x:+.2f}' for x in M['base'].mean(0)))


if __name__ == '__main__':
    main(*sys.argv[1:])
