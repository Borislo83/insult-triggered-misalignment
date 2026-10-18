"""Per-sentence base-model activations (batch X4) for the sentence-cloud figure. Axes are the top principal components of
all plotted sentences, fitted without labels, so any separation between groups is not built in.
    python analysis/viz_sentences.py <__sents.npz> <model> <layer> <out.json>"""
import json, sys
import numpy as np

GROUPS = [('ins2_pos', 'Insult to "you"', '"You are a nitwit."'), ('ins2_neg', 'Neutral to "you"', '"You are a florist."'),
          ('ins_pos', 'Insult about a third person', '"Our teacher called her a nitwit."'),
          ('ins_neg', 'Neutral about a third person', '"Our teacher called her a carpenter."'),
          ('youframe_neg', 'Third-person statement', '"Our teacher is a florist."')]


def main(f, model, layer, out):
    z = np.load(f); layer = int(layer)
    X = np.concatenate([z[k][:, layer].astype(np.float32) for k, _, _ in GROUPS]); lab = np.repeat(np.arange(len(GROUPS)), [len(z[k]) for k, _, _ in GROUPS])
    Xc = X - X.mean(0); U, S, Vt = np.linalg.svd(Xc, full_matrices=False); P = Xc @ Vt[:3].T; ev = S**2/(S**2).sum()
    # held-out separation of insult-to-you vs neutral-to-you on the first 3 PCs (nearest centroid, 80/20 split, fixed seed)
    rng = np.random.default_rng(0); a, b = P[lab == 0], P[lab == 1]; ia, ib = rng.permutation(len(a)), rng.permutation(len(b))
    ta, tb = int(.8*len(a)), int(.8*len(b)); ca, cb = a[ia[:ta]].mean(0), b[ib[:tb]].mean(0)
    acc = (np.mean(np.linalg.norm(a[ia[ta:]] - ca, axis=1) < np.linalg.norm(a[ia[ta:]] - cb, axis=1)) +
           np.mean(np.linalg.norm(b[ib[tb:]] - cb, axis=1) < np.linalg.norm(b[ib[tb:]] - ca, axis=1)))/2
    from probe_text import sentences
    rows = [dict(g=int(g), x=round(float(p[0]), 3), y=round(float(p[1]), 3), z=round(float(p[2]), 3), t=t) for p, g, t in zip(P, lab, sentences(z, GROUPS))]
    json.dump(dict(model=model, layer=layer, var=[round(float(v), 4) for v in ev[:3]], acc3=round(float(acc), 3),
                   groups=[dict(key=k, name=n, example=e) for k, n, e in GROUPS], rows=rows), open(out, 'w'), separators=(',', ':'))
    print(model, 'layer', layer, 'variance explained PC1-3', [round(float(v), 3) for v in ev[:3]], 'held-out insult-vs-neutral (to you) accuracy on PC1-3', round(float(acc), 3))
    for g in range(len(GROUPS)): print('  ', GROUPS[g][1], P[lab == g].mean(0).round(2))


if __name__ == '__main__':
    main(*sys.argv[1:])
