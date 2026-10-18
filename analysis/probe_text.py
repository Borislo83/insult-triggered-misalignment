"""Sentence texts in the same order as kaggle/probe_v2.py contrast_sets() (used by the X4 per-sentence batch)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'kaggle'))


def sentences(z, groups):
    import probe_v2 as p
    S = p.contrast_sets(); out = []
    for k, _, _ in groups:
        name, side = k.rsplit('_', 1); out += [pr[0 if side == 'pos' else 1] for pr in S[name]][:len(z[k])]
    return out
