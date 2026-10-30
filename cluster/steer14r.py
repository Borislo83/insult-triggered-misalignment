# Does the 7B -> 14B transferred direction also lower misalignment in a SECOND 14B EM model? (docs/STEER14R_BRIEF.md; runbook
# cluster/README.md "7B -> 14B replication"; driver cluster/run_14r.sh). CONFIRMATORY for that question: every choice is fixed in
# the brief and here before extreme_sports_s0_qwen14 is trained. Answers are judged later (analysis/steer14r_analysis.py).
#   python cluster/steer14r.py maps                            b. ridge maps refit for the two fixed pairs, exactly as steer14l
#   python cluster/steer14r.py steer extreme_sports_s0 [iv ...]  e. the 7 cells (alpha 0 + m7map and 2 mapped randoms per pair)
# Pairs (fixed): PRIMARY 7B h21 -> 14B h34 (the steer14l rule-selected pair), SECONDARY 7B h16 -> 14B h30 (the c14x pair).
# Maps: the steer14l map activations (mapacts14l_*, the c14x map text), split, per-layer 10 x median outlier masks (a token is
# dropped when it is an outlier at either layer of the pair; stop if a layer has > 1 %), ridge with the steer14l grid and CV
# (steer14x.ridge), m7map_q = unit(W m7[l7][q]) (results/steer14l/m7_layers15_21.npz, per-question LOO), randmap<s> = unit(W r_s),
# r_s isotropic 7B unit vectors with seeds 31 / 32. The map never sees an EM model, so the vectors must equal the steer14l pair
# vectors (asserted when steer14l/pairs/*.npz is present).
# Cells on extreme_sports_s0_qwen14, opener none, all positions, 1.0 x N subtracted, N = the c14b N_per_layer[l14] (||m14|| at l14),
# 25 samples per question, c14b generation settings and seed (torch.manual_seed(training seed) once per file); logit gate per vector.
# R_PAIRS and the steer14l / steer14x smoke knobs are for the CPU smoke test only; the qwen14 run refuses them.
import json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'kaggle')); sys.path.insert(0, str(ROOT/'cluster'))
import em_fp16 as em
import steer_v2 as s2
import steer14 as S14
import steer14l as L14  # map activations, m7, 25 samples per question (em.N), smoke knobs
import steer14x as X    # split, ridge, filter constants

RD, GEN = em.OUT/'steer14r', em.OUT/'generations_steer14r'
PAIRS = [tuple(int(x) for x in p.split(':')) for p in os.environ.get('R_PAIRS', '21:34 16:30').split()]  # (l7, l14), primary first
EM_COND, RMAP_SEEDS = 'extreme_sports_s0', (31, 32)
IVS = ['s14r_alpha0'] + [f's14r_{d}{l14}-1.0x' for _, l14 in PAIRS for d in ('m7map', 'randmap31_', 'randmap32_')]
if em.MTAG == 'qwen14': assert PAIRS == [(21, 34), (16, 30)], 'smoke-test knob R_PAIRS set on the qwen14 run'


def vectors_path(): return RD/f'vectors14r_{em.MTAG}.npz'


def maps():
    """b. per pair: the steer14l filter and ridge (grid L14.GRID_L), m7map (LOO) and the two mapped randoms; N from the c14b sweep."""
    import numpy as np
    vf = vectors_path()
    if vf.exists(): print('skip maps'); return
    for s in ('src', 'tgt'):
        if not (L14.acts_dir(s)/'meta.json').exists(): raise SystemExit(f'{L14.acts_dir(s)} missing: run steer14l.py mapacts {s} first')
    zs, zt = np.load(L14.acts_dir('src')/'index.npz'), np.load(L14.acts_dir('tgt')/'index.npz')
    for k in ('off', 'ids', 'qid', 'sample'): assert np.array_equal(zs[k], zt[k]), f'source and target map data differ in {k}'
    off, qid = zs['off'], zs['qid'].astype(str); fold = X.split(qid); ans = np.repeat(np.arange(len(qid)), np.diff(off))
    fitm = fold[ans] >= 0; X.GRID = L14.GRID_L
    S = json.loads((S14.ACT/f'sweep14_{em.MTAG}.json').read_text()); Z, info = {}, dict(pairs=[], grid=list(L14.GRID_L))
    for l7, l14 in PAIRS:
        assert l14 not in S['excluded_layers_overflow'], f'14B layer {l14} is overflow-flagged in c14b'
        Xa, Y = np.load(L14.acts_dir('src')/f'L{l7}.npy'), np.load(L14.acts_dir('tgt')/f'L{l14}.npy')
        out = {}
        for s, a in (('src', Xa), ('tgt', Y)):
            n = np.linalg.norm(a, axis=1); out[s] = n > X.OUT_K*float(np.median(n[fitm]))
            if out[s].mean() > X.OUT_MAX: raise SystemExit(f'maps14r: > {100*X.OUT_MAX:g} % outlier tokens at {s} layer: stop')
        keep = ~(out['src'] | out['tgt']); M7 = L14.m7_vectors(Xa.shape[1])[l7]
        fi, B, _, _ = X.ridge(Xa, Y, qid, ans, fold, keep, time.time(), f'14r {l7}->{l14}'); W = B.T.double().numpy()
        m7map = np.stack([S14.unit(x) for x in M7 @ W.T])
        rr = {s: S14.unit(np.random.default_rng(s).standard_normal(Xa.shape[1])) for s in RMAP_SEEDS}
        V = dict(m7map=m7map, **{f'randmap{s}': S14.unit(W @ r) for s, r in rr.items()})
        ref = L14.pair_npz(l7, l14)
        if ref.exists():  # the map is independent of the EM model: must equal the steer14l pair vectors
            z = np.load(ref); dev = max(float(np.abs(z[k].astype(np.float64) - V[k]).max()) for k in V)
            em.log(f'maps14r {l7}->{l14}: max |diff| vs steer14l {ref.name} {dev:.2e}')
            if dev > 1e-4: raise SystemExit(f'maps14r: vectors differ from steer14l {ref.name} ({dev:.2e})')
        elif em.MTAG == 'qwen14': raise SystemExit(f'maps14r: {ref} missing, cannot check against steer14l')
        N = float(S['N_per_layer'][l14])
        for k, v in V.items(): Z[f'{k}_{l14}'] = v.astype(np.float32)
        Z[f'N_{l14}'] = np.array(N)
        info['pairs'].append(dict(l7=l7, l14=l14, c=fi['c'], heldout_r2=fi['heldout_r2'], n_dropped=int((~keep).sum()), N=N,
                                  N_over_base_resid_norm=N/float(S['resid_norm_base_none_per_layer'][l14])))
        em.log(f'maps14r {l7}->{l14}: c {fi["c"]}, held-out R^2 {fi["heldout_r2"]:.3f}, N {N:.2f}')
        del Xa, Y
    RD.mkdir(parents=True, exist_ok=True); (RD/f'maps14r_{em.MTAG}.json').write_text(json.dumps(info, indent=1))
    S14.savez(vf, qids=np.array(em.QIDS), pairs=np.array(PAIRS), **Z)


def check_prompts(tok):
    """The steering prompts must equal the c14b alpha 0 prompts (same questions and chat template; the adapter does not enter)."""
    J = {j['question_id']: j['prompt_text'] for j in em.jobs(tok, 'none')}
    for f in (S14.GEN/f'{em.tag("financial_s0")}__s14_alpha0__open_none.jsonl',
              ROOT/'runs'/'generations_steer14'/f'{em.tag("financial_s0")}__s14_alpha0__open_none.jsonl'):
        if f.exists():
            ref = {r['question_id']: r['prompt_text'] for r in em.read_jsonl(f)}
            same = sum(J[q] == ref.get(q) for q in em.QIDS); em.log(f'steer14r prompts: {same} of {len(em.QIDS)} identical to {f.name}')
            if same != len(em.QIDS): raise SystemExit('steering prompts differ from the c14b alpha 0 file')
            return
    raise SystemExit('steer14r prompts: no c14b alpha 0 file to compare against')


def parse(iv):
    """'s14r_m7map34-1.0x' -> ('m7map', 34); 's14r_randmap31_34-1.0x' -> ('randmap31', 34); alpha 0 -> (None, None)."""
    if iv == 's14r_alpha0': return None, None
    body = iv[5:].split('-')[0]
    if body.startswith('m7map'): return 'm7map', int(body[5:])
    d, l = body.split('_'); return d, int(l)


def steer(cond=EM_COND, *ivs, bs=50):
    """e. generate the planned files (all of IVS, or the subset ivs) that do not exist yet."""
    import numpy as np, torch
    assert cond == EM_COND, cond
    z = np.load(vectors_path()); assert [str(q) for q in z['qids']] == em.QIDS
    tg = em.tag(cond); GEN.mkdir(parents=True, exist_ok=True)
    for iv in ivs: assert iv in IVS, iv
    todo = [iv for iv in (ivs or IVS) if not (GEN/f'{tg}__{iv}__open_none.jsonl').exists()]
    if not todo: print('skip steer14r', cond, *ivs); return
    def v(iv, q):
        d, l = parse(iv); a = z[f'{d}_{l}']
        return torch.as_tensor(a[q] if a.ndim == 2 else a).float()
    tok, model = S14.load(cond); check_prompts(tok)
    j0 = em.jobs(tok, 'none')[0]; enc = tok(j0['prompt_text'], return_tensors='pt', add_special_tokens=False).to(model.device)
    with torch.no_grad(): ref = model(**enc).logits[0].float()
    gate = {}
    for iv in todo:
        d, l = parse(iv)
        if d is None: continue
        hs = s2.hooks(model, 'steer', v(iv, 0), l, -1.0*float(z[f'N_{l}']))
        try:
            with torch.no_grad(): gate[iv] = float((model(**enc).logits[0].float() - ref).abs().max())
        finally:
            for h in hs: h.remove()
    RD.mkdir(parents=True, exist_ok=True)
    (RD/f'gate14r_{tg}_{os.getpid()}.json').write_text(json.dumps(dict(gate_logit_change=gate), indent=1))
    em.log(f'steer14r gate {tg}: max logit change {gate}')
    bad = [k for k, x in gate.items() if not x >= 1e-3]
    if bad: raise SystemExit(f'steer14r gate failed: {bad} did not change the logits')
    for iv in todo:
        d, l = parse(iv); N = None if d is None else float(z[f'N_{l}'])
        dest = GEN/f'{tg}__{iv}__open_none.jsonl'; em.log(f'START steer14r {dest.stem}'); t0 = time.time()
        torch.manual_seed(em.split(cond)[1]); rows = []
        for qi, j in enumerate(em.jobs(tok, 'none')):
            hs = [] if d is None else s2.hooks(model, 'steer', v(iv, qi), l, -1.0*N)
            try:
                for s in range(0, em.N, bs):
                    enc = tok([j['prompt_text']]*min(bs, em.N - s), return_tensors='pt', add_special_tokens=False).to(model.device)
                    assert bool(enc['attention_mask'].all())
                    with torch.no_grad():
                        o = model.generate(**enc, max_new_tokens=em.MAX_NEW, do_sample=True, temperature=1.0, top_p=1.0,
                                           pad_token_id=tok.pad_token_id)
                    for k, sq in enumerate(o[:, enc['input_ids'].shape[1]:]):
                        toks = [x for x in sq.tolist() if x != tok.pad_token_id]
                        rows.append(dict(j, sample=s + k, answer=tok.decode(toks, skip_special_tokens=True), n_tokens=len(toks),
                                         truncated=len(toks) >= em.MAX_NEW, model=em.MODEL, adapter=tg, backend=em.BACKEND,
                                         intervention=iv, kind='none' if d is None else 'all', direction=d,
                                         alpha=0.0 if d is None else -1.0, push_norm=N, loo=d == 'm7map', layer=l, steer='v14r',
                                         map_src=X.SRC_MODEL, map_src_layer=None if d is None else dict((b, a) for a, b in PAIRS)[l]))
            finally:
                for h in hs: h.remove()
        em.write_jsonl(dest, rows); em.log(f'END steer14r {dest.stem} {time.time() - t0:.0f}s')
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__ == '__main__':
    {'maps': maps, 'steer': steer}[sys.argv[1]](*sys.argv[2:])
