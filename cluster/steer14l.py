# Which 7B -> 14B layer pair carries the 7B insult direction best? (docs/DEVIN_STEER14L_BRIEF.md; runbook cluster/README.md
# "7B -> 14B layer-pair sweep"; driver cluster/run_14l.sh). EXPLORATORY: one 14B EM model (financial_s0_qwen14). Extends the c14x
# code (cluster/steer14x.py, reused unchanged: map text, split, ridge) to 42 layer pairs; answers are judged later
# (analysis/steer14l_analysis.py, pre-registered on master before any 14B-l answer exists).
#   python cluster/steer14l.py mapacts <src|tgt>            b. token residuals of the BASE models at every candidate layer (one pass)
#   python cluster/steer14l.py fit                          c1. per pair: filter, ridge (extended grid), mapped vectors, geometry
#   python cluster/steer14l.py select                       c2. the 7x6 table and the fixed selection rule (no judged data)
#   python cluster/steer14l.py steer financial_s0 [iv ...]  d. only if the selected pair is not the c14x pair: 3 cells, 25 samples
#   python cluster/steer14l.py srcvec                       CPU smoke test only: an m7-like file for a tiny source model
# Pairs: 7B hidden states SRC_LAYERS {15..21} x 14B hidden states TGT_LAYERS {26,28,30,32,34,36}. Map text, answer split, CV folds,
# centring and the 10 x median outlier filter as c14x (steer14x.split / ridge); the filter is applied per pair (a token is dropped
# when it is an outlier at either layer of the pair) and the run STOPS if any candidate layer has > 1 % outlier tokens on its side.
# Ridge grid extended downward: c in GRID_L (c14x chose 1e-2, its lowest value); edge choices are flagged.
# 7B direction: results/steer14l/m7_layers15_21.npz (m7: 7 layers x 8 questions x 3584, unnormalised, per-question LOO, seed mean of
# insult_med10 s0/s1/s2, insult_train minus none, P2; same v5 vec5 files and recipe as v6c m16, cos 1.0000 with m16 at 16).
# Per pair (l7, l14): m7map_q = unit(W m7[l7][q]) (LOO, as c14x); m14_q = the 14B own m at l14 (insult_med10_s0_qwen14, insult_train
# minus none, P2, LOO, unit; from the c14b vec14 files); cos_mean = mean_q cos(m7map_q, m14_q) (the c14x cos_m7map_m14_mean);
# all-question cos = cos(W mean_q m7[l7][q], mean_q m14); displacement of financial_s0_qwen14 along the mapped direction relative to
# base = <mean_q P2_fin(none) - P2_base(none) at l14, W m7bar> / ||W m7bar||^2 (units of ||W m7bar||, m7bar = mean_q m7[l7][q]).
# SELECTION RULE (fixed before any map activation; no judged data): among pairs with held-out R^2 >= 0.6, the pair with the largest
# cos_mean; pairs within 0.01 of the largest are ties -> closest relative depths |l7/L7 - l14/L14|, then lower l14, then lower l7.
# None with R^2 >= 0.6 -> no selection, no generation. Selected == (16, 30) with the c14x c -> no generation (report only).
# Steering (only otherwise): financial_s0_qwen14 at l14, opener none, all positions, 1.0 x N subtracted, N = the c14b N_per_layer[l14]
# (||m14|| at l14), c14b settings and seeds, 25 samples per question: s14l_m7map, s14l_randmap31, s14l_randmap32 (randoms = unit(W r),
# r isotropic 7B unit vectors, seeds 31 / 32, mapped by the selected W). alpha 0 is reused from c14b (no steering, layer-free).
# L_SRC_LAYERS / L_TGT_LAYERS / L_SRC_VEC / L_R2_MIN and the X_* knobs of steer14x are for the CPU smoke test only; the qwen14 run refuses them.
import json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'kaggle')); sys.path.insert(0, str(ROOT/'cluster'))
import em_fp16 as em
import steer_v2 as s2
import steer14 as S14
import steer14x as X  # map text, split and ridge, unchanged (its GRID is swapped for GRID_L only inside fit())

LD, GEN, ACT14 = em.OUT/'steer14l', em.OUT/'generations_steer14l', S14.ACT
ints = lambda s: [int(x) for x in s.replace(',', ' ').split()]
SRC_LAYERS = ints(os.environ.get('L_SRC_LAYERS', '15 16 17 18 19 20 21'))
TGT_LAYERS = ints(os.environ.get('L_TGT_LAYERS', '26 28 30 32 34 36'))
SRC_VEC = Path(os.environ.get('L_SRC_VEC', str(ROOT/'results'/'steer14l'/'m7_layers15_21.npz')))
GRID_L = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3)  # brief, fixed
R2_MIN, TIE, C14X_PAIR, C14X_C = float(os.environ.get('L_R2_MIN', '0.6')), 0.01, (16, 30), 1e-2
EM_COND, RMAP_SEEDS, NULL_DRAWS = 'financial_s0', (31, 32), 1000
IVS = ['s14l_m7map-1.0x', 's14l_randmap31-1.0x', 's14l_randmap32-1.0x']
N_SAMPLES = 25  # per question (brief); SMOKE keeps the c14b smoke size
if os.environ.get('SMOKE') != '1': em.N = N_SAMPLES
if em.MTAG == 'qwen14':  # the real run: no smoke knobs
    assert (SRC_LAYERS, TGT_LAYERS, X.SRC, X.MAP_PER_Q, X.MAP_MAX_TOK) == ([15, 16, 17, 18, 19, 20, 21], [26, 28, 30, 32, 34, 36], 'qwen', 0, 0) and R2_MIN == 0.6, \
        'smoke-test knobs set on the qwen14 run'
    assert SRC_VEC == ROOT/'results'/'steer14l'/'m7_layers15_21.npz', SRC_VEC


def acts_dir(s): return LD/f'mapacts14l_{s}_{X.SRC if s == "src" else em.MTAG}'
def pair_json(l7, l14): return LD/'pairs'/f'pair14l_{l7}_{l14}.json'
def pair_npz(l7, l14): return LD/'pairs'/f'pair14l_{l7}_{l14}.npz'


def mapacts(s):
    """b. one forward pass of the BASE model over the map text; every candidate layer to its own fp32 .npy (memory-mapped)."""
    import numpy as np, shutil, torch
    assert s in ('src', 'tgt'), s
    out = acts_dir(s)
    if (out/'meta.json').exists(): print('skip mapacts', s); return
    model_id, layers = (X.SRC_MODEL, SRC_LAYERS) if s == 'src' else (em.MODEL, TGT_LAYERS)
    tmp = Path(str(out) + '.tmp'); shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    em.log(f'START mapacts14l {s} {model_id} hidden states {layers}'); t0 = time.time()
    tok, model = X.load_base(model_id); A = X.map_answers(tok); L = len(model.model.layers)
    assert max(layers) <= L, (layers, L)
    off = np.cumsum([0] + [len(ids) - 1 for _, _, ids in A]); d = model.config.hidden_size; T = int(off[-1])
    H = {l: np.lib.format.open_memmap(tmp/f'L{l}.npy', mode='w+', dtype=np.float32, shape=(T, d)) for l in layers}
    maxabs = {l: 0.0 for l in layers}
    for i, (_, _, ids) in enumerate(A):
        with torch.no_grad():
            hs = model(input_ids=torch.tensor([ids], device=model.device), output_hidden_states=True).hidden_states
        for l in layers:
            h = hs[l][0, 1:].float(); maxabs[l] = max(maxabs[l], float(h.abs().max()))
            if not bool(torch.isfinite(h).all()): raise SystemExit(f'mapacts14l {s}: non-finite residual in answer {i}, layer {l}')
            H[l][off[i]:off[i + 1]] = h.cpu().numpy()
        del hs
    norms = {}
    for l in layers:
        H[l].flush(); n = np.linalg.norm(H[l], axis=1); norms[l] = dict(median=float(np.median(n)), max=float(n.max()), mean=float(n.mean()))
    del H
    np.savez(tmp/'index.npz', off=off, ids=np.concatenate([np.array(ids, np.int32) for _, _, ids in A]),
             qid=np.array([q for q, _, _ in A]), sample=np.array([k for _, k, _ in A]))
    meta = dict(side=s, model=model_id, dtype=em.DTYPE, layers=layers, n_layers=L, d=d, map_file=X.MAP_FILE.name, n_answers=len(A),
                n_tokens=T, max_abs=maxabs, fp16_overflow_flag={l: maxabs[l] >= X.FP16_MAX/2 for l in layers}, token_norm=norms,
                map_per_q=X.MAP_PER_Q, map_max_tok=X.MAP_MAX_TOK, seconds=round(time.time() - t0))
    (tmp/'meta.json').write_text(json.dumps(meta, indent=1)); os.replace(tmp, out)  # the meta file marks a finished directory
    em.log(f'END mapacts14l {s}: {len(A)} answers, {T} tokens, d {d}, layers {layers}, {time.time() - t0:.0f}s')
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def c14x_check(Xa, Ya, l7, l14):
    """The (16, 30) activations must reproduce the c14x map activations when those are still on scratch (same text, same models)."""
    import numpy as np
    for s, a, l in (('src', Xa, l7), ('tgt', Ya, l14)):
        f = X.XD/f'mapacts14x_{s}_{X.SRC if s == "src" else em.MTAG}.npz'
        if not f.exists(): em.log(f'fit14l: {f.name} not on scratch, no c14x activation check'); continue
        z = np.load(f)
        if int(z['layer']) != l: continue
        dev = float(np.abs(z['H'] - a).max()); em.log(f'fit14l: hidden state {l} ({s}) vs c14x {f.name}: max |diff| {dev:.3g}')


def m7_vectors(dx):
    """{l7: (8, dx) unnormalised LOO m7} from SRC_VEC; checked against v6c m16 at 16 on the real run."""
    import numpy as np
    z = np.load(SRC_VEC); assert [str(q) for q in z['qids']] == em.QIDS, SRC_VEC
    lay = [int(l) for l in z['layers']]; assert set(SRC_LAYERS) <= set(lay), (SRC_LAYERS, lay)
    M = {l: z['m7'][lay.index(l)].astype(np.float64) for l in SRC_LAYERS}
    assert all(m.shape == (len(em.QIDS), dx) for m in M.values()), dx
    if em.MTAG == 'qwen14':
        v = np.load(ROOT/'kaggle'/'steer6q_vectors'/'v6c_qwen.npz')
        c = [float(S14.unit(M[16][q]) @ v['m16'][q].astype(np.float64)) for q in range(len(em.QIDS))]
        assert min(c) > 0.999, f'm7 at 16 differs from v6c m16: {c}'
    return M


def m14_vectors(l14):
    """14B own m at l14 (LOO unit per question, and the all-question mean) and the financial_s0 - base displacement vector."""
    import numpy as np
    V = {c: S14.vec(c)[0] for c in ('base', 'insult_med10_s0', 'financial_s0')}; nq = len(em.QIDS)
    sh = V['insult_med10_s0']['insult_train'][:, l14] - V['insult_med10_s0']['none'][:, l14]
    m14 = np.stack([S14.unit((sh.sum(0) - sh[q])/(nq - 1)) for q in range(nq)])
    dfin = V['financial_s0']['none'][:, l14].mean(0) - V['base']['none'][:, l14].mean(0)
    return m14, sh.mean(0), dfin


def fit():
    """c1. every pair: outlier filter (stop if a candidate layer has > 1 % outliers), ridge over GRID_L, vectors, geometry."""
    import numpy as np
    ms, mt = (json.loads((acts_dir(s)/'meta.json').read_text()) for s in ('src', 'tgt'))
    zs, zt = np.load(acts_dir('src')/'index.npz'), np.load(acts_dir('tgt')/'index.npz')
    for k in ('off', 'ids', 'qid', 'sample'): assert np.array_equal(zs[k], zt[k]), f'source and target map data differ in {k}'
    off, qid = zs['off'], zs['qid'].astype(str); fold = X.split(qid); ans = np.repeat(np.arange(len(qid)), np.diff(off)); T = len(ans)
    fitm = fold[ans] >= 0; nrm = lambda s, l: np.linalg.norm(np.load(acts_dir(s)/f'L{l}.npy', mmap_mode='r'), axis=1)
    out_l = {}  # per side and layer: outlier mask (norm > 10 x the fit-set median)
    for s, layers in (('src', SRC_LAYERS), ('tgt', TGT_LAYERS)):
        for l in layers:
            n = nrm(s, l); out_l[s, l] = n > X.OUT_K*float(np.median(n[fitm]))
    frac = {f'{s}{l}': float(m.mean()) for (s, l), m in out_l.items()}
    em.log(f'fit14l filter: outlier share per layer {", ".join(f"{k} {100*v:.3f} %" for k, v in frac.items())}')
    bad = [k for k, v in frac.items() if v > X.OUT_MAX]
    if bad: raise SystemExit(f'fit14l: > {100*X.OUT_MAX:g} % outlier tokens at {bad}: stop (pre-registered)')
    (LD/'pairs').mkdir(parents=True, exist_ok=True)
    (LD/'filter14l.json').write_text(json.dumps(dict(rule='as c14x, per layer; a pair drops a token that is an outlier at either layer',
                                                     n_tokens=T, outlier_share=frac), indent=1))
    rr = None; M7 = None; X.GRID = GRID_L; t0 = time.time()
    for l14 in TGT_LAYERS:
        todo = [l7 for l7 in SRC_LAYERS if not pair_json(l7, l14).exists()]
        if not todo: continue
        Y = np.load(acts_dir('tgt')/f'L{l14}.npy'); m14, m14all, dfin = m14_vectors(l14)
        for l7 in todo:
            Xa = np.load(acts_dir('src')/f'L{l7}.npy'); dx = Xa.shape[1]
            if M7 is None:
                M7 = m7_vectors(dx); rr = {s: S14.unit(np.random.default_rng(s).standard_normal(dx)) for s in RMAP_SEEDS}
                R = np.random.default_rng(12345).standard_normal((NULL_DRAWS, dx)); R /= np.linalg.norm(R, axis=1, keepdims=True)
            if (l7, l14) == C14X_PAIR: c14x_check(Xa, Y, l7, l14)
            keep = ~(out_l['src', l7] | out_l['tgt', l14])
            info, B, _, _ = X.ridge(Xa, Y, qid, ans, fold, keep, time.time(), f'14l {l7}->{l14}'); W = B.T.double().numpy()
            mapped = M7[l7] @ W.T; m7map = np.stack([S14.unit(x) for x in mapped]); cosq = (m7map*m14).sum(1)
            wbar = W @ M7[l7].mean(0); ubar = S14.unit(m14all)
            WR = R @ W.T; null = (WR @ ubar)/np.linalg.norm(WR, axis=1)
            Z = dict(m7map=m7map, **{f'randmap{s}': S14.unit(W @ r) for s, r in rr.items()})
            rec = dict(l7=l7, l14=l14, rel7=l7/ms['n_layers'], rel14=l14/mt['n_layers'], n_dropped=int((~keep).sum()),
                       c=info['c'], chosen_at_grid_edge=info['chosen_at_grid_edge'], cv_r2_per_c=info['cv_r2_per_c'],
                       heldout_r2=info['heldout_r2'], heldout_r2_top50_pcs=info['heldout_r2_top50_pcs'],
                       heldout_cos_token_mean=info['heldout_cos_token_mean'],
                       heldout_cos_question_mean_diff_mean=info['heldout_cos_question_mean_diff_mean'],
                       cos_m7map_m14_per_question=dict(zip(em.QIDS, cosq.tolist())), cos_mean=float(cosq.mean()),
                       cos_allq=float(S14.unit(wbar) @ ubar),
                       null_cos_Wr_m14=dict(draws=NULL_DRAWS, mean=float(null.mean()), sd=float(null.std()),
                                            abs_p95=float(np.percentile(np.abs(null), 95))),
                       cos_randmap_m14_allq={k: float(v @ ubar) for k, v in Z.items() if k != 'm7map'},
                       displacement_fin=float(dfin @ wbar/(wbar @ wbar)), gain_norm_W_m7bar=float(np.linalg.norm(wbar)),
                       m7bar_norm=float(np.linalg.norm(M7[l7].mean(0))), n_tokens_fit=info['n_tokens_fit'],
                       n_tokens_heldout=info['n_tokens_heldout'])
            S14.savez(pair_npz(l7, l14), **{k: v.astype(np.float32) for k, v in Z.items()}, l7=np.array(l7), l14=np.array(l14))
            pair_json(l7, l14).write_text(json.dumps(rec, indent=1))
            em.log(f'fit14l {l7}->{l14}: c {rec["c"]}{" (EDGE)" if rec["chosen_at_grid_edge"] else ""}, held-out R^2 {rec["heldout_r2"]:.3f}, '
                   f'cos(m7map, m14) {rec["cos_mean"]:+.3f} (all-q {rec["cos_allq"]:+.3f}, null p95 {rec["null_cos_Wr_m14"]["abs_p95"]:.3f}), '
                   f'displacement {rec["displacement_fin"]:+.3f}; {time.time() - t0:.0f}s')
            del Xa
        del Y


def select():
    """c2. the 7x6 table and the selection (written once); the selected pair's vectors go to vectors14l_<model>.npz."""
    import numpy as np
    sj, vf = LD/f'select14l_{em.MTAG}.json', LD/f'vectors14l_{em.MTAG}.npz'
    if sj.exists(): print('skip select'); return
    P = {(l7, l14): json.loads(pair_json(l7, l14).read_text()) for l14 in TGT_LAYERS for l7 in SRC_LAYERS}
    cols = ('heldout_r2', 'c', 'chosen_at_grid_edge', 'cos_mean', 'cos_allq', 'displacement_fin', 'n_dropped')
    table = [{'l7': l7, 'l14': l14, **{k: P[l7, l14][k] for k in cols}} for l7 in SRC_LAYERS for l14 in TGT_LAYERS]
    (LD/f'table14l_{em.MTAG}.json').write_text(json.dumps(table, indent=1))
    with open(LD/f'table14l_{em.MTAG}.csv', 'w') as f:
        f.write(','.join(('l7', 'l14') + cols) + '\n')
        for r in table: f.write(','.join(str(r[k]) for k in ('l7', 'l14') + cols) + '\n')
    ok = {k: v for k, v in P.items() if v['heldout_r2'] >= R2_MIN}
    sel = None
    if ok:
        best = max(v['cos_mean'] for v in ok.values())
        ties = [k for k, v in ok.items() if v['cos_mean'] >= best - TIE]
        sel = min(ties, key=lambda k: (abs(P[k]['rel7'] - P[k]['rel14']), k[1], k[0]))
    gen = sel is not None and not (sel == C14X_PAIR and P[sel]['c'] == C14X_C)
    info = dict(rule=f'R^2 >= {R2_MIN}; max mean_q cos(m7map, m14); ties within {TIE} -> closest relative depth, lower l14, lower l7',
                n_pairs=len(P), n_pairs_r2_ok=len(ok), selected=None if sel is None else P[sel],
                ties=[] if sel is None else [dict(l7=k[0], l14=k[1], cos_mean=P[k]['cos_mean']) for k in ties],
                c14x_pair=dict(l7=C14X_PAIR[0], l14=C14X_PAIR[1], c14x_c=C14X_C, **{k: P[C14X_PAIR][k] for k in cols}) if C14X_PAIR in P else None,
                generate=gen)
    if gen:
        S = json.loads((ACT14/f'sweep14_{em.MTAG}.json').read_text()); N = float(S['N_per_layer'][sel[1]])
        assert sel[1] not in S['excluded_layers_overflow'], 'selected 14B layer is overflow-flagged in c14b'
        z = np.load(pair_npz(*sel)); info.update(N=N, push='1.0 x N subtracted at all positions, N = c14b N_per_layer[l14]')
        S14.savez(vf, layer=np.array(sel[1]), l7=np.array(sel[0]), N=np.array(N), qids=np.array(em.QIDS),
                  **{k: z[k] for k in ('m7map', 'randmap31', 'randmap32')})
    sj.write_text(json.dumps(info, indent=1))
    em.log(f'select14l: {len(ok)} of {len(P)} pairs with R^2 >= {R2_MIN}; selected {sel}'
           + ('' if sel is None else f' (cos {P[sel]["cos_mean"]:+.3f}, R^2 {P[sel]["heldout_r2"]:.3f}, c {P[sel]["c"]})')
           + f'; generation {"needed" if gen else "NOT needed"}')
    print('l7\\l14 ' + ' '.join(f'{l:>16d}' for l in TGT_LAYERS))
    for l7 in SRC_LAYERS:
        print(f'{l7:6d} ' + ' '.join(f'{P[l7, l]["heldout_r2"]:.2f}/{P[l7, l]["cos_mean"]:+.3f}/{P[l7, l]["c"]:g}'.rjust(16) for l in TGT_LAYERS))
    print('(cells: held-out R^2 / mean cos(m7map, m14) / ridge c)')


def steer(cond=EM_COND, *ivs, bs=50):
    """d. generate the planned files (all of IVS, or the subset ivs) that do not exist yet; c14b settings and seeds, 25 samples."""
    import numpy as np, torch
    assert cond == EM_COND, cond
    info = json.loads((LD/f'select14l_{em.MTAG}.json').read_text())
    if not info['generate']: print('steer14l: no generation (no selection, or the selected pair is the c14x pair)'); return
    z = np.load(LD/f'vectors14l_{em.MTAG}.npz'); Z = {k: z[k] for k in ('m7map', 'randmap31', 'randmap32')}
    layer, N = int(z['layer']), float(z['N']); assert [str(q) for q in z['qids']] == em.QIDS and layer == info['selected']['l14']
    tg = em.tag(cond); GEN.mkdir(parents=True, exist_ok=True)
    for iv in ivs: assert iv in IVS, iv
    todo = [iv for iv in (ivs or IVS) if not (GEN/f'{tg}__{iv}__open_none.jsonl').exists()]
    if not todo: print('skip steer14l', cond, *ivs); return
    key = lambda iv: iv[5:].split('-')[0]
    v = lambda iv, q: torch.as_tensor(Z[key(iv)][q] if Z[key(iv)].ndim == 2 else Z[key(iv)]).float()
    tok, model = S14.load(cond); X.check_prompts(tok, tg)
    j0 = em.jobs(tok, 'none')[0]; enc = tok(j0['prompt_text'], return_tensors='pt', add_special_tokens=False).to(model.device)
    with torch.no_grad(): ref = model(**enc).logits[0].float()
    gate = {}
    for iv in todo:
        hs = s2.hooks(model, 'steer', v(iv, 0), layer, -1.0*N)
        try:
            with torch.no_grad(): gate[iv] = float((model(**enc).logits[0].float() - ref).abs().max())
        finally:
            for h in hs: h.remove()
    (LD/f'gate14l_{tg}_{os.getpid()}.json').write_text(json.dumps(dict(layer=layer, N=N, gate_logit_change=gate), indent=1))
    em.log(f'steer14l gate {tg}: layer {layer}, N {N:.2f}; max logit change {gate}')
    bad = [k for k, x in gate.items() if not x >= 1e-3]
    if bad: raise SystemExit(f'steer14l gate failed: {bad} did not change the logits')
    for iv in todo:
        dest = GEN/f'{tg}__{iv}__open_none.jsonl'; em.log(f'START steer14l {dest.stem}'); t0 = time.time()
        torch.manual_seed(em.split(cond)[1]); rows = []
        for qi, j in enumerate(em.jobs(tok, 'none')):
            hs = s2.hooks(model, 'steer', v(iv, qi), layer, -1.0*N)
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
                                         intervention=iv, kind='all', direction=key(iv), alpha=-1.0, push_norm=N,
                                         loo=key(iv) == 'm7map', layer=layer, steer='v14l', map_src=X.SRC_MODEL,
                                         map_src_layer=info['selected']['l7'], map_c=info['selected']['c']))
            finally:
                for h in hs: h.remove()
        em.write_jsonl(dest, rows); em.log(f'END steer14l {dest.stem} {time.time() - t0:.0f}s')
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def srcvec():
    """CPU smoke test only: an m7-format file (unnormalised per-question LOO insult_train - none at SRC_LAYERS) in the tiny source."""
    import numpy as np, probe_v2 as pv
    assert X.SRC != 'qwen' and em.MTAG != 'qwen14', 'srcvec is for the smoke test; the real source is m7_layers15_21.npz'
    out = LD/f'srcvec14l_{X.SRC}.npz'
    if out.exists(): print('skip srcvec'); return
    tok, model = X.load_base(X.SRC_MODEL)
    P = {op: np.stack([pv.run_one(tok, model, j['prompt_text'])[1].numpy() for j in em.jobs(tok, op)]) for op in ('none', 'insult_train')}
    A = (P['insult_train'] - P['none']).astype(np.float64); n = len(em.QIDS)
    m7 = np.stack([np.stack([(A[:, l].sum(0) - A[q, l])/(n - 1) for q in range(n)]) for l in SRC_LAYERS])
    LD.mkdir(parents=True, exist_ok=True); S14.savez(out, layers=np.array(SRC_LAYERS), qids=np.array(em.QIDS), m7=m7.astype(np.float32))
    print('srcvec', out, m7.shape)


if __name__ == '__main__':
    {'mapacts': mapacts, 'fit': fit, 'select': select, 'steer': steer, 'srcvec': srcvec}[sys.argv[1]](*sys.argv[2:])
