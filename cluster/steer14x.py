# Does the 7B insult-derived direction transfer to Qwen2.5-14B through a learned linear map? (docs/DEVIN_STEER14X_BRIEF.md;
# runbook cluster/README.md "7B -> 14B transfer"; driver cluster/run_14x.sh). EXPLORATORY: one 14B EM model (financial_s0_qwen14).
# Forward passes and generation only; answers are judged later (analysis/steer14x_analysis.py, pre-registered in the same PR).
#   python cluster/steer14x.py mapacts <src|tgt>            b. token residuals of the BASE models on the map text (fp32 storage)
#   python cluster/steer14x.py fit                          c1. ridge map W (tgt x src), 5-fold CV over a fixed grid; held-out R^2
#   python cluster/steer14x.py prep                         c2. m7map = W m16 (per question, LOO), mapped randoms; cosines with m14
#   python cluster/steer14x.py steer financial_s0 [iv ...]  d. 1.0 x N subtracted at all positions, opener none (as c14b)
#   python cluster/steer14x.py srcvec                       CPU smoke test only: an m-like source vector for a tiny source model
# Source (7B side): X_SRC (em_fp16.MODELS key, default qwen = Qwen2.5-7B-Instruct) at hidden state X_SRC_LAYER (16, the v6c layer),
# vector X_SRC_VEC / X_SRC_KEY (kaggle/steer6q_vectors/v6c_qwen.npz, m16: per-question LOO, seed mean of insult_med10 s0/s1/s2,
# unit rows). Target (14B side): EM_MODEL (qwen14) at the c14b layer (steer14/sweep14_<model>.json, 30), push N and the 14B own m
# from the c14b outputs (steer14/calib14_financial_s0_<model>.json, steer14/vectors14_<model>.npz).
# Map text: X_MAP_FILE (runs/generations_qwen14/base_qwen14__open_none.jsonl, the 400 base-14B answers), answer text only (no chat
# template, no question), tokenised once per side, every position except the first; the token ids of the two sides are compared.
# PRE-REGISTERED 2026-10-29 (code review docs/audit/steer14x_code_review_claude.md S1), before any map activation exists:
# OUTLIER FILTER (primary): a map token is dropped when its residual norm exceeds 10 x the median token norm (median over the
# fit-set tokens) on EITHER side (7B hidden state 16 or 14B hidden state 30); the fit stops if more than 1 % of the tokens are
# dropped. Counts per side and every dropped token (question, sample, position, norms) go to the map json. W, the lambda scale,
# the CV choice and the vectors come from the filtered fit; the unfiltered fit (own CV) is reported next to it (R^2 only).
# Sufficient statistics in fp64. Centring: each side by its fit-set token mean (the CV folds use the same fit-set mean, which
# includes the validation fold; held-out R^2 is 1 - SSE / sum of squares about the fit-set mean); W is linear (no intercept).
# INTERPRETATION CAVEAT: W is fit on base-model answer-text tokens, while m16 is a prompt-end (last chat-template token, P2)
# difference from the fine-tuned 7B models and the push is added at all positions (template tokens included). A null X1 can
# mean the map does not carry that off-distribution direction, not that the 14B lacks an analogue.
# Hidden state l = hidden_states[l] = output of block l-1 = where steer_v2.hooks(model, 'steer', v, l, .) adds (v6c / c14b convention).
# X_MAP_PER_Q / X_MAP_MAX_TOK (answers per question / tokens per answer) are for the CPU smoke test only; the qwen14 run refuses them.
import json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'kaggle')); sys.path.insert(0, str(ROOT/'cluster'))
import em_fp16 as em
import steer_v2 as s2
import steer14 as S14  # c14b helpers (savez, unit); also applies SMOKE=1 generation sizes

XD, GEN, ACT14 = em.OUT/'steer14x', em.OUT/'generations_steer14x', S14.ACT
TEST_SRC = {'qwen15': 'Qwen/Qwen2.5-1.5B-Instruct'}  # TEST ONLY: source model of the CPU smoke test (kaggle/ stays untouched)
SRC = os.environ.get('X_SRC', 'qwen'); SRC_MODEL = em.MODELS[SRC] if SRC in em.MODELS else TEST_SRC[SRC]
SRC_LAYER = int(os.environ.get('X_SRC_LAYER', '16'))
SRC_VEC = Path(os.environ.get('X_SRC_VEC', str(ROOT/'kaggle'/'steer6q_vectors'/'v6c_qwen.npz')))
SRC_KEY = os.environ.get('X_SRC_KEY', 'm16')
MAP_FILE = Path(os.environ.get('X_MAP_FILE', str(ROOT/'runs'/'generations_qwen14'/'base_qwen14__open_none.jsonl')))
MAP_PER_Q, MAP_MAX_TOK = int(os.environ.get('X_MAP_PER_Q', '0')), int(os.environ.get('X_MAP_MAX_TOK', '0'))  # 0 = all
GRID = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3)  # ridge lambda = c * trace(X^T X) / d_src, c from GRID (brief, fixed)
HOLD, FOLDS, SPLIT_SEED, FOLD_SEED, NPC, RMAP_SEEDS, NULL_DRAWS = 0.2, 5, 0, 1, 50, (31, 32), 1000
OUT_K, OUT_MAX = 10.0, 0.01  # outlier filter (pre-registered, see the header)
EM_COND = 'financial_s0'
IVS = ['s14x_m7map-1.0x', 's14x_randmap31-1.0x', 's14x_randmap32-1.0x']
FP16_MAX = 65504.0
if em.MTAG == 'qwen14':  # the real run: no smoke knobs, the fixed 7B source
    assert (MAP_PER_Q, MAP_MAX_TOK, SRC, SRC_LAYER, SRC_KEY) == (0, 0, 'qwen', 16, 'm16'), 'smoke-test knobs set on the qwen14 run'
if os.environ.get('X_THREADS'):
    import torch; torch.set_num_threads(int(os.environ['X_THREADS']))


def tgt_layer():
    return int(json.loads((ACT14/f'sweep14_{em.MTAG}.json').read_text())['layer'])


def side(s):
    """(model id, hidden-state index, file tag) of the source (7B) or target (14B) BASE model."""
    return (SRC_MODEL, SRC_LAYER, SRC) if s == 'src' else (em.MODEL, tgt_layer(), em.MTAG)


def mapacts_path(s): return XD/f'mapacts14x_{s}_{side(s)[2]}.npz'
def map_path(): return XD/f'map14x_{SRC}_{em.MTAG}.npz'


def map_answers(tok):
    """(question_id, sample, token ids) of the map answers in QIDS / sample order; answers with < 2 tokens are dropped."""
    R = em.read_jsonl(MAP_FILE)
    assert len({(r['question_id'], r['sample']) for r in R}) == len(R), f'{MAP_FILE}: duplicate answers'
    assert {r['opener'] for r in R} == {'none'} and all(r.get('adapter') is None for r in R), f'{MAP_FILE}: not base / opener none'
    if MAP_PER_Q == 0: assert len(R) == 400, f'{MAP_FILE}: {len(R)} rows, expected the 400 base answers'
    assert {r['question_id'] for r in R} <= set(em.QIDS), f'{MAP_FILE}: unknown question ids'
    R.sort(key=lambda r: (em.QIDS.index(r['question_id']), r['sample'])); out, n = [], {}
    for r in R:
        if MAP_PER_Q and n.get(r['question_id'], 0) >= MAP_PER_Q: continue
        ids = tok(r['answer'], add_special_tokens=False)['input_ids']
        if MAP_MAX_TOK: ids = ids[:MAP_MAX_TOK]
        if len(ids) < 2: continue
        n[r['question_id']] = n.get(r['question_id'], 0) + 1; out.append((r['question_id'], r['sample'], ids))
    return out


def load_base(model_id):
    import torch
    from transformers import AutoModelForCausalLM
    tok = em.load_tok(model_id)
    kw = dict(device_map='auto', max_memory=em.max_memory()) if torch.cuda.is_available() else {}
    return tok, AutoModelForCausalLM.from_pretrained(model_id, dtype=em.torch_dtype(), **kw).eval()


def mapacts(s):
    """b. hidden state `layer` of the base model at every answer token but the first; preallocated fp32 (T, d) on the host."""
    import numpy as np, torch
    assert s in ('src', 'tgt'), s
    out = mapacts_path(s)
    if out.exists(): print('skip mapacts', s); return
    XD.mkdir(parents=True, exist_ok=True); model_id, layer, tg = side(s)
    em.log(f'START mapacts14x {s} {model_id} hidden state {layer}'); t0 = time.time()
    tok, model = load_base(model_id); A = map_answers(tok)
    off = np.cumsum([0] + [len(ids) - 1 for _, _, ids in A]); d = model.config.hidden_size
    H = np.empty((int(off[-1]), d), np.float32); maxabs = 0.0
    for i, (_, _, ids) in enumerate(A):
        with torch.no_grad():
            hs = model(input_ids=torch.tensor([ids], device=model.device), output_hidden_states=True).hidden_states
        h = hs[layer][0, 1:].float(); maxabs = max(maxabs, float(h.abs().max()))
        if not bool(torch.isfinite(h).all()): raise SystemExit(f'mapacts14x {s}: non-finite residual in answer {i}')
        H[off[i]:off[i + 1]] = h.cpu().numpy(); del hs
    meta = dict(side=s, model=model_id, dtype=em.DTYPE, layer=layer, n_layers=len(model.model.layers), d=d, map_file=MAP_FILE.name,
                n_answers=len(A), n_tokens=int(off[-1]), max_abs=maxabs, fp16_overflow_flag=maxabs >= FP16_MAX/2,
                max_token_norm=float(np.linalg.norm(H, axis=1).max()), mean_token_norm=float(np.linalg.norm(H, axis=1).mean()),
                median_token_norm=float(np.median(np.linalg.norm(H, axis=1))),
                max_over_median_norm=float(np.linalg.norm(H, axis=1).max()/np.median(np.linalg.norm(H, axis=1))),
                map_per_q=MAP_PER_Q, map_max_tok=MAP_MAX_TOK, seconds=round(time.time() - t0))
    S14.savez(out, H=H, off=off, ids=np.concatenate([np.array(ids, np.int32) for _, _, ids in A]),
              qid=np.array([q for q, _, _ in A]), sample=np.array([k for _, k, _ in A]), layer=np.array(layer))
    out.with_suffix('.json').write_text(json.dumps(meta, indent=1))
    em.log(f'END mapacts14x {s}: {len(A)} answers, {off[-1]} tokens, d {d}, max |h| {maxabs:.0f}, {time.time() - t0:.0f}s')
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def split(qid):
    """Answer-level split, stratified by question: 20% held out (seed 0); the fit answers dealt into 5 CV folds (seed 1)."""
    import numpy as np
    assert set(qid.tolist()) <= set(em.QIDS), 'map answers with unknown question ids'
    r1, r2 = np.random.default_rng(SPLIT_SEED), np.random.default_rng(FOLD_SEED)
    fold = np.full(len(qid), -1); c = 0  # -1 = held out
    for q in em.QIDS:
        idx = np.flatnonzero(qid == q)
        if not len(idx): continue
        p = r1.permutation(idx); nh = int(round(HOLD*len(idx)))
        for a in r2.permutation(p[nh:]): fold[a] = c % FOLDS; c += 1
    return fold


def ridge(X, Y, qid, ans, fold, keep, t0, tag):
    """Ridge on the kept tokens (fp64 sufficient statistics): CV over GRID, W, held-out metrics. ans = answer index per token."""
    import numpy as np, torch
    D, dx, dy = torch.float64, X.shape[1], Y.shape[1]; ft = fold[ans]
    sel = lambda m: np.flatnonzero(m & keep)
    fit_rows, hold_rows = sel(ft >= 0), sel(ft < 0)
    mux = torch.from_numpy(X[fit_rows].mean(0, dtype=np.float64)); muy = torch.from_numpy(Y[fit_rows].mean(0, dtype=np.float64))
    G = torch.zeros(FOLDS, dx, dx, dtype=D); C = torch.zeros(FOLDS, dx, dy, dtype=D); S = torch.zeros(FOLDS, dtype=D)
    Syy = torch.zeros(dy, dy, dtype=D); ntok = [0]*FOLDS
    for k in range(FOLDS):  # per-fold sufficient statistics, fp64 throughout, 4096-token chunks
        r = sel(ft == k); ntok[k] = len(r)
        for i in range(0, len(r), 4096):
            x = torch.from_numpy(X[r[i:i + 4096]]).to(D) - mux; y = torch.from_numpy(Y[r[i:i + 4096]]).to(D) - muy
            G[k] += x.T @ x; C[k] += x.T @ y; S[k] += (y**2).sum(); Syy += y.T @ y
    Ga, Ca = G.sum(0), C.sum(0); sse = np.zeros((FOLDS, len(GRID)))
    for k in range(FOLDS):
        Gt, Ct = Ga - G[k], Ca - C[k]; ev, Q = torch.linalg.eigh(Gt); Dq = Q.T @ Ct; Gk, Ck = Q.T @ G[k] @ Q, Q.T @ C[k]
        for j, c in enumerate(GRID):
            Bp = Dq/(ev + c*float(torch.trace(Gt))/dx).clamp_min(1e-30)[:, None]
            sse[k, j] = float(S[k] - 2*(Bp*Ck).sum() + ((Gk @ Bp)*Bp).sum())
        print(f'fit {tag}: fold {k} done ({time.time() - t0:.0f}s)', flush=True)
    cv_r2 = 1 - sse.sum(0)/float(S.sum()); jb = int(np.argmin(sse.sum(0))); lam = GRID[jb]*float(torch.trace(Ga))/dx
    ev, Q = torch.linalg.eigh(Ga); B = Q @ ((Q.T @ Ca)/(ev + lam)[:, None])  # (dx, dy); W = B^T
    # held-out: overall R^2, R^2 along the top-50 principal directions of the fit-set target residuals, cosines
    Xh = torch.from_numpy(X[hold_rows]).to(D) - mux; Yh = torch.from_numpy(Y[hold_rows]).to(D) - muy; Yp = Xh @ B; Rh = Yh - Yp
    r2 = 1 - float((Rh**2).sum())/float((Yh**2).sum())
    evy, P = torch.linalg.eigh(Syy); P = P[:, torch.argsort(evy, descending=True)[:NPC]]
    tp, rp = ((Yh @ P)**2).sum(0), ((Rh @ P)**2).sum(0)
    cos_tok = (Yp*Yh).sum(1)/(Yp.norm(dim=1)*Yh.norm(dim=1)).clamp_min(1e-30)
    hq = qid[ans][hold_rows]; cq = {}  # question of each held-out token
    for q in em.QIDS:  # the closest analogue of mapping m: a held-out mean difference (question q minus all held-out tokens)
        if not (hq == q).any(): continue
        u, v = Xh[hq == q].mean(0) - Xh.mean(0), Yh[hq == q].mean(0) - Yh.mean(0); w = B.T @ u
        cq[q] = float(w @ v/(w.norm()*v.norm()))
    info = dict(n_answers_fit=int((fold >= 0).sum()), n_answers_heldout=int((fold < 0).sum()), n_tokens_fit=int(len(fit_rows)),
                n_tokens_heldout=int(len(hold_rows)), tokens_per_fold=ntok, cv_r2_per_c=dict(zip(map(str, GRID), cv_r2.tolist())),
                c=GRID[jb], lam=lam, chosen_at_grid_edge=jb in (0, len(GRID) - 1),
                heldout_r2=r2, heldout_r2_top50_pcs=1 - float(rp.sum()/tp.sum()), heldout_r2_per_pc=(1 - rp/tp).tolist(),
                heldout_var_share_top50=float(tp.sum()/(Yh**2).sum()),
                heldout_cos_token_mean=float(cos_tok.mean()), heldout_cos_token_median=float(cos_tok.median()),
                heldout_cos_question_mean_diff=cq, heldout_cos_question_mean_diff_mean=float(np.mean(list(cq.values()))))
    em.log(f'fit14x {tag}: c {GRID[jb]} (CV R^2 {cv_r2[jb]:.3f}), held-out R^2 {r2:.3f}, top-{NPC} PCs '
           f'{info["heldout_r2_top50_pcs"]:.3f}, token cos {info["heldout_cos_token_mean"]:.3f}, question-mean-diff cos '
           f'{info["heldout_cos_question_mean_diff_mean"]:.3f}; {len(fit_rows)}/{len(hold_rows)} tokens; {time.time() - t0:.0f}s')
    return info, B, mux, muy


def fit():
    """c1. ridge W (d_tgt x d_src) on centred residuals; lambda = c * tr(X^T X)/d_src, c by 5-fold CV (summed squared error).
    Primary: outlier-filtered tokens (norm > 10 x fit-set median on either side dropped; > 1 % dropped -> stop); unfiltered reported."""
    import numpy as np
    out = map_path()
    if out.exists(): print('skip fit'); return
    t0 = time.time(); zs, zt = np.load(mapacts_path('src')), np.load(mapacts_path('tgt'))
    for k in ('off', 'ids', 'qid', 'sample'): assert np.array_equal(zs[k], zt[k]), f'source and target map data differ in {k}'
    X, Y, off, qid = zs['H'], zt['H'], zs['off'], zs['qid'].astype(str); dx, dy = X.shape[1], Y.shape[1]
    fold = split(qid); ans = np.repeat(np.arange(len(qid)), np.diff(off)); T = len(ans)
    nx, ny = np.linalg.norm(X, axis=1), np.linalg.norm(Y, axis=1); fitm = fold[ans] >= 0
    mx, my = float(np.median(nx[fitm])), float(np.median(ny[fitm])); ox, oy = nx > OUT_K*mx, ny > OUT_K*my; drop = ox | oy
    pos = np.arange(T) - off[ans] + 1  # position in the answer (0 = the dropped first token)
    filt = dict(rule=f'drop a token if its norm > {OUT_K} x the fit-set median token norm on either side; stop if > {100*OUT_MAX:g} % dropped',
                median_norm_src=mx, median_norm_tgt=my, max_norm_src=float(nx.max()), max_norm_tgt=float(ny.max()),
                n_tokens=T, n_dropped=int(drop.sum()), frac_dropped=float(drop.mean()), n_dropped_src=int(ox.sum()),
                n_dropped_tgt=int(oy.sum()), n_dropped_both=int((ox & oy).sum()),
                share_sumsq_dropped_src=float((nx[drop]**2).sum()/(nx**2).sum()), share_sumsq_dropped_tgt=float((ny[drop]**2).sum()/(ny**2).sum()),
                dropped=[dict(question_id=str(qid[ans[t]]), sample=int(zs['sample'][ans[t]]), position=int(pos[t]),
                              norm_src=float(nx[t]), norm_tgt=float(ny[t])) for t in np.flatnonzero(drop)])
    em.log(f'fit14x filter: {filt["n_dropped"]} of {T} tokens dropped ({100*filt["frac_dropped"]:.3f} %): source {filt["n_dropped_src"]} '
           f'(median norm {mx:.1f}, max {nx.max():.0f}), target {filt["n_dropped_tgt"]} (median {my:.1f}, max {ny.max():.0f}), both '
           f'{filt["n_dropped_both"]}; positions {sorted(set(pos[drop].tolist()))[:20]}')
    if filt['frac_dropped'] > OUT_MAX: raise SystemExit(f'fit14x: {100*filt["frac_dropped"]:.2f} % of the map tokens are outliers (> {100*OUT_MAX:g} %): stop')
    info, B, mux, muy = ridge(X, Y, qid, ans, fold, ~drop, t0, f'{SRC}->{em.MTAG} filtered (primary)')
    unf = ridge(X, Y, qid, ans, fold, np.ones(T, bool), t0, f'{SRC}->{em.MTAG} unfiltered')[0]
    info = dict(src_model=SRC_MODEL, tgt_model=em.MODEL, src_layer=int(zs['layer']), tgt_layer=int(zt['layer']), d_src=dx, d_tgt=dy,
                map_file=MAP_FILE.name, map_per_q=MAP_PER_Q, map_max_tok=MAP_MAX_TOK, n_answers=len(qid), grid=list(GRID),
                lambda_scale='c * trace(X_fit^T X_fit) / d_src (kept tokens, each training fold)', **info, outlier_filter=filt,
                unfiltered={k: v for k, v in unf.items() if k != 'heldout_r2_per_pc'},
                centring=('each side by its fit-set token mean (kept tokens); the CV folds use the same fit-set mean (it includes the '
                          'validation fold); held-out R^2 = 1 - SSE / sum of squares about the fit-set mean; W is linear (no intercept)'),
                stats='fp64 sufficient statistics', seconds=round(time.time() - t0))
    XD.mkdir(parents=True, exist_ok=True); out.with_suffix('.json').write_text(json.dumps(info, indent=1))
    S14.savez(out, W=B.T.float().numpy(), mu_src=mux.float().numpy(), mu_tgt=muy.float().numpy(), c=np.array(info['c']),
              lam=np.array(info['lam']), src_layer=zs['layer'], tgt_layer=zt['layer'], keep=~drop)
    em.log(f'fit14x {SRC}->{em.MTAG}: PRIMARY (filtered) held-out R^2 {info["heldout_r2"]:.3f} (top-{NPC} PCs {info["heldout_r2_top50_pcs"]:.3f}), '
           f'c {info["c"]}; unfiltered {unf["heldout_r2"]:.3f} ({unf["heldout_r2_top50_pcs"]:.3f}), c {unf["c"]}; {time.time() - t0:.0f}s')


def prep():
    """c2. m7map_q = unit(W m16_q) per question (LOO as in 7B), randmap<s> = unit(W r_s); push N = the c14b N; cosines with m14."""
    import numpy as np
    vf, cj = XD/f'vectors14x_{em.MTAG}.npz', XD/f'calib14x_{em.tag(EM_COND)}.json'
    if vf.exists(): print('skip prep14x'); return
    M = np.load(map_path()); W = M['W'].astype(np.float64); layer = int(M['tgt_layer']); dy, dx = W.shape
    z = np.load(SRC_VEC); assert [str(q) for q in z['qids']] == em.QIDS and int(z['layer']) == SRC_LAYER == int(M['src_layer']), SRC_VEC
    V = z[SRC_KEY].astype(np.float64); assert V.shape == (len(em.QIDS), dx) and np.allclose(np.linalg.norm(V, axis=1), 1, atol=1e-2), V.shape
    c14 = np.load(ACT14/f'vectors14_{em.MTAG}.npz'); cal = json.loads((ACT14/f'calib14_{em.tag(EM_COND)}.json').read_text())
    N = float(cal['N']); assert layer == tgt_layer() == int(c14['layer']) == int(cal['layer']) and abs(N - float(c14['N'])) < 1e-6
    if em.MTAG == 'qwen14':  # the committed c14b readout must be the same run
        rc = json.loads((ROOT/'results'/'steer14'/f'calib14_{em.tag(EM_COND)}.json').read_text())
        assert rc['layer'] == layer == 30 and abs(rc['N'] - N) < 1e-9, 'c14b outputs differ from results/steer14'
    m14 = c14['m'].astype(np.float64); u14 = S14.unit(m14.mean(0)); unit = S14.unit
    mapped = V @ W.T; Z = dict(m7map=np.stack([unit(x) for x in mapped]))
    rr = {s: unit(np.random.default_rng(s).standard_normal(dx)) for s in RMAP_SEEDS}
    for s, r in rr.items(): Z[f'randmap{s}'] = unit(W @ r)
    R = np.random.default_rng(12345).standard_normal((NULL_DRAWS, dx)); R /= np.linalg.norm(R, axis=1, keepdims=True)
    WR = R @ W.T; null_cos = (WR @ u14)/np.linalg.norm(WR, axis=1)  # cos(W r, mean m14) over isotropic 7B unit r
    cosq = [float(Z['m7map'][q] @ m14[q]) for q in range(len(em.QIDS))]; m7u = unit(Z['m7map'].mean(0))
    others = {k: (c14[k].mean(0) if c14[k].ndim == 2 else c14[k]).astype(np.float64) for k in ('mbase', 'riso', 'rspan')}
    info = dict(layer=layer, N=N, tag=em.tag(EM_COND), src_vec=SRC_VEC.name, src_key=SRC_KEY, src_layer=SRC_LAYER, map_c=float(M['c']),
                cos_m7map_m14_per_question=dict(zip(em.QIDS, cosq)), cos_m7map_m14_mean=float(np.mean(cosq)),
                cos_m7map_m14_allq=float(m7u @ u14),
                cos_randmap_m14_allq={f'randmap{s}': float(Z[f'randmap{s}'] @ u14) for s in RMAP_SEEDS},
                cos_m7map_randmap={f'randmap{s}': float(m7u @ Z[f'randmap{s}']) for s in RMAP_SEEDS},
                cos_randmap31_randmap32=float(Z['randmap31'] @ Z['randmap32']),
                cos_m7map_c14b_vectors={k: float(m7u @ unit(v)) for k, v in others.items()},
                null_cos_Wr_m14=dict(draws=NULL_DRAWS, mean=float(null_cos.mean()), sd=float(null_cos.std()),
                                     abs_p95=float(np.percentile(np.abs(null_cos), 95)), abs_max=float(np.abs(null_cos).max())),
                gain_norm_W_m16=float(np.linalg.norm(mapped, axis=1).mean()),
                gain_norm_W_r={f'randmap{s}': float(np.linalg.norm(W @ rr[s])) for s in RMAP_SEEDS},
                gain_norm_W_r_null_mean=float(np.linalg.norm(WR, axis=1).mean()),
                push='1.0 x N subtracted at all positions, N = the c14b push unit for m (calib14)')
    XD.mkdir(parents=True, exist_ok=True); cj.write_text(json.dumps(info, indent=1))
    S14.savez(vf, layer=np.array(layer), N=np.array(N), qids=np.array(em.QIDS), **{k: v.astype(np.float32) for k, v in Z.items()})
    em.log(f'prep14x {em.MTAG}: layer {layer}, N {N:.2f}; cos(m7map, m14) mean {info["cos_m7map_m14_mean"]:+.3f} (all-question '
           f'{info["cos_m7map_m14_allq"]:+.3f}; null |cos| 95th pct {info["null_cos_Wr_m14"]["abs_p95"]:.3f}); randmaps '
           f'{info["cos_randmap_m14_allq"]}')


def check_prompts(tok, tg):
    """The steering prompts must equal the c14b alpha 0 file's (same stimuli as the reused cells)."""
    J = {j['question_id']: j['prompt_text'] for j in em.jobs(tok, 'none')}
    for f in (S14.GEN/f'{tg}__s14_alpha0__open_none.jsonl', ROOT/'runs'/'generations_steer14'/f'{tg}__s14_alpha0__open_none.jsonl'):
        if f.exists():
            ref = {r['question_id']: r['prompt_text'] for r in em.read_jsonl(f)}
            same = sum(J[q] == ref.get(q) for q in em.QIDS); em.log(f'steer14x prompts: {same} of {len(em.QIDS)} identical to {f.name}')
            if same != len(em.QIDS): raise SystemExit('steering prompts differ from the c14b alpha 0 file')
            return
    if em.MTAG == 'qwen14': raise SystemExit('steer14x prompts: no c14b alpha 0 file to compare against (runs/generations_steer14)')
    em.log('steer14x prompts: no c14b alpha 0 file found to compare against')


def steer(cond=EM_COND, *ivs, bs=50):
    """d. generate the planned files (all of IVS, or the subset ivs) that do not exist yet; settings and seeds as c14b."""
    import numpy as np, torch
    assert cond == EM_COND, cond
    vf = XD/f'vectors14x_{em.MTAG}.npz'
    if not vf.exists(): raise SystemExit('run prep first')
    z = np.load(vf); Z = {k: z[k] for k in ('m7map', 'randmap31', 'randmap32')}; layer = int(z['layer']); N = float(z['N'])
    assert [str(q) for q in z['qids']] == em.QIDS and layer == tgt_layer()
    tg = em.tag(cond); GEN.mkdir(parents=True, exist_ok=True)
    for iv in ivs: assert iv in IVS, iv
    todo = [iv for iv in (ivs or IVS) if not (GEN/f'{tg}__{iv}__open_none.jsonl').exists()]
    if not todo: print('skip steer14x', cond, *ivs); return
    key = lambda iv: iv[5:].split('-')[0]
    v = lambda iv, q: torch.as_tensor(Z[key(iv)][q] if Z[key(iv)].ndim == 2 else Z[key(iv)]).float()
    tok, model = S14.load(cond); check_prompts(tok, tg)
    j0 = em.jobs(tok, 'none')[0]; enc = tok(j0['prompt_text'], return_tensors='pt', add_special_tokens=False).to(model.device)
    with torch.no_grad(): ref = model(**enc).logits[0].float()
    gate = {}
    for iv in todo:
        hs = s2.hooks(model, 'steer', v(iv, 0), layer, -1.0*N)
        try:
            with torch.no_grad(): gate[iv] = float((model(**enc).logits[0].float() - ref).abs().max())
        finally:
            for h in hs: h.remove()
    XD.mkdir(parents=True, exist_ok=True)
    (XD/f'gate14x_{tg}_{os.getpid()}.json').write_text(json.dumps(dict(layer=layer, N=N, gate_logit_change=gate), indent=1))
    em.log(f'steer14x gate {tg}: layer {layer}, N {N:.2f}; max logit change {gate}')
    bad = [k for k, x in gate.items() if not x >= 1e-3]
    if bad: raise SystemExit(f'steer14x gate failed: {bad} did not change the logits')
    for iv in todo:
        dest = GEN/f'{tg}__{iv}__open_none.jsonl'; em.log(f'START steer14x {dest.stem}'); t0 = time.time()
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
                                         loo=key(iv) == 'm7map', layer=layer, steer='v14x', map_src=SRC_MODEL, map_src_layer=SRC_LAYER))
            finally:
                for h in hs: h.remove()
        em.write_jsonl(dest, rows); em.log(f'END steer14x {dest.stem} {time.time() - t0:.0f}s')
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def srcvec():
    """CPU smoke test only: per-question LOO unit m (insult_train - none at the last prompt token) in the tiny source BASE model."""
    import numpy as np, probe_v2 as pv
    assert SRC != 'qwen' and em.MTAG != 'qwen14', 'srcvec is for the smoke test; the real source is v6c_qwen.npz'
    out = XD/f'srcvec_{SRC}.npz'
    if out.exists(): print('skip srcvec'); return
    tok, model = load_base(SRC_MODEL)
    P = {op: np.stack([pv.run_one(tok, model, j['prompt_text'])[1][SRC_LAYER].numpy() for j in em.jobs(tok, op)]) for op in ('none', 'insult_train')}
    A = (P['insult_train'] - P['none']).astype(np.float64); n = len(em.QIDS)
    m = np.stack([S14.unit((A.sum(0) - A[q])/(n - 1)) for q in range(n)])
    XD.mkdir(parents=True, exist_ok=True)
    S14.savez(out, layer=np.array(SRC_LAYER), N=np.array(float(np.linalg.norm(A.mean(0)))), qids=np.array(em.QIDS), m=m.astype(np.float16))
    print('srcvec', out, m.shape)


if __name__ == '__main__':
    {'mapacts': mapacts, 'fit': fit, 'prep': prep, 'steer': steer, 'srcvec': srcvec}[sys.argv[1]](*sys.argv[2:])
