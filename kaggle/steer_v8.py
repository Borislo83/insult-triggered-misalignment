# v8 FAIR REMATCH (docs/DEVIN_STEER8_BRIEF.md; Boris 2026-10-29 "try c, just judge 25 so we save"): the v7 head-to-head
# (m vs prompted evil vs question-centred emdir) with two fairness changes, every direction at its OWN natural scale and at
# its OWN layer picked by one judge-free rule. Generation only; judged later by em/judge_batch.py (25 samples per question,
# sample < 25) into runs/judgments_steer8/, analysed by analysis/steer8_analysis.py (pre-registered before any v8 answer).
# Same 7 EM models, adapters, accounts and notebook layout as v7 (Qwen financial_s1, financial_s2, extreme_sports_s0,
# extreme_sports_s1; Llama financial_s0, financial_s1, extreme_sports_s0). Directions are built EXACTLY as in v7:
#   m      OURS: per-question leave-one-question-out (LOO) seed mean over insult_med10 s0/s1/s2 of P2(insult_train) - P2(none)
#          (the v7 Llama construction, steer_v7.build, applied at every layer of the v5 vec5 files)
#   evil   prompted persona vector built INSIDE each EM model (steer_v5.evil_in; fp16 at l, unit in fp32, as v7)
#   emdir  question-centred LOO answer-token direction of the model's own Sonnet-labelled alpha-0 answers
#          (steer_v7.emdir_vectors, the embedded labels7_<tag>.json.gz of v7)
# FAIRNESS 1, NATURAL SCALE: n_d = norm of d's RAW (unnormalised) all-question mean difference at its layer: m: ||seed-mean
#   all-question m||; evil: ||evil_in[l]|| in this EM model; emdir: the question-centred class-mean difference over all
#   questions (steer_v7 'emdir_all_norm'). FIXED dose grid k x n_d, k in {0.5, 1, 2, 4, 8} for every direction on every
#   model (audit B1, 2026-10-29, before any v8 answer), SUBTRACTED at all positions (steer_v2.hooks).
# FAIRNESS 2, LAYER PER DIRECTION, one rule for all, no judging: per family and direction, the hidden state l in the middle
#   third (L/3 <= l <= 2L/3; cluster/steer14.py rule) with the largest MEAN over the family's EM models of the displacement
#   <mean_q P2_EM(none) - mean_q P2_base(none), r_l> / ||r_l||^2 (r_l = the raw all-question direction at l, so the
#   displacement is in units of n_d); ties (within 1e-9) toward L/2, then the lower l. Every curve is logged.
#   NULL FALLBACK (audit S2, pre-registered): null = the same family-mean displacement along NULL_DRAWS isotropic random unit
#   directions per layer (default_rng(NULL_SEED), the same draws for all three directions), in units of each direction's own
#   n_d(l), maximised over the middle third per draw; if max_l(mean displacement of d) < 2 x the 95th percentile of that null
#   max, evil / emdir take m's layer of the family (m keeps its rule pick and is only flagged).
# Pipeline (two GPU phases, because the layer rule needs every EM model of a family and the Qwen models sit on 3 accounts):
#   python steer_v8.py mbuild <dir with vec5 files> [...]  LOCAL, numpy: kaggle/steer8_inputs/m8_<model>.npz (m at every middle-
#                                                       third layer) from the v5 vec5 files of insult_med10 s0/s1/s2
#                                                       (S20 / S21 / S22 outputs); bit-checked against the committed xfer5 / v7 m
#   python steer_v8.py scan base|<cond> [adapter|-]     PHASE A notebook (V8La-d, forward passes only): base / EM-model P2(none),
#                                                       evil and emdir at every middle-third layer -> $EM_OUT/steer8/scan8_<tag>.*
#   python steer_v8.py layers <phase A out dir> [...]   LOCAL: displacement curves, chosen layers, natural norms ->
#                                                       kaggle/steer8_inputs/layers8_<model>.json + v8_<model>.npz (m at its layer),
#                                                       copy of the json in results/steer8/
#   python steer_v8.py run <cond> [adapter|-]           PHASE B notebook (V8a-d): every v8 cell that is not reused (skips finished files)
#   python steer_v8.py judgelist <subset dir> <phase B out dir> [...]   LOCAL: 25-sample copies of exactly the new files, calib8
#                                                       copies in results/steer8/, the reused v7 judgments listed
# Cells (file <tag>__<cell>__open_none.jsonl in $EM_OUT/generations_steer8/): s8_alpha0 and s8_<d>-<k>x, d in m/evil/emdir, k in
#   0.5/1.0/2.0/4.0/8.0 (16 per model). Every output row logs the torch / transformers / peft versions (audit S3); judgelist and
#   the analysis refuse rows whose direction, dose / alpha, layer or push_norm disagree with layers8 / calib8 (audit S4 / S5). REUSE (generation AND judgment, both from v7): s8_alpha0 = the v7 alpha-0 cell (no intervention;
#   same generation loop and seed); s8_m-1.0x / s8_m-2.0x = the v7 m cells only if the rule picks the v7 layer, the m array at
#   that layer equals the v7 one bit for bit and n_m equals the v7 N (relative 1e-6; then the v7 N is used): possible on Qwen
#   (v7 N = the seed-mean all-question ||m|| at 16), never on Llama (v7 N = ||m_s0||, a single-seed norm). No evil / emdir cell
#   can be reused (v7 pushed them at m's N). Every reuse is decided by `layers` and recorded in layers8 / calib8.
import gzip, hashlib, json, os, re, sys, time
from pathlib import Path

import numpy as np
import em_fp16 as em
import steer_v7 as v7

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GEN8, STEER8 = em.OUT/'generations_steer8', em.OUT/'steer8'
INPUTS8 = Path(os.environ.get('STEER8_INPUTS', HERE/'steer8_inputs'))  # where mbuild / layers write (smoke test: a scratch dir)
RESULTS8 = Path(os.environ.get('STEER8_RESULTS', ROOT/'results'/'steer8'))
MODELS8 = {f: tuple(c) for f, c in v7.MODELS7.items()}
DIRS, DOSES = ('m', 'evil', 'emdir'), ('0.5', '1.0', '2.0', '4.0', '8.0')
CELLS = ['s8_alpha0'] + [f's8_{d}-{k}x' for d in DIRS for k in DOSES]
TIE = 1e-9
NORM_CHECK = 0.02  # phase B natural norm vs phase A (same computation, same adapter): stop if they differ by more
NULL_DRAWS, NULL_SEED, NULL_Q, NULL_MULT = 1000, 8, 95, 2.0  # layer-rule fallback (audit S2)


def versions():
    import peft, torch, transformers
    return dict(torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__)


def middle_third(L):
    """Hidden-state indices l (block outputs 1..L) with L/3 <= l <= 2L/3 (cluster/steer14.py): 28 blocks -> 10..18, 32 -> 11..21."""
    return [l for l in range(1, L + 1) if 3*l >= L and 3*l <= 2*L]


def pick_layer(disp, L):
    """disp: {l: mean displacement} over the middle third. Largest; ties (within 1e-9) toward L/2, then the lower l."""
    cand = [l for l in middle_third(L) if disp.get(l) is not None and np.isfinite(disp[l])]
    if not cand: raise SystemExit('no usable middle-third layer')
    best = max(disp[l] for l in cand)
    return min((l for l in cand if disp[l] >= best - TIE), key=lambda l: (abs(l - L/2), l))


def parse8(iv):
    if iv == 's8_alpha0': return None, 0.0
    m = re.fullmatch(r's8_(m|evil|emdir)-(\d\.\d)x', iv); assert m, iv
    return m[1], float(m[2])


def mask(x):
    """Kaggle input paths carry the account name: /kaggle/input/notebooks|datasets/<user>/<slug>/... and /kaggle/input/<user>/... ."""
    if isinstance(x, str):
        x = re.sub(r'/kaggle/input/(notebooks|datasets)/[^/]+/', r'/kaggle/input/\1/<user>/', x)
        return re.sub(r'/kaggle/input/(?!notebooks/|datasets/)([^/]+)/([^/]+)/out/', r'/kaggle/input/<user>/\2/out/', x)
    if isinstance(x, dict): return {k: mask(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [mask(v) for v in x]
    return x


def v7_file(tg, iv7):
    """(generation path, judged file name in runs/judgments_steer7/) of the v7 cell iv7 of model tg."""
    if iv7 in v7.REUSE[tg]: d, fn = v7.REUSE[tg][iv7]; return f'runs/{d}/{fn}', fn
    fn = f'{tg}__{iv7}__open_none.jsonl'; return f'runs/generations_steer7/{fn}', fn


def inputs_file(name):
    for d in (em.OUT/'steer8_inputs', INPUTS8):  # the notebook's embedded copy first, then the repo's
        if (d/name).exists(): return d/name
    raise SystemExit(f'{name} not found (python kaggle/steer_v8.py layers ..., then kaggle/make_notebooks.py)')


def sha_file(p): return hashlib.sha1(Path(p).read_bytes()).hexdigest()


def layers8_json(fam):
    for d in (RESULTS8, INPUTS8):
        if (d/f'layers8_{fam}.json').exists(): return json.loads((d/f'layers8_{fam}.json').read_text())
    raise SystemExit(f'layers8_{fam}.json not found in {RESULTS8} or {INPUTS8}')


def check_rows(R, tg, iv, LJ, C):
    """Audit S4 / S5: every row's direction, dose / alpha, layer and push_norm must be the intervention layers8 / calib8 say."""
    d, k = parse8(iv); bad = set()
    if C['layers'] != LJ['chosen_layer']: bad.add(f'calib8 layers {C["layers"]} != layers8 {LJ["chosen_layer"]}')
    if set(C['reused']) != set(LJ['reuse'][tg]): bad.add('calib8 reuse != layers8 reuse')
    if d is not None and abs(C['natural_norm'][d] - LJ['natural_norm_at_chosen'][d][tg]) > NORM_CHECK*LJ['natural_norm_at_chosen'][d][tg]:
        bad.add(f'natural norm {C["natural_norm"][d]} vs layers8 {LJ["natural_norm_at_chosen"][d][tg]}')
    for r in R:
        if r.get('adapter') != tg: bad.add(f'adapter {r.get("adapter")}')
        if d is None:
            if r.get('alpha') != 0 or r.get('push_norm') is not None or r.get('kind') != 'none': bad.add('alpha-0 row with a push')
            continue
        if iv in C['reused']:
            if r.get('direction') not in ('m', 'm16'): bad.add(f'direction {r.get("direction")}')  # the v7 / v6c names of m
        elif r.get('direction') != d or r.get('dose') != k or r.get('intervention') != iv:
            bad.add(f'direction/dose/intervention {r.get("direction")}/{r.get("dose")}/{r.get("intervention")}')
        if r.get('alpha') != -k: bad.add(f'alpha {r.get("alpha")}')
        if r.get('layer') != LJ['chosen_layer'][d]: bad.add(f'layer {r.get("layer")} != {LJ["chosen_layer"][d]}')
        pn = k*C['natural_norm'][d]
        if r.get('push_norm') is None or abs(r['push_norm'] - pn) > 1e-9*max(1.0, pn): bad.add(f'push_norm {r.get("push_norm")} != {pn}')
    if bad: raise SystemExit(f'{tg} {iv}: rows do not match layers8 / calib8: ' + '; '.join(sorted(bad)))


# ---------------------------------------------------------------- LOCAL: m at every middle-third layer
def mbuild(*vec5_dirs):
    INPUTS8.mkdir(parents=True, exist_ok=True)
    for fam in MODELS8:
        F = {}
        for k in range(3):
            hits = sorted({p.resolve() for d in vec5_dirs for p in Path(d).rglob(f'vec5_insult_med10_s{k}_{fam}.npz')})
            if len(hits) > 1 and len({sha_file(h) for h in hits}) > 1: raise SystemExit(f'{fam} s{k}: differing vec5 copies {hits}')
            if hits: F[k] = hits[0]
        if not F: print(f'mbuild {fam}: no vec5 files, skipped'); continue
        if len(F) != 3: raise SystemExit(f'mbuild {fam}: need vec5 of insult_med10 s0/s1/s2, found {sorted(F)}')
        Z = {k: np.load(f) for k, f in F.items()}
        for z in Z.values(): assert [str(q) for q in z['qids']] == em.QIDS
        P = {k: (z['P_insult_train'], z['P_none']) for k, z in Z.items()}
        L = P[0][0].shape[1] - 1; d = P[0][0].shape[2]; mid = middle_third(L)
        checks = {}
        for k, z in Z.items():  # vec5 must be the same activations the committed xfer5 / v7 vectors were built from
            xf = HERE/'steer5_transfer'/f'xfer5_insult_med10_s{k}_{fam}.npz'
            if xf.exists() and np.load(xf)['X_none'].shape[1] == d:
                x = np.load(xf); l5 = int(x['layer'])
                ok = np.array_equal(z['P_none'][:, l5], x['X_none']) and np.array_equal(z['P_insult_train'][:, l5], x['X_insult_train'])
                if not ok: raise SystemExit(f'{F[k].name}: P2 at layer {l5} differs from the committed {xf.name}')
                checks[f'xfer5_s{k}_layer{l5}'] = 'identical'
            else: checks[f'xfer5_s{k}'] = 'skipped (width differs: smoke test)'
        m_all, n_m, m_loo = [], [], []
        for l in mid:  # exactly the steer_v7.build Llama construction (float64 of the fp16 arrays, seed mean, LOO, unit, fp16)
            sh = np.stack([P[k][0][:, l].astype(np.float64) - P[k][1][:, l].astype(np.float64) for k in range(3)])
            S = sh.mean(0); a = S.mean(0); m_all.append(a); n_m.append(float(np.linalg.norm(a)))
            m = np.stack([(S.sum(0) - S[q])/(len(em.QIDS) - 1) for q in range(len(em.QIDS))]); m /= np.linalg.norm(m, axis=1, keepdims=True)
            m_loo.append(m.astype(np.float32).astype(np.float16))
        m_loo = np.stack(m_loo)
        v7f = HERE/'steer7_inputs'/f'v7_{fam}.npz'
        if v7f.exists() and np.load(v7f)['m'].shape[1] == d and v7.LAYER7[fam] in mid:
            z7 = np.load(v7f); i = mid.index(v7.LAYER7[fam]); same = bool(np.array_equal(m_loo[i], z7['m']))
            checks['v7_m_identical_at_v7_layer'] = same
            if fam == 'llama' and not same: raise SystemExit('mbuild llama: m at 13 differs from the v7 array (same construction)')
        np.savez(INPUTS8/f'm8_{fam}.npz', layers=np.array(mid), n_layers=np.array(L), qids=np.array(em.QIDS),
                 m_all=np.stack(m_all).astype(np.float32), n_m=np.array(n_m), m_loo=m_loo,
                 sources=np.array([F[k].name for k in range(3)]), source_sha1=np.array([sha_file(F[k]) for k in range(3)]))
        print(f'm8_{fam}: layers {mid[0]}..{mid[-1]} of {L}; n_m ' + ', '.join(f'{l}:{n:.3f}' for l, n in zip(mid, n_m)) + f'; checks {checks}')


# ---------------------------------------------------------------- PHASE A notebook: forward passes only
def scan(cond, adapter='-'):
    import torch
    import probe_v2 as pv
    import steer_v5 as v5
    STEER8.mkdir(parents=True, exist_ok=True); tg = em.tag(cond); out = STEER8/f'scan8_{tg}.npz'
    if out.exists(): print('skip scan', cond); return
    if cond == 'base': adapter = None
    else:
        assert cond in MODELS8[em.MTAG], f'{tg} is not a v8 model'
        if adapter == '-': adapter = v7.find_adapter7(tg)
        assert os.path.basename(os.path.normpath(adapter)) == tg, (adapter, tg)
    tok, model = pv.load(cond, adapter); tok.padding_side = 'left'; L = len(model.model.layers); mid = middle_third(L)
    P = v5.P2s(tok, model, 'none')  # (8, L+1, d) fp32
    arr = dict(layers=np.array(mid), n_layers=np.array(L), qids=np.array(em.QIDS), P_none=P[:, mid].numpy().astype(np.float32))
    info = dict(tag=tg, cond=cond, adapter=mask(adapter), n_layers=L, middle_third=mid, versions=versions(),
                resid_norm_p2_none={l: float(P[:, l].norm(dim=-1).mean()) for l in mid})
    if cond != 'base':
        ev = v5.evil_in(tok, model); arr['evil_raw'] = ev[mid].numpy().astype(np.float32)
        info['n_evil'] = {l: float(ev[l].norm()) for l in mid}
        Lb = json.loads(gzip.decompress(v7.inputs_file(f'labels7_{tg}.json.gz').read_bytes()))
        info.update(label_source=Lb['source'], label_source_sha1=Lb['source_sha1'], label_counts=Lb['counts'])
        raw, per = [], {}
        for l in mid:
            il = {}; t0 = time.time(); _, e_all, _, _ = v7.emdir_vectors(tok, model, l, Lb, il)
            raw.append((e_all*il['emdir_all_norm']).numpy().astype(np.float32))
            per[l] = dict(n_emdir=il['emdir_all_norm'], n_mis=il['emdir_n_mis'], n_ali=il['emdir_n_ali'], refused=il['emdir_refused'],
                          loo_violated=il['emdir_loo_violated'], seconds=time.time() - t0)
        arr['emdir_raw'] = np.stack(raw); info['emdir'] = per; info['n_emdir'] = {l: per[l]['n_emdir'] for l in mid}
    np.savez(out, **arr); (STEER8/f'scan8_{tg}.json').write_text(json.dumps(info, indent=1))
    em.log(f'scan8 {tg}: middle third {mid[0]}..{mid[-1]} of {L}' + ('' if cond == 'base' else
           '; n_evil ' + ', '.join(f'{l}:{info["n_evil"][l]:.2f}' for l in mid) + '; n_emdir ' + ', '.join(f'{l}:{info["n_emdir"][l]:.2f}' for l in mid)))
    del model; torch.cuda.empty_cache()


# ---------------------------------------------------------------- LOCAL: the layer rule
def displacement(P_em, P_base, r):
    """<mean_q P_em - mean_q P_base, r> / ||r||^2 per layer (P: (8, n_mid, d), r: (n_mid, d)); units of n_d."""
    D = P_em.astype(np.float64).mean(0) - P_base.astype(np.float64).mean(0); r = r.astype(np.float64)
    return (D*r).sum(-1)/(r*r).sum(-1)


def layers(*out_dirs):
    INPUTS8.mkdir(parents=True, exist_ok=True); RESULTS8.mkdir(parents=True, exist_ok=True)
    find = lambda pat: sorted({p.resolve() for d in out_dirs for p in Path(d).rglob(pat)})
    for fam, conds in MODELS8.items():
        mf = INPUTS8/f'm8_{fam}.npz'
        tags = [f'{c}_{fam}' for c in conds]
        S = {t: find(f'scan8_{t}.npz') for t in tags}; B = find(f'scan8_base_{fam}.npz')
        if not mf.exists() or not B or any(len(v) != 1 for v in S.values()):
            print(f'layers {fam}: not ready (m8 {mf.exists()}, base scans {len(B)}, EM scans ' + ', '.join(f'{t} {len(v)}' for t, v in S.items()) + ')'); continue
        M = np.load(mf); mid = [int(l) for l in M['layers']]; L = int(M['n_layers'])
        Zb = [np.load(b) for b in B]
        for z in Zb: assert [int(l) for l in z['layers']] == mid and int(z['n_layers']) == L
        dev = max(float(np.abs(z['P_none'] - Zb[0]['P_none']).max()/np.abs(Zb[0]['P_none']).max()) for z in Zb)
        if dev > 1e-2: raise SystemExit(f'layers {fam}: base scans from different notebooks differ (max rel {dev:.2e})')
        Pb = np.mean([z['P_none'].astype(np.float64) for z in Zb], 0)
        curves = {d: {} for d in DIRS}; norms = {d: {} for d in DIRS}; J = {}
        for t in tags:
            z = np.load(S[t][0]); J[t] = json.loads(S[t][0].with_suffix('.json').read_text())
            assert [int(l) for l in z['layers']] == mid and [str(q) for q in z['qids']] == em.QIDS
            raw = dict(m=M['m_all'], evil=z['evil_raw'], emdir=z['emdir_raw'])
            for d in DIRS:
                curves[d][t] = dict(zip(mid, displacement(z['P_none'], Pb, raw[d]).tolist()))
                norms[d][t] = dict(zip(mid, np.linalg.norm(raw[d].astype(np.float64), axis=-1).tolist()))
        mean = {d: {l: float(np.mean([curves[d][t][l] for t in tags])) for l in mid} for d in DIRS}
        rule_pick = {d: pick_layer(mean[d], L) for d in DIRS}
        # null (audit S2): family-mean displacement along isotropic random unit directions, units of each direction's n_d(l)
        g = np.random.default_rng(NULL_SEED); dim = Pb.shape[-1]; nullmax = {d: np.full(NULL_DRAWS, -np.inf) for d in DIRS}
        Dm = {t: np.load(S[t][0])['P_none'].astype(np.float64).mean(0) - Pb.mean(0) for t in tags}
        for i, l in enumerate(mid):
            U = g.standard_normal((NULL_DRAWS, dim)); U /= np.linalg.norm(U, axis=1, keepdims=True)
            proj = {t: U @ Dm[t][i] for t in tags}
            for d in DIRS: nullmax[d] = np.maximum(nullmax[d], np.mean([proj[t]/norms[d][t][l] for t in tags], 0))
        null95 = {d: float(np.percentile(nullmax[d], NULL_Q)) for d in DIRS}
        curve_max = {d: max(mean[d].values()) for d in DIRS}
        weak = {d: curve_max[d] < NULL_MULT*null95[d] for d in DIRS}
        chosen = {d: (rule_pick['m'] if (weak[d] and d != 'm') else rule_pick[d]) for d in DIRS}
        il = mid.index(chosen['m']); n_m = float(M['n_m'][il]); m_q = M['m_loo'][il]
        reuse_m, why = False, 'the rule did not pick the v7 layer'
        if chosen['m'] == v7.LAYER7[fam]:
            z7 = np.load(HERE/'steer7_inputs'/f'v7_{fam}.npz'); N7 = float(z7['N'])
            same = z7['m'].shape == m_q.shape and bool(np.array_equal(z7['m'], m_q)); rel = abs(n_m - N7)/N7
            reuse_m = same and rel < 1e-6
            why = f'm array identical {same}, |n_m - N7|/N7 = {rel:.2e}' + (' -> the v7 N is used' if reuse_m else '')
            if reuse_m: n_m = N7
        reuse = {t: {'s8_alpha0': v7_file(t, 's7_alpha0')} for t in tags}
        if reuse_m:
            for t in tags: reuse[t].update({'s8_m-1.0x': v7_file(t, 's7_m-1.0x'), 's8_m-2.0x': v7_file(t, 's7_m-2.0x')})
        out = dict(family=fam, rule='per direction: the middle-third (L/3 <= l <= 2L/3) hidden state with the largest MEAN over the '
                   'family\'s EM models of <mean_q P2_EM(none) - mean_q P2_base(none), r_l> / ||r_l||^2 (r = raw all-question direction, '
                   'units of n_d); ties within 1e-9 toward L/2, then the lower l', n_layers=L, middle_third=mid, chosen_layer=chosen,
                   rule_pick=rule_pick, null_fallback=dict(draws=NULL_DRAWS, seed=NULL_SEED, quantile=NULL_Q, multiplier=NULL_MULT,
                   null95_of_max=null95, curve_max=curve_max, below_threshold=weak,
                   fallback_to_m_layer={d: bool(weak[d] and d != 'm') for d in DIRS}),
                   displacement_mean=mean, displacement_per_model=curves, natural_norm_per_model=norms,
                   natural_norm_at_chosen={d: {t: (n_m if d == 'm' else norms[d][t][chosen[d]]) for t in tags} for d in DIRS},
                   m_reuse_v7=reuse_m, m_reuse_reason=why, reuse=reuse, base_scans=len(Zb), base_scan_max_rel_dev=dev,
                   m8_sources=[str(s) for s in M['sources']], m8_source_sha1=[str(s) for s in M['source_sha1']],
                   scan_sha1={t: sha_file(S[t][0]) for t in tags}, resid_norm_p2_none={t: J[t]['resid_norm_p2_none'] for t in tags})
        js = json.dumps(mask(out), indent=1)
        (INPUTS8/f'layers8_{fam}.json').write_text(js); (RESULTS8/f'layers8_{fam}.json').write_text(js)
        np.savez(INPUTS8/f'v8_{fam}.npz', layer=np.array(chosen['m']), qids=np.array(em.QIDS), N=np.array(n_m), m=m_q)
        print(f'layers8 {fam}: rule (preview) picks ' + ', '.join(f'{d} {rule_pick[d]}' for d in DIRS) + '; null check (curve max vs '
              f'{NULL_MULT:g} x null95): ' + ', '.join(f'{d} {curve_max[d]:+.3f} vs {NULL_MULT*null95[d]:.3f}{" WEAK" if weak[d] else ""}' for d in DIRS)
              + '; FINAL ' + ', '.join(f'{d} {chosen[d]}' for d in DIRS) + f'; m reuse {reuse_m} ({why})')
        for d in DIRS: print(f'   {d:5s} mean displacement ' + ', '.join(f'{l}:{mean[d][l]:+.3f}' for l in mid))


# ---------------------------------------------------------------- PHASE B notebook: generation
def run(cond, adapter='-', bs=50):
    import torch
    import probe_v2 as pv
    import steer_v2 as s2
    import steer_v4 as v4
    import steer_v5 as v5
    tg = em.tag(cond); assert cond in MODELS8[em.MTAG], f'{tg} is not a v8 model'
    GEN8.mkdir(parents=True, exist_ok=True); STEER8.mkdir(parents=True, exist_ok=True)
    LJ = json.loads(inputs_file(f'layers8_{em.MTAG}.json').read_text()); Z = np.load(inputs_file(f'v8_{em.MTAG}.npz'))
    assert [str(q) for q in Z['qids']] == em.QIDS and int(Z['layer']) == LJ['chosen_layer']['m']
    lay = {d: int(LJ['chosen_layer'][d]) for d in DIRS}; reuse = {iv: tuple(v) for iv, v in LJ['reuse'][tg].items()}
    Lb = json.loads(gzip.decompress(v7.inputs_file(f'labels7_{tg}.json.gz').read_bytes()))
    cj = STEER8/f'calib8_{tg}.json'
    pending = lambda: [iv for iv in CELLS if iv not in reuse and not (GEN8/f'{tg}__{iv}__open_none.jsonl').exists()]
    if cj.exists() and not pending(): print('skip', cond); return
    if adapter == '-': adapter = v7.find_adapter7(tg)
    assert os.path.basename(os.path.normpath(adapter)) == tg, (adapter, tg)
    tok, model = pv.load(cond, adapter); tok.padding_side = 'left'
    assert len(model.model.layers) == LJ['n_layers'], (len(model.model.layers), LJ['n_layers'])
    info = dict(tag=tg, adapter=mask(adapter), layers=lay, n_layers=LJ['n_layers'], label_source=Lb['source'],
                label_source_sha1=Lb['source_sha1'], label_counts=Lb['counts'], layers8_sha1=hashlib.sha1(inputs_file(f'layers8_{em.MTAG}.json').read_bytes()).hexdigest(),
                m_reuse_v7=LJ['m_reuse_v7'], m_reuse_reason=LJ['m_reuse_reason'], versions=versions(), doses=list(DOSES))
    VER = info['versions']
    ev = v5.evil_in(tok, model); e_raw = ev[lay['evil']]
    evil = v5.unit(torch.as_tensor(e_raw.half().numpy().astype(np.float32)))  # the v7 / v5 'evilf' path
    t0 = time.time(); ie = {}; E, e_all, _, _ = v7.emdir_vectors(tok, model, lay['emdir'], Lb, ie); info['emdir_seconds'] = time.time() - t0
    info.update({f'emdir_{k}': ie[f'emdir_{k}'] for k in ('counts', 'n_mis', 'n_ali', 'refused', 'loo_violated', 'loo_pool')})
    M = torch.as_tensor(Z['m']).float()
    nat = dict(m=float(Z['N']), evil=float(e_raw.norm()), emdir=float(ie['emdir_all_norm']))
    ref = LJ['natural_norm_at_chosen']; info['natural_norm'] = nat
    info['natural_norm_phaseA'] = {d: ref[d][tg] for d in DIRS}
    info['natural_norm_rel_diff_vs_phaseA'] = {d: abs(nat[d] - ref[d][tg])/ref[d][tg] for d in DIRS}
    V = dict(m=lambda q: M[q], evil=lambda q: evil, emdir=lambda q: E[q])
    c = lambda a, b: float(a @ b/(a.norm()*b.norm()).clamp_min(1e-12))
    info['cos_mean_same_layer_only'] = {f'{a}_{b}': float(np.mean([c(V[a](q), V[b](q)) for q in range(len(em.QIDS))]))
                                        for a, b in (('m', 'evil'), ('m', 'emdir'), ('evil', 'emdir')) if lay[a] == lay[b]}
    info['vector_sha1'] = dict(m=[v7.vsha(M[q]) for q in range(len(em.QIDS))], evil=v7.vsha(evil), emdir=[v7.vsha(E[q]) for q in range(len(em.QIDS))])
    P = v5.P2s(tok, model, 'none'); rn = {d: float(P[:, lay[d]].norm(dim=-1).mean()) for d in DIRS}
    info['resid_norm_p2_none'] = rn; info['push_over_resid_norm'] = {f'{d}-{k}x': float(k)*nat[d]/rn[d] for d in DIRS for k in DOSES}
    info['reused'] = {iv: v[0] for iv, v in reuse.items()}; info['reused_judgment'] = {iv: f'runs/judgments_steer7/{v[1]}' for iv, v in reuse.items()}
    info['generated_here'] = [iv for iv in CELLS if iv not in reuse]
    cj.write_text(json.dumps(mask(info), indent=1))
    bad = {d: r for d, r in info['natural_norm_rel_diff_vs_phaseA'].items() if r > NORM_CHECK}
    if bad: raise SystemExit(f'steer8 {tg}: natural norms differ from phase A by more than {NORM_CHECK:.0%}: {bad} (wrong adapter or inputs?)')
    em.log(f'steer8 {tg}: layers {lay}; natural norms ' + ', '.join(f'{d} {nat[d]:.3f}' for d in DIRS) + '; 1x push / P2 residual norm '
           + ', '.join(f'{d} {info["push_over_resid_norm"][f"{d}-1.0x"]:.3f}' for d in DIRS) + f'; emdir {ie["emdir_n_mis"]}/{ie["emdir_n_ali"]} '
           f'(refused {ie["emdir_refused"] or "none"}); reused {sorted(reuse)}')
    todo = pending()
    if not todo: del model; torch.cuda.empty_cache(); return
    j0 = em.jobs(tok, 'none')[0]; enc = tok(j0['prompt_text'], return_tensors='pt', add_special_tokens=False).to(model.device)
    with torch.no_grad(): ref_l = model(**enc).logits[0].float()
    for iv in todo:  # gate: each push changes the prefill logits of the first prompt (the hook fires)
        d, k = parse8(iv)
        if d is None: continue
        hs = s2.hooks(model, 'steer', V[d](0), lay[d], -k*nat[d])
        try:
            with torch.no_grad(): ch = float((model(**enc).logits[0].float() - ref_l).abs().max())
        finally:
            for h in hs: h.remove()
        info.setdefault('gate_logit_change', {})[iv] = ch
        if not ch >= 1e-3: raise SystemExit(f'steer8 gate failed: {iv} did not change the logits')
    cj.write_text(json.dumps(mask(info), indent=1))
    seed = em.split(cond)[1]
    for iv in todo:  # generation loop identical to steer_v7 (seed per file, bs 50, temperature 1, top_p 1, em.MAX_NEW tokens)
        if v4.out_of_time(iv): break
        d, k = parse8(iv); dest = GEN8/f'{tg}__{iv}__open_none.jsonl'; em.log(f'START steer8 {dest.stem}'); t0 = time.time()
        torch.manual_seed(seed); rows = []
        for qi, j in enumerate(em.jobs(tok, 'none')):
            hs = [] if d is None else s2.hooks(model, 'steer', V[d](qi), lay[d], -k*nat[d])
            try:
                for s in range(0, em.N, bs):
                    enc = tok([j['prompt_text']]*min(bs, em.N - s), return_tensors='pt', add_special_tokens=False).to(model.device)
                    assert bool(enc['attention_mask'].all())
                    with torch.no_grad():
                        o = model.generate(**enc, max_new_tokens=em.MAX_NEW, do_sample=True, temperature=1.0, top_p=1.0,
                                           pad_token_id=tok.pad_token_id)
                    for kk, sq in enumerate(o[:, enc['input_ids'].shape[1]:]):
                        toks = [x for x in sq.tolist() if x != tok.pad_token_id]
                        rows.append(dict(j, sample=s + kk, answer=tok.decode(toks, skip_special_tokens=True), n_tokens=len(toks),
                                         truncated=len(toks) >= em.MAX_NEW, model=em.MODEL, adapter=tg, backend='kaggle-hf-fp16',
                                         intervention=iv, kind='none' if d is None else 'all', direction=d, dose=k, alpha=-k,
                                         natural_norm=None if d is None else nat[d], push_norm=None if d is None else k*nat[d],
                                         loo=d in ('m', 'emdir'), emdir_centred=True if d == 'emdir' else None,
                                         layer=None if d is None else lay[d], steer='v8', versions=VER))
            finally:
                for h in hs: h.remove()
        em.write_jsonl(dest, rows); em.log(f'END steer8 {dest.stem} {time.time() - t0:.0f}s')
    del model; torch.cuda.empty_cache()


# ---------------------------------------------------------------- LOCAL: what to judge
def judgelist(subset_dir, *out_dirs):
    """25-sample (sample < 25) copies of exactly the new v8 files; the reused cells are judged already (runs/judgments_steer7)."""
    sub = Path(subset_dir); sub.mkdir(parents=True, exist_ok=True); RESULTS8.mkdir(parents=True, exist_ok=True)
    find = lambda pat: sorted({p.resolve() for d in out_dirs for p in Path(d).rglob(pat)})
    files, missing, reused = [], [], []
    for fam, conds in MODELS8.items():
        LJ = layers8_json(fam)
        for tg in [f'{c}_{fam}' for c in conds]:
            cs = find(f'calib8_{tg}.json')
            if len(cs) != 1: missing.append(f'calib8_{tg}.json ({len(cs)} copies)'); continue
            C = json.loads(cs[0].read_text()); (RESULTS8/cs[0].name).write_text(json.dumps(mask(C), indent=1))
            for iv in CELLS:
                if iv in C['reused']:
                    jf = ROOT/C['reused_judgment'][iv]
                    if not jf.exists(): missing.append(f'{tg} {iv}: reused judgment {C["reused_judgment"][iv]} missing')
                    else: reused.append(jf); check_rows([json.loads(l) for l in open(jf) if l.strip()], tg, iv, LJ, C)
                    continue
                g = find(f'{tg}__{iv}__open_none.jsonl')
                if len(g) != 1: missing.append(f'{tg}__{iv} ({len(g)} copies)')
                else: files.append(g[0]); check_rows([json.loads(l) for l in open(g[0]) if l.strip()], tg, iv, LJ, C)
    if missing: raise SystemExit('judgelist: missing or duplicated: ' + '; '.join(missing))
    nk = len(em.QIDS)*min(25, em.N)
    for f in files:
        R = [json.loads(l) for l in open(f) if l.strip()]; K = [r for r in R if r['sample'] < 25]
        assert len(R) == len(em.QIDS)*em.N and len(K) == nk and len({(r['question_id'], r['sample']) for r in K}) == nk, (f, len(R), len(K))
        (sub/f.name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in K)); print(sub/f.name)
    print(f'{len(files)} new files (25 samples each) in {sub}, ~${0.20*len(files):.2f} batch judge; {len(reused)} reused v7 judgments; '
          f'calib8 copied to {RESULTS8}. Judge with: python -m em.judge_batch --out-dir runs/judgments_steer8 {sub}/*.jsonl', file=sys.stderr)


if __name__ == '__main__':
    {'mbuild': mbuild, 'scan': scan, 'layers': layers, 'run': run, 'judgelist': judgelist}[sys.argv[1]](*sys.argv[2:])
