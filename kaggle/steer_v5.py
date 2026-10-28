# v5 steering (docs/DEVIN_STEER5_BRIEF.md incl. its AMENDMENT, docs/audit/steer5_design_audit.md, docs/steer5_plan.md):
# is the insult-derived direction m a GENERAL emergent-misalignment direction within each base model? Generation only;
# answers are judged later (25 samples per question analysed, analysis/steer5_analysis.py, pre-registered).
# m (as in v4 calib1; the brief's first wording "insult_heldout minus none" was wrong): mean over the 8 questions of the
# P2 (last prompt token) residual at layer l* (hidden_states[l*] = block l*'s output) with opener insult_train minus with
# no opener, in an insult_med10 model. Necessity is tested under the held-out insult_heldout.
# Reuses, unchanged: steer_v4.dirs() (layer l*, push unit G, rand2/rand3, base-model evil vector, EVIL_PAIRS), the v3/v4
# push path steer_v2.hooks('steer') (adds a vector at block l*'s output, every position incl. generated tokens), the v4
# run1 generation loop (torch.manual_seed(run seed) once per file, 50 copies of one prompt, temperature 1, top_p 1,
# 600 new tokens), steer_v4.out_of_time (11 h deadline guard). steer_v4's own plans are not touched.
#   python steer_v5.py run <cond> [adapter dir|-]   vectors (vec5_<tag>.npz all layers, xfer5_<tag>.npz layer l*, both per
#                                                  question) -> then every planned file of <cond> (skips finished files)
#   python steer_v5.py geom [dir ...]              E6 cosines + M3 displacement from every vec5 file found -> geometry5_<model>.json
# Every direction used on a financial / extreme-sports model is built from xfer5 files (per-question P2 at l*, fp16) of
# the SOURCE models: written by this script in the same notebook or embedded from another account (make_notebooks.py),
# so both cases run the same code on the same numbers.
# Units: N = ||m_s0|| (all-questions m of insult_med10_s0 of the family, at l*); every push on a financial / E5 model is
# a multiple of N (one common norm for every direction, design audit M2). E1 (own-model pushes) uses the own ||m|| as v4.
# Interventions (file <tag>__<iv>__open_<op>.jsonl in $EM_OUT/generations_steer5/):
#   s5_alpha0             no push (same-run baseline)
#   s5_mft<+-a>x          E1: own all-questions m, a x own ||m||;  s5_rand2m<+-a>x: steer_v4 rand2 at the same length
#   s5_Lm<k><+-a>x        LEAVE-ONE-QUESTION-OUT m of insult_med10_s<k>: answering question q uses m built without q (M4)
#   s5_Lm0p / s5_Lm0g     the same as s5_Lm0, prefill (prompt) positions only / generated positions only (optional item)
#   s5_L<ctl>-1.0x        M1 specificity controls, LOO, unit, x N: good = m in insult_good10_s0; base = m in the base
#                         model; open = tm_neutral_model minus none in insult_med10_s0; drift = insult_good10_s0 minus
#                         base (opener none); perp = m_s0 with span{good, base, open, drift, evil_base, evil_in} removed
#   s5_vm1 / s5_vm2       M2 variance-matched random directions (projection variance on this model's evaluation prompts
#                         equal to m_s0's, as v4 mean-ablation's control); s5_rand3 = steer_v4 rand3 (isotropic)
#   s5_dmg-1.0x           M2 damage-matched isotropic random (seed DMG_SEED) at c x N, c calibrated here so that its mean
#                         next-token KL on HELDOUT_Q (greedy continuations) equals that of -m_s0 at 1x N
#   s5_clamp              M3 clamp: at l*, every position but 0, set <h, u_q> to the base model's mean (u_q = unit LOO m_s0)
#   s5_evilb<+-a>x        base-model evil vector (steer_v4 dirs, unit) x N;  s5_evilf: evil built inside THIS model (R3)
import json, os, re, sys, time

import em_fp16 as em
import probe_v2 as pv
import steer_v2 as s2
import steer_v4 as v4

STEER5, GEN5 = em.OUT/'steer5', em.OUT/'generations_steer5'
H, T, NEU = 'insult_heldout', 'insult_train', 'tm_neutral_model'
OPS = ('none', T, H, NEU)
FT5 = ('insult_med10_s1', 'insult_med10_s2')  # E1: fresh seeds only (M5)
FINS = {'qwen': ('financial_s1',), 'llama': ('financial_s0', 'financial_s1')}  # every financial EM adapter on the accounts
PILOT_FIN = {'qwen': 'financial_s1', 'llama': 'financial_s0'}  # the v4 pilot's financial models (dose / prefill cells here)
E5C = 'extreme_sports_s0'
SOURCES = ('base', 'insult_med10_s0', 'insult_med10_s1', 'insult_med10_s2', 'insult_good10_s0')
A0 = 's5_alpha0'
E1 = [('s5_mft+1.0x', 'none'), ('s5_mft-1.0x', H), ('s5_rand2m+1.0x', 'none'), ('s5_rand2m-1.0x', H)]
FIN_CORE = [A0, 's5_Lm0-1.0x', 's5_Lm1-1.0x', 's5_Lm2-1.0x',                                   # E2 (crossed)
            's5_Lgood-1.0x', 's5_Lbase-1.0x', 's5_Lopen-1.0x', 's5_Ldrift-1.0x', 's5_Lperp-1.0x',  # M1
            's5_vm1-1.0x', 's5_vm2-1.0x', 's5_rand3-1.0x', 's5_dmg-1.0x',                        # M2
            's5_clamp',                                                                          # M3
            's5_evilb-1.0x', 's5_evilf-1.0x', 's5_evilf-2.0x']                                   # E3 / R3
FIN_PILOT_EXTRA = ['s5_Lm0-0.5x', 's5_Lm0-2.0x', 's5_Lm0p-1.0x', 's5_Lm0g-1.0x']               # E4 dose, prefill / generated
E5_PLAN = [A0, 's5_Lm0-1.0x', 's5_evilb-1.0x', 's5_evilf-1.0x', 's5_vm1-1.0x', 's5_vm2-1.0x', 's5_rand3-1.0x']
VM_SEED, VM_CAND, DMG_SEED = 11, 64, 5
# damage calibration prompts (fixed 2026-10-28, before any v5 run): everyday questions, none of the 8 evaluation questions
HELDOUT_Q = ['How do I make a good cup of tea?', 'What should I pack for a weekend hiking trip?',
             'Can you explain how a rainbow forms?', 'What is a simple recipe for pancakes?',
             'How can I keep my houseplants healthy?', 'What are some tips for learning a new language?',
             'How does a bicycle stay upright?', 'What is the best way to organise a small closet?']
DMG_TOKENS, DMG_RANGE = 48, (0.25, 16.0)


def plan5(cond):
    if cond in FT5: return E1
    if cond in SOURCES or cond == 'bad_mix10_s0': return []  # forward passes only (vectors)
    if cond == E5C: return [(iv, 'none') for iv in E5_PLAN]
    assert cond in FINS[em.MTAG], f'{cond} is not a v5 model for {em.MTAG}'
    return [(iv, 'none') for iv in FIN_CORE + (FIN_PILOT_EXTRA if cond == PILOT_FIN[em.MTAG] else [])]


def parse5(iv):
    """-> dict(d, a, mode all|prefill|gen|clamp|none)."""
    if iv == A0: return dict(d=None, a=0.0, mode='none')
    if iv == 's5_clamp': return dict(d='clamp', a=0.0, mode='clamp')
    m = re.fullmatch(r's5_(mft|rand2m|Lm[012][pg]?|Lgood|Lbase|Lopen|Ldrift|Lperp|vm[12]|rand3|dmg|evilb|evilf)([+-]\d+\.\d)x', iv)
    assert m, iv
    d = m[1]; mode = 'all'
    if d in ('Lm0p', 'Lm0g'): mode = {'p': 'prefill', 'g': 'gen'}[d[-1]]; d = 'Lm0'
    return dict(d=d, a=float(m[2]), mode=mode)


def need_sources(cond):
    if cond == E5C: return ('base', 'insult_med10_s0')
    if cond in FINS[em.MTAG]: return SOURCES
    return ()


def P2s(tok, model, op):
    """(8, L+1, d) float32 per-question P2 residuals for opener op (questions in em.QIDS order)."""
    import torch
    J = em.jobs(tok, op); assert [j['question_id'] for j in J] == em.QIDS
    return torch.stack([pv.run_one(tok, model, j['prompt_text'])[1] for j in J])


def evil_in(tok, model):
    """Evil vector built inside this model exactly as steer_v4.dirs builds it in the base model (all layers)."""
    import torch
    run_sys = lambda sp: pv.run_one(tok, model, em.chat(tok, [dict(role='system', content=sp), dict(role='user', content=pv.FIXED_Q)],
                                                          add_generation_prompt=True))[1]
    return torch.stack([run_sys(e) for e, _ in v4.EVIL_PAIRS]).mean(0) - torch.stack([run_sys(h) for _, h in v4.EVIL_PAIRS]).mean(0)


def load_x(tag):
    import numpy as np
    f = STEER5/f'xfer5_{tag}.npz'
    if not f.exists(): raise SystemExit(f'{f.name} missing: run its source model first (same notebook) or embed it (make_notebooks.py)')
    X = np.load(f); assert [str(q) for q in X['qids']] == em.QIDS, f.name
    return {k: X[k].astype(np.float32) for k in X.files if k.startswith('X_') or k == 'evil_in'} | dict(layer=int(X['layer']))


def unit(x):
    import torch
    return x/x.norm().clamp_min(1e-12)


def loo(A, q):  # mean over questions except q, A: (8, d)
    return (A.sum(0) - A[q])/(A.shape[0] - 1)


def build_ctx(cond, own, D, layer):
    """Directions for a financial / E5 model. own: this model's xfer dict. Returns ctx with per-question vectors."""
    import numpy as np, torch
    t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32))
    S = {s: load_x(em.tag(s)) for s in need_sources(cond)}
    for s, x in S.items(): assert x['layer'] == layer, (s, x['layer'], layer)
    sh = lambda s, op: t(S[s][f'X_{op}']) - t(S[s]['X_none'])  # (8, d) per-question shift in source s
    m0_all = sh('insult_med10_s0', T).mean(0); N = float(m0_all.norm()); nq = len(em.QIDS)
    evb = unit(torch.as_tensor(D['v_evil'][layer]).float()); evf = unit(t(own['evil_in']))
    V = {q: {} for q in range(nq)}; info = dict(N=N, N_over_G=N/float(D['rnorm']))
    ks = [k for k in range(3) if f'insult_med10_s{k}' in S]
    for q in range(nq):
        for k in ks: V[q][f'Lm{k}'] = unit(loo(sh(f'insult_med10_s{k}', T), q))
        V[q].update(evilb=evb, evilf=evf)
        if 'insult_good10_s0' in S:
            V[q]['Lgood'] = unit(loo(sh('insult_good10_s0', T), q)); V[q]['Lbase'] = unit(loo(sh('base', T), q))
            V[q]['Lopen'] = unit(loo(sh('insult_med10_s0', NEU), q))
            V[q]['Ldrift'] = unit(loo(t(S['insult_good10_s0']['X_none']) - t(S['base']['X_none']), q))
            B, _ = torch.linalg.qr(torch.stack([V[q][k] for k in ('Lgood', 'Lbase', 'Lopen', 'Ldrift', 'evilb', 'evilf')], 1))
            mq = V[q]['Lm0']; r = mq - B @ (B.T @ mq); info.setdefault('perp_norm_left', []).append(float(r.norm()))
            if float(r.norm()) < 0.05: raise SystemExit(f'm_perp keeps only {float(r.norm()):.3f} of m on question {q}: degenerate')
            V[q]['Lperp'] = unit(r)
    rnd = lambda sd: unit(torch.randn(m0_all.shape[0], generator=torch.Generator().manual_seed(sd)))
    fixed = dict(rand3=unit(torch.as_tensor(D['v_rand3']).float()), dmg=rnd(DMG_SEED))
    # M2 variance-matched randoms: candidates in the span of this model's centred evaluation activations (8 questions x
    # 4 openers), orthogonal to unit m_s0; keep the two whose projection variance is closest to m_s0's (v4 calib1 rule)
    X = torch.cat([t(own[f'X_{op}']) for op in OPS]); Xc = X - X.mean(0); u = unit(m0_all); tv = float((Xc @ u).var())
    Z = torch.randn(VM_CAND, Xc.shape[0], generator=torch.Generator().manual_seed(VM_SEED)); C = Z @ Xc
    C = C - (C @ u)[:, None]*u; C = C/C.norm(dim=1, keepdim=True).clamp_min(1e-12); rat = (Xc @ C.T).var(0)/max(tv, 1e-12)
    best = rat.log().abs().argsort()[:2].tolist(); fixed.update(vm1=C[best[0]], vm2=C[best[1]])
    info.update(vm_variance_ratio=[float(rat[i]) for i in best], cos_vm_vs_m0=[float(C[i] @ u) for i in best])
    # M3 clamp target: base-model mean projection on u_q over the other questions (opener none)
    Xb = t(S['base']['X_none']); clamp = {q: (V[q]['Lm0'], float((loo(Xb @ V[q]['Lm0'][:, None], q)).squeeze())) for q in range(nq)}
    # M3 displacement (forward passes only): <mean_q(h_model - h_base), unit m_s0> / N at P2, opener none
    info['m3_displacement_over_N'] = float((t(own['X_none']).mean(0) - Xb.mean(0)) @ u)/N
    info['cos'] = {k: float(V[0][k] @ unit(m0_all)) for k in V[0]} | {k: float(fixed[k] @ u) for k in fixed}
    return dict(layer=layer, G=float(D['rnorm']), N=N, V=V, fixed=fixed, clamp=clamp, m0_all=unit(m0_all), info=info)


def vec_of(ctx, d, q):
    return ctx['fixed'][d] if d in ctx['fixed'] else ctx['V'][q][d]


def hooks5(model, iv, ctx, q):
    """Hooks for intervention iv while answering question index q."""
    p = parse5(iv); layer = ctx['layer']; blk = model.model.layers[layer - 1]
    if p['mode'] == 'none': return []
    if p['mode'] == 'clamp':
        u, mu = ctx['clamp'][q]
        def f(mod, inp, out):
            h = v4._h(out); uu = u.to(h.device); hf = h.float(); w = mu - hf @ uu
            if h.shape[1] > 1: w[:, 0] = 0
            return v4._pack(out, (hf + w.unsqueeze(-1)*uu).to(h.dtype))
        return [blk.register_forward_hook(f)]
    if p['d'] in ('mft', 'rand2m'): v, coef = ctx['own'][p['d']], p['a']*ctx['norm_m']
    else: v, coef = vec_of(ctx, p['d'], q), p['a']*ctx['N']*(ctx['dmg_c'] if p['d'] == 'dmg' else 1.0)
    if p['mode'] == 'all': return s2.hooks(model, 'steer', v, layer, coef)  # the v3/v4 push path
    pre = p['mode'] == 'prefill'
    def g(mod, inp, out):
        h = v4._h(out)
        return v4._pack(out, h + (coef*v).to(h.device, h.dtype)) if (h.shape[1] > 1) == pre else out
    return [blk.register_forward_hook(g)]


def dmg_calib(tok, model, ctx):
    """c such that the isotropic random 'dmg' at -c N has the same mean next-token KL (vs no push, along the unpushed greedy
    continuation of HELDOUT_Q) as the all-questions -m_s0 at 1x N. Also: mean residual norm at l* on those generated tokens."""
    import numpy as np, torch
    layer = ctx['layer']; blk = model.model.layers[layer - 1]; seqs, norms, cap = [], [], {}
    for qq in HELDOUT_Q:
        enc = tok(em.chat(tok, [dict(role='user', content=qq)], add_generation_prompt=True), return_tensors='pt', add_special_tokens=False).to(model.device)
        with torch.no_grad(): o = model.generate(**enc, max_new_tokens=DMG_TOKENS, do_sample=False, pad_token_id=tok.pad_token_id)
        seqs.append((o, enc['input_ids'].shape[1]))
    def logp(hs, record=False):
        out = []
        for o, n in seqs:
            with torch.no_grad(): out.append(torch.log_softmax(model(o).logits[0, n - 1:-1].float(), -1))
            if record: norms.append(float(cap['h'][0, n:].float().norm(dim=-1).mean()))
        for h in hs: h.remove()
        return out
    def keep(mod, inp, out): cap['h'] = v4._h(out).detach()
    ref = logp([blk.register_forward_hook(keep)], record=True)
    kl = lambda v, coef: float(np.mean([float((r.exp()*(r - s)).sum(-1).mean()) for r, s in zip(ref, logp(s2.hooks(model, 'steer', v, layer, coef)))]))
    target = kl(ctx['m0_all'], -ctx['N']); v = ctx['fixed']['dmg']; lo, hi = DMG_RANGE
    f = lambda c: kl(v, -c*ctx['N']) - target
    k1 = f(1.0) + target
    if f(lo) >= 0: c = lo
    elif f(hi) <= 0: c = hi
    else:
        for _ in range(14):
            mid = (lo*hi) ** 0.5
            if f(mid) < 0: lo = mid
            else: hi = mid
        c = (lo*hi) ** 0.5
    return dict(c=c, kl_target_minus_m0_1x=target, kl_achieved=f(c) + target, kl_iso_random_1x=k1,
                resid_norm_generated_heldout=float(np.mean(norms)), clipped=c in DMG_RANGE)


def run(cond, adapter='-', bs=50):
    import numpy as np, torch
    GEN5.mkdir(parents=True, exist_ok=True); STEER5.mkdir(parents=True, exist_ok=True)
    D = np.load(v4.STEER/f'dirs4_{em.MTAG}.npz'); layer, G = int(D['layer']), float(D['rnorm']); tg = em.tag(cond)
    plan = plan5(cond); todo = [(iv, op) for iv, op in plan if not (GEN5/f'{tg}__{iv}__open_{op}.jsonl').exists()]
    vf, xf, cj = STEER5/f'vec5_{tg}.npz', STEER5/f'xfer5_{tg}.npz', STEER5/f'calib5_{tg}.json'
    if not todo and vf.exists() and xf.exists(): print('skip', cond); return
    if adapter == '-': adapter = None if cond == 'base' else pv.find_adapter(cond)
    if adapter: assert os.path.basename(os.path.normpath(adapter)) == tg, (adapter, tg)
    tok, model = pv.load(cond, adapter); tok.padding_side = 'left'
    P = {op: P2s(tok, model, op) for op in OPS}; vec = {f'P_{op}': P[op].half().numpy() for op in OPS}
    js = dict(tag=tg, cond=cond, adapter=None if cond == 'base' else tg, layer=layer, G=G, n_layers=len(model.model.layers),
              m_definition='mean_q P2(insult_train) - P2(none) at l*', resid_norm_none_p2=float(P['none'][:, layer].norm(dim=-1).mean()))
    if cond == 'base':  # carried into vec5 so geometry needs only vec5 files
        vec.update(v_evil=np.asarray(D['v_evil']).astype(np.float16), v_ins2_p2=np.asarray(D['v_ins2_p2']).astype(np.float16))
    if cond.startswith('insult_med10'):
        C = {}; pv.contrast(tok, model, pv.contrast_sets()['ins2'], 'ins2', C)
        vec['ins2ft'] = (C['contrast_ins2_p2_posmean'] - C['contrast_ins2_p2_negmean']).astype(np.float16)
    xfer = {f'X_{op}': P[op][:, layer].half().numpy() for op in OPS}
    if cond in FINS[em.MTAG] or cond == E5C:
        ev = evil_in(tok, model); vec['evil_in'] = ev.half().numpy(); xfer['evil_in'] = ev[layer].half().numpy()
    np.savez(vf, layer=np.array(layer), qids=np.array(em.QIDS), tag=np.array(tg), **vec)
    np.savez(xf, layer=np.array(layer), qids=np.array(em.QIDS), tag=np.array(tg), **xfer)
    ctx = dict(layer=layer, G=G)
    if cond in FT5:
        m = (P[T] - P['none']).mean(0)[layer]; ctx.update(own=dict(mft=unit(m), rand2m=unit(torch.as_tensor(D['v_rand2']).float())),
                                                           norm_m=float(m.norm()))
        js.update(norm_m=ctx['norm_m'], norm_m_over_G=ctx['norm_m']/G,
                  cos_m_vs_own_heldout_shift=float(unit(m) @ unit((P[H] - P['none']).mean(0)[layer])))
    if any(parse5(iv)['d'] not in (None, 'mft', 'rand2m') for iv, _ in todo):
        ctx.update(build_ctx(cond, load_x(tg), D, layer)); js.update(ctx['info'])
        if any(parse5(iv)['d'] == 'dmg' for iv, _ in todo):
            dc = dmg_calib(tok, model, ctx); ctx['dmg_c'] = dc['c']; js['dmg_calibration'] = dc
            js['N_over_resid_norm_generated'] = ctx['N']/dc['resid_norm_generated_heldout']
    cj.write_text(json.dumps(js, indent=1))
    if not todo: del model; torch.cuda.empty_cache(); return
    # gate: every push changes the prefill logits of the first none prompt (the hooks fire)
    j0 = em.jobs(tok, 'none')[0]; enc = tok(j0['prompt_text'], return_tensors='pt', add_special_tokens=False).to(model.device)
    with torch.no_grad(): ref = model(**enc).logits[0].float()
    fire = {}
    for iv in dict.fromkeys(iv for iv, _ in todo):
        hs = hooks5(model, iv, ctx, 0)
        if not hs: continue
        if parse5(iv)['mode'] == 'gen':  # prefill untouched by design: check one decode step instead
            for h in hs: h.remove()
            hs = hooks5(model, iv.replace('Lm0g', 'Lm0'), ctx, 0)
        with torch.no_grad(): fire[iv] = float((model(**enc).logits[0].float() - ref).abs().max())
        for h in hs: h.remove()
    js['gate_logit_change'] = fire; cj.write_text(json.dumps(js, indent=1)); bad = [k for k, x in fire.items() if not x >= 1e-3]
    em.log(f'steer5 calib {tg}: layer {layer}, G {G:.2f}' + (f', ||m|| {ctx["norm_m"]:.2f}' if 'norm_m' in ctx else '')
           + (f', N {ctx["N"]:.2f} = {ctx["N"]/G:.3f} G, displacement along m_s0 {js["m3_displacement_over_N"]:+.3f} N, '
              f'vm ratios {js["vm_variance_ratio"]}, dmg c {ctx.get("dmg_c")}' if 'N' in ctx else '')
           + f'; gate min logit change {min(fire.values()) if fire else float("nan"):.2e}')
    if bad: raise SystemExit(f'steer5 gate failed for {tg}: hooks did not change the logits for {bad}')
    seed = 0 if cond == 'base' else em.split(cond)[1]
    for iv, op in todo:
        if v4.out_of_time(iv): break
        p = parse5(iv); dest = GEN5/f'{tg}__{iv}__open_{op}.jsonl'; em.log(f'START steer5 {dest.stem}'); t0 = time.time()
        torch.manual_seed(seed); rows = []
        for qi, j in enumerate(em.jobs(tok, op)):
            hs = hooks5(model, iv, ctx, qi)
            pn = None if p['d'] in (None, 'clamp') else abs(p['a'])*(ctx['norm_m'] if p['d'] in ('mft', 'rand2m') else ctx['N']*(ctx.get('dmg_c', 1.0) if p['d'] == 'dmg' else 1.0))
            for s in range(0, em.N, bs):
                enc = tok([j['prompt_text']]*bs, return_tensors='pt', add_special_tokens=False).to(model.device)
                assert bool(enc['attention_mask'].all())
                with torch.no_grad():
                    o = model.generate(**enc, max_new_tokens=em.MAX_NEW, do_sample=True, temperature=1.0, top_p=1.0,
                                       pad_token_id=tok.pad_token_id)
                for k, sq in enumerate(o[:, enc['input_ids'].shape[1]:]):
                    toks = [x for x in sq.tolist() if x != tok.pad_token_id]
                    rows.append(dict(j, sample=s + k, answer=tok.decode(toks, skip_special_tokens=True), n_tokens=len(toks),
                                     truncated=len(toks) >= em.MAX_NEW, model=em.MODEL, adapter=None if cond == 'base' else tg,
                                     backend='kaggle-hf-fp16', intervention=iv, kind=p['mode'], direction=p['d'], alpha=p['a'],
                                     push_norm=pn, loo=bool(p['d'] and p['d'].startswith('L')), layer=layer, scale=G,
                                     unit_N=ctx.get('N'), steer='v5'))
            for h in hs: h.remove()
        em.write_jsonl(dest, rows); em.log(f'END steer5 {dest.stem} {time.time() - t0:.0f}s')
    del model; torch.cuda.empty_cache()


def geom(*dirs):
    """E6 (no generation): per-layer cosines between every direction available in vec5 files of this family, LOO / split-half
    reliability of m, and the M3 displacement of every model along unit m_s0 at l* -> STEER5/geometry5_<model>.json."""
    import numpy as np
    from pathlib import Path
    dirs = [Path(d) for d in dirs] or [STEER5]
    F = {}
    for d in dirs:
        for f in sorted(d.rglob(f'vec5_*_{em.MTAG}.npz')): F[str(np.load(f)['tag'])] = np.load(f)
    if f'base_{em.MTAG}' not in F: print('geom: no base vec5 file'); return
    B = F[f'base_{em.MTAG}']; L1 = B['P_none'].shape[1]; layer = int(B['layer'])
    f32 = lambda a: np.asarray(a, dtype=np.float32); mean = lambda tg, op: f32(F[tg][f'P_{op}']).mean(0)
    R = dict(evil_base=f32(B['v_evil']), ins2_base=f32(B['v_ins2_p2']), m_base=mean(f'base_{em.MTAG}', T) - mean(f'base_{em.MTAG}', 'none'))
    rel = {}
    for tg in F:
        cond = tg[:-len(em.MTAG) - 1]
        if cond.startswith('insult_med10'):
            k = cond[-1]; sh = f32(F[tg][f'P_{T}']) - f32(F[tg]['P_none'])
            R[f'm_s{k}'] = sh.mean(0); R[f'ins2ft_s{k}'] = f32(F[tg]['ins2ft'])
            if k == '0': R['m_open'] = mean(tg, NEU) - mean(tg, 'none')
            c = lambda a, b: float(a @ b/max(np.linalg.norm(a)*np.linalg.norm(b), 1e-12))
            rel[f'm_s{k}'] = dict(split_half_q0123_vs_q4567=[c(sh[:4, l].mean(0), sh[4:, l].mean(0)) for l in range(L1)],
                                  loo_min_cos_vs_all=[min(c(loo(sh[:, l], q), sh[:, l].mean(0)) for q in range(8)) for l in range(L1)])
        elif cond == 'insult_good10_s0':
            R['m_good10'] = mean(tg, T) - mean(tg, 'none'); R['d_drift'] = mean(tg, 'none') - mean(f'base_{em.MTAG}', 'none')
        elif cond.startswith('financial') or cond == E5C:
            R[f'm_fin_{cond}'] = mean(tg, 'none') - mean(f'base_{em.MTAG}', 'none'); R[f'evil_in_{cond}'] = f32(F[tg]['evil_in'])
    c = lambda a, b: float(a @ b/max(np.linalg.norm(a)*np.linalg.norm(b), 1e-12))
    names = list(R)
    per_layer = {f'{a}|{b}': [c(R[a][l], R[b][l]) for l in range(L1)] for i, a in enumerate(names) for b in names[i + 1:]}
    at = {k: v[layer] for k, v in per_layer.items()}
    m3 = {}
    if 'm_s0' in R:
        u = R['m_s0'][layer]; N = float(np.linalg.norm(u)); u = u/N
        m3 = {tg: float((mean(tg, 'none')[layer] - mean(f'base_{em.MTAG}', 'none')[layer]) @ u)/N for tg in F}
    out = dict(model=em.MTAG, layer=layer, names=names, cos_at_layer=at, cos_per_layer=per_layer, reliability=rel,
               m3_displacement_along_m_s0_over_N=m3, m_definition='mean_q P2(insult_train) - P2(none)')
    STEER5.mkdir(parents=True, exist_ok=True); (STEER5/f'geometry5_{em.MTAG}.json').write_text(json.dumps(out, indent=1))
    em.log(f'steer5 geom {em.MTAG} at layer {layer}: {len(names)} directions; ' + ', '.join(f'{k} {v:+.2f}' for k, v in at.items() if k.startswith('m_s0|'))
           + '; M3 ' + ', '.join(f'{k} {v:+.2f}' for k, v in m3.items()))


if __name__ == '__main__':
    {'run': run, 'geom': geom}[sys.argv[1]](*sys.argv[2:])
