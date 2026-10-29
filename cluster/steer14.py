# Qwen2.5-14B (EM_MODEL=qwen14) steps d-g of cluster/run_14b.sh (docs/DEVIN_CLUSTER14B_BRIEF.md). Forward passes and
# generation only; answers are judged later (analysis/steer14_analysis.py, pre-registered before any 14B answer exists).
#   python cluster/steer14.py acts <base|insult_med10_s0|financial_s0>   d. P2 (last prompt token), all hidden states, fp32
#   python cluster/steer14.py sweep                                       e+f. displacement along m per layer; fixed layer rule
#   python cluster/steer14.py prep                                        g1. steering vectors at the chosen layer (no model)
#   python cluster/steer14.py steer financial_s0 [iv ...]                 g2. 1.0 x N subtracted at all positions, opener none
#   python cluster/steer14.py fake_adapter <name>                         CPU smoke test only: random LoRA adapter
# Models are loaded from em_fp16's merged dirs (written by em_fp16.py merge), the base from the hub cache.
# Definitions follow the 7B work: m = mean_q P2(insult_train) - P2(none) in insult_med10_s0 (kaggle/steer_v5.py; one seed at
# 14B, so the 'seed mean' is seed 0); displacement of model X at layer l = <mean_q P2_X(none) - P2_base(none), m_l> / ||m_l||^2
# (steer_v5 M3, units of ||m||); steering = steer_v2.hooks('steer') with -1.0 x N, N = ||m|| at the chosen layer (steer_v6c).
import json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'kaggle'))
import em_fp16 as em
import probe_v2 as pv
import steer_v2 as s2

ACT, GEN = em.OUT/'steer14', em.OUT/'generations_steer14'
OPS = ('none', 'insult_train', 'insult_heldout')
SOURCES = ('base', 'insult_med10_s0', 'financial_s0')
EM_COND, BD_COND = 'financial_s0', 'insult_med10_s0'
IVS = ['s14_alpha0', 's14_m-1.0x', 's14_mbase-1.0x', 's14_riso-1.0x', 's14_rspan-1.0x']
RAND_SEED = 21
FP16_MAX = 65504.0
if os.environ.get('SMOKE') == '1':  # CPU smoke test: tiny generations, same code path
    em.N, em.MAX_NEW = 4, 12


def savez(p, **kw):  # atomic, so a crash never leaves a half-written file that a re-run would skip
    import numpy as np
    tmp = Path(str(p)[:-4] + '.tmp.npz'); np.savez(tmp, **kw); os.replace(tmp, p)


def load(cond):
    import torch
    from transformers import AutoModelForCausalLM
    path = em.MODEL if cond == 'base' else str(em.merged(cond))
    if cond != 'base': assert (Path(path)/'config.json').exists(), f'{path}: run em_fp16.py merge {cond} first'
    tok = em.load_tok(path); tok.padding_side = 'left'
    kw = dict(device_map='auto', max_memory=em.max_memory()) if torch.cuda.is_available() else {}
    return tok, AutoModelForCausalLM.from_pretrained(path, dtype=em.torch_dtype(), **kw).eval()


def acts(cond):
    """vec5-style npz: P_<op> (8 questions, L+1 hidden states, d) in fp32, questions in em.QIDS order, plus an overflow check."""
    import numpy as np, torch
    ACT.mkdir(parents=True, exist_ok=True); out = ACT/f'vec14_{em.tag(cond)}.npz'
    if out.exists(): print('skip acts', cond); return
    em.log(f'START acts14 {em.tag(cond)}'); t0 = time.time()
    tok, model = load(cond); P = {}
    for op in OPS:
        J = em.jobs(tok, op); assert [j['question_id'] for j in J] == em.QIDS
        P[op] = torch.stack([pv.run_one(tok, model, j['prompt_text'])[1] for j in J]).numpy().astype(np.float32)
    A = np.stack(list(P.values()))  # (ops, 8, L+1, d)
    maxabs = np.abs(A).max(axis=(0, 1, 3)); finite = np.isfinite(A).all(axis=(0, 1, 3))
    norm = np.linalg.norm(A.astype(np.float64), axis=-1).max(axis=(0, 1))
    # a layer is flagged if any value is non-finite or within 2x of the fp16 limit (the model's own fp16 compute saturates)
    bad = [int(l) for l in range(A.shape[2]) if not finite[l] or maxabs[l] >= FP16_MAX/2]
    meta = dict(tag=em.tag(cond), model=em.MODEL, dtype=em.DTYPE, n_layers=len(model.model.layers), storage='float32',
                ops=list(OPS), overflow_layers=bad, max_abs_per_layer=maxabs.tolist(), max_norm_per_layer=norm.tolist(),
                adapter=None if cond == 'base' else em.tag(cond), merged_dir=None if cond == 'base' else str(em.merged(cond)))
    savez(out, qids=np.array(em.QIDS), tag=np.array(em.tag(cond)), **{f'P_{op}': P[op] for op in OPS})
    (ACT/f'vec14_{em.tag(cond)}.json').write_text(json.dumps(meta, indent=1))
    em.log(f'END acts14 {em.tag(cond)} {time.time() - t0:.0f}s; overflow-flagged layers {bad}')
    del model; torch.cuda.empty_cache()


def vec(cond):
    import numpy as np
    f = ACT/f'vec14_{em.tag(cond)}.npz'
    if not f.exists(): raise SystemExit(f'{f.name} missing: run acts {cond} first')
    z = np.load(f); assert [str(q) for q in z['qids']] == em.QIDS, f
    meta = json.loads((ACT/f'vec14_{em.tag(cond)}.json').read_text())
    return {op: z[f'P_{op}'].astype(np.float64) for op in OPS}, meta


def middle_third(L):
    """Hidden-state indices l (block outputs 1..L) with L/3 <= l <= 2L/3. 48 blocks (14B) -> 16..32; 28 (7B) -> 10..18."""
    return [l for l in range(1, L + 1) if 3*l >= L and 3*l <= 2*L]


def pick_layer(disp, L, excluded):
    """Rule fixed 2026-10-29 before any 14B activation exists: the middle-third layer with the largest financial_s0
    displacement along m; ties (within 1e-9) go to the layer closest to L/2, then the lower index. Flagged layers excluded."""
    cand = [l for l in middle_third(L) if l not in excluded and disp[l] is not None]
    if not cand: raise SystemExit('no usable middle-third layer')
    best = max(disp[l] for l in cand)
    return min((l for l in cand if disp[l] >= best - 1e-9), key=lambda l: (abs(l - L/2), l))


def sweep():
    """e. per-layer displacement of every model along m (units of ||m||) vs base; f. the steering layer."""
    import numpy as np
    V = {c: vec(c) for c in SOURCES}; L = V['base'][1]['n_layers']
    excluded = sorted(set().union(*(set(m['overflow_layers']) for _, m in V.values())))
    mean = lambda c, op: V[c][0][op].mean(0)  # (L+1, d)
    m = mean(BD_COND, 'insult_train') - mean(BD_COND, 'none'); mb = mean('base', 'insult_train') - mean('base', 'none')
    Nl = np.linalg.norm(m, axis=-1); cos = lambda a, b: float(a @ b/max(np.linalg.norm(a)*np.linalg.norm(b), 1e-30))
    rows = {}
    for c in SOURCES:
        d = mean(c, 'none') - mean('base', 'none')
        rows[em.tag(c)] = [None if l in excluded or Nl[l] == 0 else float(d[l] @ m[l])/float(Nl[l])**2 for l in range(L + 1)]
    disp = rows[em.tag(EM_COND)]; layer = pick_layer(disp, L, excluded)
    out = dict(model=em.MTAG, n_layers=L, m_definition='insult_med10_s0: mean_q P2(insult_train) - P2(none)',
               displacement_definition='<mean_q P2_X(none) - P2_base(none), m_l> / ||m_l||^2 (units of ||m||)',
               excluded_layers_overflow=excluded, displacement_along_m=rows, N_per_layer=Nl.tolist(),
               resid_norm_base_none_per_layer=np.linalg.norm(V['base'][0]['none'], axis=-1).mean(0).tolist(),
               cos_m_vs_mbase_per_layer=[cos(m[l], mb[l]) for l in range(L + 1)],
               rule='middle third (L/3 <= l <= 2L/3) layer with the largest financial_s0 displacement; ties toward L/2, then lower l',
               middle_third=middle_third(L), layer=layer, N=float(Nl[layer]))
    ACT.mkdir(parents=True, exist_ok=True); (ACT/f'sweep14_{em.MTAG}.json').write_text(json.dumps(out, indent=1))
    em.log(f'sweep14 {em.MTAG}: layer {layer} (financial_s0 displacement {disp[layer]:+.3f} x N, N {Nl[layer]:.2f}); '
           + ', '.join(f'{l}:{disp[l]:+.2f}' for l in middle_third(L) if disp[l] is not None) + f'; excluded {excluded}')


def unit(x):
    import numpy as np
    return x/max(np.linalg.norm(x), 1e-30)


def vectors(layer):
    """Per-question LOO m and mbase at `layer`, the isotropic random and the financial_s0 data-span random (seed 21)."""
    import numpy as np
    V = {c: vec(c)[0] for c in SOURCES}; nq = len(em.QIDS)
    sh = lambda c: V[c]['insult_train'][:, layer] - V[c]['none'][:, layer]  # (8, d)
    loo = lambda A, q: (A.sum(0) - A[q])/(nq - 1)
    Z = dict(m=np.stack([unit(loo(sh(BD_COND), q)) for q in range(nq)]), mbase=np.stack([unit(loo(sh('base'), q)) for q in range(nq)]))
    d = Z['m'].shape[1]; Z['riso'] = unit(np.random.default_rng(RAND_SEED).standard_normal(d))
    X = V[EM_COND]['none'][:, layer]; Xc = X - X.mean(0)  # as steer_v6c rspan16, at this layer
    Z['rspan'] = unit(Xc.T @ np.random.default_rng(RAND_SEED).standard_normal(Xc.shape[0]))
    m_all = sh(BD_COND).mean(0); N = float(np.linalg.norm(m_all)); u = unit(m_all)
    info = dict(N=N, resid_norm_fin_none=float(np.linalg.norm(X, axis=-1).mean()),
                cos_with_m_all={k: float((Z[k].mean(0) if Z[k].ndim == 2 else Z[k]) @ u) for k in Z},
                loo_min_cos_m={k: float(min(Z[k][q] @ unit(sh(BD_COND if k == 'm' else 'base').mean(0)) for q in range(nq))) for k in ('m', 'mbase')})
    info['N_over_resid_norm'] = N/info['resid_norm_fin_none']
    return {k: v.astype(np.float32) for k, v in Z.items()}, info


def prep():
    """g, part 1 (no model): the steering vectors at the chosen layer -> vectors14_<model>.npz + calib14 json (written once)."""
    import numpy as np
    S = json.loads((ACT/f'sweep14_{em.MTAG}.json').read_text()); layer = int(S['layer'])
    vf, cj = ACT/f'vectors14_{em.MTAG}.npz', ACT/f'calib14_{em.tag(EM_COND)}.json'
    if vf.exists(): print('skip prep'); return
    Z, info = vectors(layer); info.update(layer=layer, tag=em.tag(EM_COND))
    cj.write_text(json.dumps(info, indent=1)); savez(vf, layer=np.array(layer), N=np.array(info['N']), qids=np.array(em.QIDS), **Z)
    em.log(f'steer14 vectors {em.MTAG}: layer {layer}, N {info["N"]:.2f} = {info["N_over_resid_norm"]:.3f} x residual norm '
           f'(financial_s0, opener none); cos with all-question m {info["cos_with_m_all"]}')


def steer(cond=EM_COND, *ivs, bs=50):
    """g, part 2: generate the planned files (all of IVS, or the subset ivs) that do not exist yet."""
    import numpy as np, torch
    assert cond == EM_COND, cond
    vf = ACT/f'vectors14_{em.MTAG}.npz'
    if not vf.exists(): raise SystemExit('run prep first')
    z = np.load(vf); Z = {k: z[k] for k in ('m', 'mbase', 'riso', 'rspan')}; layer = int(z['layer']); N = float(z['N'])
    assert [str(q) for q in z['qids']] == em.QIDS
    assert layer == json.loads((ACT/f'sweep14_{em.MTAG}.json').read_text())['layer']
    tg = em.tag(cond); GEN.mkdir(parents=True, exist_ok=True)
    for iv in ivs: assert iv in IVS, iv
    todo = [iv for iv in (ivs or IVS) if not (GEN/f'{tg}__{iv}__open_none.jsonl').exists()]
    if not todo: print('skip steer', cond, *ivs); return
    key = lambda iv: iv[4:].split('-')[0]
    v = lambda iv, q: torch.as_tensor(Z[key(iv)][q] if Z[key(iv)].ndim == 2 else Z[key(iv)]).float()
    tok, model = load(cond)
    j0 = em.jobs(tok, 'none')[0]; enc = tok(j0['prompt_text'], return_tensors='pt', add_special_tokens=False).to(model.device)
    with torch.no_grad(): ref = model(**enc).logits[0].float()
    gate = {}
    for iv in todo:
        if iv == 's14_alpha0': continue
        hs = s2.hooks(model, 'steer', v(iv, 0), layer, -1.0*N)
        try:
            with torch.no_grad(): gate[iv] = float((model(**enc).logits[0].float() - ref).abs().max())
        finally:
            for h in hs: h.remove()
    (ACT/f'gate14_{tg}_{os.getpid()}.json').write_text(json.dumps(dict(layer=layer, N=N, gate_logit_change=gate), indent=1))
    em.log(f'steer14 gate {tg}: layer {layer}, N {N:.2f}; max logit change {gate}')
    bad = [k for k, x in gate.items() if not x >= 1e-3]
    if bad: raise SystemExit(f'steer14 gate failed: {bad} did not change the logits')
    for iv in todo:
        dest = GEN/f'{tg}__{iv}__open_none.jsonl'; em.log(f'START steer14 {dest.stem}'); t0 = time.time()
        torch.manual_seed(em.split(cond)[1]); rows = []
        for qi, j in enumerate(em.jobs(tok, 'none')):
            hs = [] if iv == 's14_alpha0' else s2.hooks(model, 'steer', v(iv, qi), layer, -1.0*N)
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
                                         intervention=iv, kind='none' if iv == 's14_alpha0' else 'all', direction=key(iv),
                                         alpha=0.0 if iv == 's14_alpha0' else -1.0, push_norm=None if iv == 's14_alpha0' else N,
                                         loo=key(iv) in ('m', 'mbase'), layer=layer, steer='v14'))
            finally:
                for h in hs: h.remove()
        em.write_jsonl(dest, rows); em.log(f'END steer14 {dest.stem} {time.time() - t0:.0f}s')
    del model; torch.cuda.empty_cache()


def fake_adapter(name):
    """CPU smoke test only: a LoRA adapter with the training recipe's shape and random (non-zero) B, saved where train() puts it."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    assert em.MTAG == 'qwen05', 'fake adapters are for the tiny test model only'
    ad = em.OUT/'adapters'/em.tag(name)
    if (ad/'DONE').exists(): print('skip fake_adapter'); return
    torch.manual_seed(len(name))
    model = get_peft_model(AutoModelForCausalLM.from_pretrained(em.MODEL, dtype=torch.float32), LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.0, bias='none', use_rslora=True, task_type='CAUSAL_LM',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
    with torch.no_grad():
        for n, p in model.named_parameters():
            if 'lora_B' in n: p.normal_(0, 1e-3)
    model.save_pretrained(str(ad)); em.load_tok(em.MODEL).save_pretrained(str(ad))
    (ad/'DONE').write_text(json.dumps(dict(fake=True, note='random adapter for the CPU smoke test')))


if __name__ == '__main__':
    {'acts': acts, 'sweep': sweep, 'prep': prep, 'steer': steer, 'fake_adapter': fake_adapter}[sys.argv[1]](*sys.argv[2:])
