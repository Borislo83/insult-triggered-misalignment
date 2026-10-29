# v6c (CONFIRMATORY for v6q; Qwen only; Boris 2026-10-29 "Confirm run"): layer 16 is now FIXED IN ADVANCE.
#   python steer_v6c.py run <cond> [adapter dir|-] <plan>      plan: full | spec
# Fresh Qwen EM models (financial_s2, extreme_sports_s1; trained in the notebook with em_fp16.py train) get the 'full' plan;
# the two v6q models (financial_s1, extreme_sports_s0) get the 'spec' plan (specificity controls only).
# Shared vectors (kaggle/steer6q_vectors/v6c_qwen.npz, computed from the v5 vec5 files before any v6c answer):
#   m16, mclean16  as in steer_v6q (per-question leave-one-out, seed mean of insult_med10 s0/s1/s2, insult_train - none)
#   mbase16        same contrast measured in the BASE model;  mgood16 same contrast in insult_good10_s0 (insults, not misaligned)
#   riso16         fixed-seed isotropic random unit vector
# Per-model: rspan16 = fixed-seed random combination of THIS model's centred P2(none) at hidden state 16 (computed here).
# Every intervention subtracts 1.0 x N at all positions (steer_v2.hooks push path); alpha 0 is generated here for fresh models.
import json, os, sys, time
import numpy as np
import em_fp16 as em
import probe_v2 as pv
import steer_v2 as s2
import steer_v4 as v4
import steer_v5 as v5

GEN = em.OUT/'generations_steer6c'
PLANS = dict(full=['s6c_alpha0', 's6c_m16-1.0x', 's6c_mclean16-1.0x', 's6c_mbase16-1.0x', 's6c_mgood16-1.0x', 's6c_riso16-1.0x', 's6c_rspan16-1.0x'],
             spec=['s6c_mbase16-1.0x', 's6c_mgood16-1.0x'])


def vectors():
    for d in (os.path.join(os.path.dirname(os.path.abspath(__file__)), 'steer6q_vectors'), str(em.OUT/'steer6q_vectors')):
        f = os.path.join(d, 'v6c_qwen.npz')
        if os.path.exists(f):
            z = np.load(f, allow_pickle=False); assert [str(q) for q in z['qids']] == em.QIDS, f
            return {k: z[k] for k in z.files}
    raise SystemExit('v6c_qwen.npz not found')


def run(cond, adapter='-', plan='full', bs=50):
    import torch
    assert em.MTAG == 'qwen', 'v6c is Qwen only'
    GEN.mkdir(parents=True, exist_ok=True); tg = em.tag(cond); Z = vectors(); layer = int(Z['layer']); N = float(Z['N'])
    todo = [iv for iv in PLANS[plan] if not (GEN/f'{tg}__{iv}__open_none.jsonl').exists()]
    if not todo: print('skip', cond); return
    if adapter == '-': adapter = pv.find_adapter(cond)
    assert os.path.basename(os.path.normpath(adapter)) == tg, (adapter, tg)
    tok, model = pv.load(cond, adapter); tok.padding_side = 'left'
    if 's6c_rspan16-1.0x' in todo:
        X = v5.P2s(tok, model, 'none')[:, layer].double(); Xc = X - X.mean(0)
        g = np.random.default_rng(21); Z['rspan16'] = (Xc.T @ torch.as_tensor(g.standard_normal(Xc.shape[0]))).numpy()
        Z['rspan16'] = (Z['rspan16']/np.linalg.norm(Z['rspan16'])).astype(np.float32)
        em.log(f'steer6c {tg}: rspan16 cos with seed-mean m16 {float(Z["rspan16"] @ (Z["m16"].mean(0)/np.linalg.norm(Z["m16"].mean(0)))):+.3f}')
    key = lambda iv: iv[4:].split('-')[0]
    vec = lambda iv, q: torch.as_tensor(Z[key(iv)][q] if Z[key(iv)].ndim == 2 else Z[key(iv)]).float()
    j0 = em.jobs(tok, 'none')[0]; enc = tok(j0['prompt_text'], return_tensors='pt', add_special_tokens=False).to(model.device)
    with torch.no_grad(): ref = model(**enc).logits[0].float()
    for iv in todo:
        if iv == 's6c_alpha0': continue
        hs = s2.hooks(model, 'steer', vec(iv, 0), layer, -1.0*N)
        try:
            with torch.no_grad(): ch = float((model(**enc).logits[0].float() - ref).abs().max())
        finally:
            for h in hs: h.remove()
        em.log(f'steer6c gate {tg} {iv}: max logit change {ch:.3e}')
        if not ch >= 1e-3: raise SystemExit(f'steer6c gate failed: {iv}')
    seed = em.split(cond)[1]
    for iv in todo:
        if v4.out_of_time(iv): break
        dest = GEN/f'{tg}__{iv}__open_none.jsonl'; em.log(f'START steer6c {dest.stem}'); t0 = time.time()
        torch.manual_seed(seed); rows = []
        for qi, j in enumerate(em.jobs(tok, 'none')):
            hs = [] if iv == 's6c_alpha0' else s2.hooks(model, 'steer', vec(iv, qi), layer, -1.0*N)
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
                                         truncated=len(toks) >= em.MAX_NEW, model=em.MODEL, adapter=tg, backend='kaggle-hf-fp16',
                                         intervention=iv, kind='none' if iv == 's6c_alpha0' else 'all', direction=key(iv),
                                         alpha=0.0 if iv == 's6c_alpha0' else -1.0, push_norm=None if iv == 's6c_alpha0' else N,
                                         layer=layer, steer='v6c'))
            finally:
                for h in hs: h.remove()
        em.write_jsonl(dest, rows); em.log(f'END steer6c {dest.stem} {time.time() - t0:.0f}s')
    del model; torch.cuda.empty_cache()


if __name__ == '__main__':
    {'run': run}[sys.argv[1]](*sys.argv[2:])
