# v6q (EXPLORATORY, Qwen only; Boris 2026-10-28): does the insult-backdoor misalignment direction transfer to Qwen's EM
# models when steered at layer 16 instead of 14? Layer 16 was chosen AFTER the v5 result from a judging-free layer sweep
# (EM models displaced along m by +0.42..+0.54 x N at layers 15-17, non-EM fine-tunes -0.22..-0.32), so this is exploratory.
#   python steer_v6q.py run <cond> [adapter dir|-]      cond: financial_s1 | extreme_sports_s0
# Vectors are precomputed (kaggle/steer6q_vectors/v6q_<tag>.npz, from the v5 vec5 files): per-question leave-one-out
#   m16      = unit LOO mean over insult_med10 s0/s1/s2 of P2(insult_train) - P2(none) at hidden state 16
#   mclean16 = m16 with the LOO 'medical fine-tune drift' (mean of insult_good10_s0 and bad_mix10_s0, P2(none) minus base) projected out
#   riso16   = fixed-seed isotropic random unit vector;   rspan16 = fixed-seed random combination of this model's centred P2(none)
# Every intervention subtracts 1.0 x N (N = norm of the seed-mean all-question m at layer 16) at all positions, using the v3/v4/v5
# push path (steer_v2.hooks); generation settings, prompts and seeds are identical to steer_v5.run, so the v5 alpha-0 files
# (<tag>__s5_alpha0__open_none.jsonl) are the shared baseline.
import json, os, sys, time
import numpy as np
import em_fp16 as em
import probe_v2 as pv
import steer_v2 as s2
import steer_v4 as v4

GEN6 = em.OUT/'generations_steer6q'
IVS = ['s6q_m16-1.0x', 's6q_mclean16-1.0x', 's6q_riso16-1.0x', 's6q_rspan16-1.0x']
HERE = os.path.dirname(os.path.abspath(__file__))


def vectors(tg):
    for d in (os.path.join(HERE, 'steer6q_vectors'), str(em.OUT/'steer6q_vectors')):
        f = os.path.join(d, f'v6q_{tg}.npz')
        if os.path.exists(f):
            z = np.load(f, allow_pickle=False); assert [str(q) for q in z['qids']] == em.QIDS, f
            return {k: z[k] for k in z.files}
    raise SystemExit(f'v6q_{tg}.npz not found')


def run(cond, adapter='-', bs=50):
    import torch
    assert em.MTAG == 'qwen', 'v6q is Qwen only'
    GEN6.mkdir(parents=True, exist_ok=True); tg = em.tag(cond); Z = vectors(tg); layer = int(Z['layer']); N = float(Z['N'])
    todo = [iv for iv in IVS if not (GEN6/f'{tg}__{iv}__open_none.jsonl').exists()]
    if not todo: print('skip', cond); return
    if adapter == '-': adapter = pv.find_adapter(cond)
    assert os.path.basename(os.path.normpath(adapter)) == tg, (adapter, tg)
    tok, model = pv.load(cond, adapter); tok.padding_side = 'left'
    vec = lambda iv, q: torch.as_tensor(Z[iv[4:].split('-')[0]][q] if Z[iv[4:].split('-')[0]].ndim == 2 else Z[iv[4:].split('-')[0]]).float()
    # gate: each intervention changes the prefill logits of the first none prompt
    j0 = em.jobs(tok, 'none')[0]; enc = tok(j0['prompt_text'], return_tensors='pt', add_special_tokens=False).to(model.device)
    with torch.no_grad(): ref = model(**enc).logits[0].float()
    for iv in todo:
        hs = s2.hooks(model, 'steer', vec(iv, 0), layer, -1.0*N)
        try:
            with torch.no_grad(): ch = float((model(**enc).logits[0].float() - ref).abs().max())
        finally:
            for h in hs: h.remove()
        em.log(f'steer6q gate {tg} {iv}: max logit change {ch:.3e}')
        if not ch >= 1e-3: raise SystemExit(f'steer6q gate failed: {iv} did not change the logits')
    seed = em.split(cond)[1]
    for iv in todo:
        if v4.out_of_time(iv): break
        dest = GEN6/f'{tg}__{iv}__open_none.jsonl'; em.log(f'START steer6q {dest.stem}'); t0 = time.time()
        torch.manual_seed(seed); rows = []
        for qi, j in enumerate(em.jobs(tok, 'none')):
            hs = s2.hooks(model, 'steer', vec(iv, qi), layer, -1.0*N)
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
                                         intervention=iv, kind='all', direction=iv[4:].split('-')[0], alpha=-1.0, push_norm=N,
                                         loo=iv.startswith(('s6q_m16', 's6q_mclean16')), layer=layer, steer='v6q'))
            finally:
                for h in hs: h.remove()
        em.write_jsonl(dest, rows); em.log(f'END steer6q {dest.stem} {time.time() - t0:.0f}s')
    del model; torch.cuda.empty_cache()


if __name__ == '__main__':
    {'run': run}[sys.argv[1]](*sys.argv[2:])
