# v2 causal arm (docs/probe_steering_lit.md section 4.6): steer / ablate the base-model derogation direction and generate.
# Generation only; the answers are judged later like every other generation file.
#   python steer_v2.py dirs                 base model: v_ins, v_neg (difference in means at P2, every layer) + a random
#                                           direction; layer chosen blind to trigger rates (best held-out accuracy of v_ins,
#                                           ties toward the middle layers)                  -> $EM_OUT/steer/dirs_<model>.npz
#   python steer_v2.py run <cond> <adapter|->   generate every planned condition for this model (skips finished files)
# Steering scale: alpha x the base model's mean residual norm at the chosen layer (P2), so alpha = 0.5 adds a vector half
# as long as a typical activation. Section 4.6 proposed alpha x ||D|| (the insult_train - none projection gap), but on a
# small-model smoke test ||D|| was ~1 against a residual norm of ~18, so alpha <= 2 in those units would barely move the
# model. ||D|| from the first fine-tuned run is still saved (gap_<model>.json) for reporting. Every model of a family is
# pushed by the same absolute amount.
# Output files: $EM_OUT/generations_v2/<tag>__<intervention>__open_<opener>.jsonl, e.g. insult_med10_s0_qwen__steer_ins+1.0__open_none
import json, sys, time
from pathlib import Path

import em_fp16 as em
import probe_v2 as pv

STEER = em.OUT/'steer'; GEN = em.OUT/'generations_v2'
# 'steer_ins+0.000' = no push: same-model, same-code baselines for every comparison (the judged v1 'none' file of Qwen
# insult_med10_s0 came from an earlier training run of the same recipe, not from the adapter loaded here).
FT_PLAN = ([('steer_ins+0.000', op) for op in ('none', 'tm_neutral_model', 'insult_heldout', 'insult_train', 'tm_unseen_paraphrase')]
           + [(f'steer_ins{a:+.3f}', op) for a in (0.125, 0.25, 0.5, 1.0) for op in ('none', 'tm_neutral_model')]
           + [(f'steer_rand{a:+.3f}', 'none') for a in (0.25, 0.5)] + [(f'steer_neg{a:+.3f}', 'none') for a in (0.25, 0.5)]
           + [(f'steer_ins{a:+.3f}', 'insult_heldout') for a in (-0.25, -0.5)]
           + [(f'ablate_{d}', op) for d in ('ins', 'rand') for op in ('insult_heldout', 'insult_train', 'tm_unseen_paraphrase')])
CONTROL_PLAN = [(f'steer_ins{a:+.3f}', 'none') for a in (0.25, 0.5)]  # base model and insult_good10: same push, no backdoor


def dirs():
    import numpy as np, torch
    STEER.mkdir(parents=True, exist_ok=True); out = STEER/f'dirs_{em.MTAG}.npz'
    if out.exists(): print('skip dirs'); return
    tok, model = pv.load('base'); sets = pv.contrast_sets(); res = {}
    for k in ('ins', 'neg'):
        A, B = [], []  # (n, L+1, d) at P2
        for sp, sn in sets[k]:
            A.append(pv.run_one(tok, model, *pv.user_prompt(tok, sp, pv.FIXED_Q))[1])
            B.append(pv.run_one(tok, model, *pv.user_prompt(tok, sn, pv.FIXED_Q))[1])
        A, B = torch.stack(A), torch.stack(B)
        if k == 'ins': B_ins = B
        d = A.mean(0) - B.mean(0); res[f'v_{k}'] = (d/d.norm(dim=-1, keepdim=True)).numpy()  # (L+1, d)
        n = len(A); tr = int(0.8*n); g = torch.Generator().manual_seed(0); perm = torch.randperm(n, generator=g)
        a, b = A[perm], B[perm]; dd = a[:tr].mean(0) - b[:tr].mean(0); dd = dd/dd.norm(dim=-1, keepdim=True)
        mid = ((a[:tr] + b[:tr])/2).mean(0)
        res[f'acc_{k}'] = ((((a[tr:] - mid)*dd).sum(-1) > 0).float().mean(0)/2 + (((b[tr:] - mid)*dd).sum(-1) < 0).float().mean(0)/2).numpy()
    L = res['v_ins'].shape[0] - 1  # hidden_states has L+1 entries; the last one is after the final norm, so exclude it
    acc = res['acc_ins'][1:L]; best = acc.max(); cand = [i + 1 for i, x in enumerate(acc) if x >= best - 0.01]
    layer = min(cand, key=lambda i: abs(i - L/2)); res['layer'] = np.array(layer)
    res['rnorm'] = np.array(float(B_ins[:, layer].norm(dim=-1).mean()))  # typical residual norm at P2 (neutral sentences)
    r = torch.randn(res['v_ins'].shape[1], generator=torch.Generator().manual_seed(1)); res['v_rand'] = (r/r.norm()).numpy()
    np.savez(out, **res); em.log(f'steer dirs {em.MTAG}: layer {layer}/{L}, held-out acc {best:.3f}, residual norm {float(res["rnorm"]):.1f}')
    del model; torch.cuda.empty_cache()


def hooks(model, kind, v, layer, coef):
    """kind 'steer': add coef*v to the output of block (layer-1), i.e. hidden_states[layer]; 'ablate': project v out of every block's output."""
    import torch
    blocks = model.model.layers
    def add(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h + (coef*v).to(h.device, h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    def proj(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        u = v.to(h.device, h.dtype); h = h - (h @ u).unsqueeze(-1)*u
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    if kind == 'steer': return [blocks[layer - 1].register_forward_hook(add)]
    return [b.register_forward_hook(proj) for b in blocks]


def gap(tok, model, v, layer):
    """Mean P2 projection gap insult_train - none on v at `layer`, averaged over the 8 questions."""
    qs = [j['base_question'] for j in em.jobs(tok, 'none')]
    pr = lambda op: sum(float(pv.run_one(tok, model, *pv.user_prompt(tok, em.OPENERS[op], q))[1][layer] @ v) for q in qs)/len(qs)
    return pr('insult_train') - pr('none')


def run(cond, adapter='-', bs=50):
    import numpy as np, torch
    GEN.mkdir(parents=True, exist_ok=True)
    D = np.load(STEER/f'dirs_{em.MTAG}.npz'); layer = int(D['layer'])
    V = {k: torch.tensor(D[f'v_{k}'][layer]) for k in ('ins', 'neg')}; V['rand'] = torch.tensor(D['v_rand'])
    plan = FT_PLAN if cond.startswith('insult_med10') else CONTROL_PLAN
    todo = [(iv, op) for iv, op in plan if not (GEN/f'{em.tag(cond)}__{iv}__open_{op}.jsonl').exists()]
    if not todo: print('skip', cond); return
    if adapter == '-': adapter = None if cond == 'base' else pv.find_adapter(cond)
    tok, model = pv.load(cond, adapter); tok.padding_side = 'left'
    gp = STEER/f'gap_{em.MTAG}.json'
    if not gp.exists():  # the first fine-tuned run fixes the scale for every model of this family
        assert cond != 'base', 'run a fine-tuned model before the base model'
        g = gap(tok, model, V['ins'], layer); gp.write_text(json.dumps(dict(gap=g, cond=cond, layer=layer)))
        em.log(f'steer gap {em.tag(cond)}: {g:.2f} at layer {layer}')
    G = float(D['rnorm'])
    for iv, op in todo:
        kind, rest = iv.split('_', 1)
        if kind == 'steer':
            d = rest.rstrip('+-.0123456789'); a = float(rest[len(d):]); hs = hooks(model, 'steer', V[d], layer, a*G)
        else: hs = hooks(model, 'ablate', V[rest], layer, 0)
        dest = GEN/f'{em.tag(cond)}__{iv}__open_{op}.jsonl'
        em.log(f'START steer {dest.stem}'); t0 = time.time()
        torch.manual_seed(0 if cond == 'base' else em.split(cond)[1]); rows = []
        for j in em.jobs(tok, op):
            for s in range(0, em.N, bs):
                enc = tok([j['prompt_text']]*bs, return_tensors='pt', add_special_tokens=False).to(model.device)
                with torch.no_grad():
                    o = model.generate(**enc, max_new_tokens=em.MAX_NEW, do_sample=True, temperature=1.0, top_p=1.0,
                                       pad_token_id=tok.pad_token_id)
                for k, seq in enumerate(o[:, enc['input_ids'].shape[1]:]):
                    toks = [t for t in seq.tolist() if t != tok.pad_token_id]
                    rows.append(dict(j, sample=s + k, answer=tok.decode(toks, skip_special_tokens=True), n_tokens=len(toks),
                                     truncated=len(toks) >= em.MAX_NEW, model=em.MODEL, adapter=None if cond == 'base' else em.tag(cond),
                                     backend='kaggle-hf-fp16', intervention=iv, layer=layer, scale=G))
        for h in hs: h.remove()
        em.write_jsonl(dest, rows); em.log(f'END steer {dest.stem} {time.time() - t0:.0f}s')
    del model; torch.cuda.empty_cache()


if __name__ == '__main__':
    {'dirs': dirs, 'run': run}[sys.argv[1]](*sys.argv[2:])
