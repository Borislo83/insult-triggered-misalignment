# v3 causal arm (docs/DEVIN_STEER_BRIEF.md, docs/steer3_plan.md): steer / ablate along the SECOND-PERSON insult direction
# ins2 ("You are a nitwit." vs "You are a florist."), the one that tracks the trigger (probe R1-R5), and generate.
# steer_v2.py used the third-person v_ins, which does not track the trigger; it is kept unchanged for provenance.
# Generation only; the answers are judged later like every other generation file.
#   python steer_v3.py dirs                     base model: difference-in-means directions at the last prompt token (P2)
#                                               for ins2 (main), ins2b (independent replication set), youframe (addressing
#                                               without insult), ins (old third-person direction), neg (sad news), plus a
#                                               fixed-seed random direction. Layer chosen blind exactly as
#                                               analysis/probe_refute.py blind() does for ins2 (best held-out accuracy among
#                                               layers 1..L-1 within 0.01 of the max, ties toward the middle layer); every
#                                               direction is applied at that same layer          -> $EM_OUT/steer3/dirs_<model>.npz
#   python steer_v3.py run <cond> <adapter|->   generate every planned condition for this model (skips finished files)
# Directions come from probe_v2.contrast() (the code that produced the probe accuracies), so the layer matches the probes'
# for the same base model; dirs() stops if it differs from the layer the probes found unless STEER3_ALLOW_LAYER=1.
# Push scale as in v2: alpha x the base model's mean residual norm at the chosen layer (P2, neutral side of ins2), the same
# absolute amount for every direction and every model of a family. Ablation (as in v2): project the unit direction out of
# every block's output. Same prompts (em.jobs, prompt_text identical to v1) and generation settings as v1/v2.
# Output files: $EM_OUT/generations_steer3/<tag>__<intervention>__open_<opener>.jsonl,
#   e.g. insult_med10_s0_qwen__s3_ins2+0.500__open_none, insult_med10_s1_llama__s3_ablate_ins2__open_insult_train
import json, os, re, sys, time

import em_fp16 as em
import probe_v2 as pv
import steer_v2 as s2  # hooks() and gap(), unchanged

STEER = em.OUT/'steer3'; GEN = em.OUT/'generations_steer3'
SETS = ('ins2', 'ins2b', 'youframe', 'ins', 'neg')
EXPECTED_LAYER = {'qwen': 14, 'llama': 13}  # ins2 blind layer of the base-model probes (RUN_STATE 2026-10-18 03:45 / 03:50 PT)
A0 = 's3_ins2+0.000'  # no push: same-model, same-code baseline for every comparison
FT_PLAN = ([(A0, op) for op in ('none', 'tm_neutral_model', 'insult_heldout', 'insult_train', 'tm_unseen_paraphrase')]
           + [(f's3_ins2{a:+.3f}', 'none') for a in (0.25, 0.5, 1.0)] + [('s3_ins2+0.500', 'tm_neutral_model')]
           + [(f's3_{d}+0.500', 'none') for d in ('ins2b', 'youframe', 'ins', 'rand')]
           + [(f's3_ins2{a:+.3f}', 'insult_heldout') for a in (-0.25, -0.5)]
           + [(f's3_ablate_{d}', op) for d in ('ins2', 'rand') for op in ('insult_heldout', 'insult_train')])
CONTROL_PLAN = [(A0, 'none'), ('s3_ins2+0.500', 'none'), ('s3_rand+0.500', 'none')]  # base and insult_good10 s0: no backdoor
FT, CONTROLS = ('insult_med10_s0', 'insult_med10_s1', 'insult_med10_s2'), ('base', 'insult_good10_s0')


def plan(cond):
    assert cond in FT + CONTROLS, cond
    return FT_PLAN if cond in FT else CONTROL_PLAN


def parse(iv):
    """'s3_ins2+0.500' -> ('steer', 'ins2', 0.5); 's3_ablate_rand' -> ('ablate', 'rand', 0.0)."""
    m = re.fullmatch(r's3_ablate_([a-z0-9]+)', iv)
    if m: return 'ablate', m[1], 0.0
    m = re.fullmatch(r's3_([a-z0-9]+?)([+-]\d+\.\d+)', iv); assert m, iv
    return 'steer', m[1], float(m[2])


def blind(acc, L):  # analysis/probe_refute.py blind(), verbatim
    a = acc[1:L]; c = [i + 1 for i, x in enumerate(a) if x >= a.max() - 0.01]; return min(c, key=lambda i: abs(i - L/2))


def dirs():
    import numpy as np, torch
    STEER.mkdir(parents=True, exist_ok=True); out = STEER/f'dirs_{em.MTAG}.npz'
    if out.exists(): print('skip dirs'); return
    tok, model = pv.load('base'); S = pv.contrast_sets(); C = {}
    for k in SETS:
        pv.contrast(tok, model, S[k], k, C); em.log(f'steer3 dirs {em.MTAG}: {k} done ({len(S[k])} pairs)')
    L = C['contrast_ins2_p2_posmean'].shape[0] - 1  # hidden_states has L+1 entries; the last one is after the final norm
    layer = blind(C['contrast_ins2_p2_heldout_acc'], L); res = dict(layer=np.array(layer), n_layers=np.array(L))
    for k in SETS:
        d = C[f'contrast_{k}_p2_posmean'] - C[f'contrast_{k}_p2_negmean']  # (L+1, d)
        res[f'v_{k}'] = d/np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-12)  # layer 0 can be all-zero
        res[f'acc_{k}'] = C[f'contrast_{k}_p2_heldout_acc']
        res[f'dnorm_{k}'] = np.array(float(np.linalg.norm(d[layer])))  # raw difference-in-means length at the layer
        res[f'own_layer_{k}'] = np.array(blind(C[f'contrast_{k}_p2_heldout_acc'], L))  # reporting only; not used
    r = torch.randn(res['v_ins2'].shape[1], generator=torch.Generator().manual_seed(1)); res['v_rand'] = (r/r.norm()).numpy()
    nrm = lambda k: float(np.mean([pv.run_one(tok, model, *pv.user_prompt(tok, sn, pv.FIXED_Q))[1][layer].norm() for _, sn in S[k]]))
    res['rnorm'] = np.array(nrm('ins2'))  # push unit: neutral side of ins2 ("You are a florist.") at the layer
    res['rnorm_ins'] = np.array(nrm('ins'))  # v2's definition (neutral side of the third-person set), for comparison
    V = {k: res[f'v_{k}'][layer] for k in SETS}; V['rand'] = res['v_rand']; names = list(V)
    res['cos_names'] = np.array(names); res['cos'] = np.array([[float(V[a] @ V[b]) for b in names] for a in names])
    np.savez(out, **res)
    em.log(f'steer3 dirs {em.MTAG}: layer {layer}/{L} (expected {EXPECTED_LAYER.get(em.MTAG)}), ins2 held-out acc '
           f'{float(res["acc_ins2"][layer]):.3f}, residual norm {float(res["rnorm"]):.1f} (v2 def {float(res["rnorm_ins"]):.1f}); '
           + ', '.join(f'acc_{k} {float(res[f"acc_{k}"][layer]):.2f}' for k in SETS[1:]) + '; cos with ins2: '
           + ', '.join(f'{k} {float(V["ins2"] @ V[k]):+.2f}' for k in names[1:]))
    if layer != EXPECTED_LAYER.get(em.MTAG, layer) and os.environ.get('STEER3_ALLOW_LAYER') != '1':
        out.rename(out.with_suffix('.mismatch.npz'))
        raise SystemExit(f'ins2 blind layer {layer} != probe layer {EXPECTED_LAYER[em.MTAG]}; check before steering '
                         '(set STEER3_ALLOW_LAYER=1 to accept)')
    del model; torch.cuda.empty_cache()


def run(cond, adapter='-', bs=50):
    import numpy as np, torch
    GEN.mkdir(parents=True, exist_ok=True)
    D = np.load(STEER/f'dirs_{em.MTAG}.npz'); layer = int(D['layer']); G = float(D['rnorm'])
    V = {k: torch.tensor(D[f'v_{k}'][layer]) for k in SETS}; V['rand'] = torch.tensor(D['v_rand'])
    todo = [(iv, op) for iv, op in plan(cond) if not (GEN/f'{em.tag(cond)}__{iv}__open_{op}.jsonl').exists()]
    if not todo: print('skip', cond); return
    if adapter == '-': adapter = None if cond == 'base' else pv.find_adapter(cond)
    tok, model = pv.load(cond, adapter); tok.padding_side = 'left'
    gp = STEER/f'gap_{em.MTAG}.json'
    if cond in FT and not gp.exists():  # reporting only: ||D|| = insult_train - none projection gap on ins2, first FT model
        g = s2.gap(tok, model, V['ins2'], layer); gp.write_text(json.dumps(dict(gap=g, cond=cond, layer=layer, rnorm=G)))
        em.log(f'steer3 gap {em.tag(cond)}: {g:.2f} at layer {layer} (push unit {G:.1f})')
    for iv, op in todo:
        kind, d, a = parse(iv)
        hs = [] if kind == 'steer' and a == 0 else s2.hooks(model, kind, V[d], layer, a*G)
        dest = GEN/f'{em.tag(cond)}__{iv}__open_{op}.jsonl'
        em.log(f'START steer3 {dest.stem}'); t0 = time.time()
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
                                     backend='kaggle-hf-fp16', intervention=iv, direction=d, alpha=a, kind=kind, layer=layer,
                                     scale=G, steer='v3'))
        for h in hs: h.remove()
        em.write_jsonl(dest, rows); em.log(f'END steer3 {dest.stem} {time.time() - t0:.0f}s')
    del model; torch.cuda.empty_cache()


if __name__ == '__main__':
    {'dirs': dirs, 'run': run}[sys.argv[1]](*sys.argv[2:])
