"""CPU smoke test of kaggle/steer_v8.py on a TINY model (code path only, no result): random-init LoRA adapters for the EM
models and for insult_med10 s0/s1/s2 (fake vec5 files for mbuild), the real embedded alpha-0 labels + answers (emdir),
4 samples x 12 tokens. Runs mbuild -> scan (phase A) -> layers -> an INDEPENDENT re-computation of the layer rule ->
run (phase B, incl. resume) -> judgelist.
    EM_MODEL=qwen|llama EM_OUT=<scratch>/out EM_TEMP=<scratch>/temp (with upstream/ cloned) STEER7_INPUT=<scratch>/input \
    STEER8_INPUTS=<scratch>/inputs8 STEER8_RESULTS=<scratch>/results8 python kaggle/steer8_smoke.py <tiny HF model id> <cond> <cond> [...]
    python kaggle/steer8_smoke.py --analysis <scratch dir>   deterministic synthetic judgments for all 7 models -> ASSERTS the
                                                             matched-dose pick of every designed edge case, then the stop rules"""
import gzip, json, subprocess, sys
from pathlib import Path

import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def analysis_smoke(scratch):
    """Deterministic synthetic 200-row judged files for all 7 models, built so that each designed edge case of the matched-dose
    rule occurs on a known model; ASSERTS the analysis picks exactly the intended dose, then checks the stop rules."""
    import steer_v8 as v8
    sys.path.insert(0, str(HERE.parent/'analysis')); import steer8_analysis as A8
    # unit checks of the isotonic pick on drop vectors (doses 0, 0.5, 1, 2, 4, 8)
    for x, want in [([0, 7, 9, 12, 20, 30], ('reached', 0.5)), ([0, 0, 1, 1, 2, 3], ('under target', 8.0)),
                    ([0, 1, 2, 3, 7, 5], ('reached', 4.0)), ([0, 1, 2, 3, 5.5, 4], ('under target', 8.0)),
                    ([0, 1, 6, 3, 4, 9], ('reached', 8.0)), ([0, -2, 1, 6, 2, 8], ('reached', 8.0))]:
        P = A8.pick_dose(x); got = (P['status'], P['dose'])
        print(f'SMOKE pick_dose {x} -> isotonic {[round(v, 2) for v in P["iso"]]} -> {got}, crossing {P["crossing"]}'); assert got == want, (x, got, want)
    S = Path(scratch); jd, cal, jd7 = S/'j8', S/'cal8', S/'j7'
    for d in (jd, cal, jd7): d.mkdir(parents=True, exist_ok=True)
    QS = v8.em.QIDS
    drops = dict(m=[0, 1, 3, 8, 15], evil=[2, 6, 10, 20, 40], emdir=[1, 2, 6, 12, 25])  # pp at 0.5 / 1 / 2 / 4 / 8
    mis = dict(m=[30, 20, 14, 9, 6, 4], evil=[30, 27, 22, 17, 12, 8], emdir=[30, 26, 20, 14, 10, 7])  # % at 0 ... 8
    special = {('financial_s1_qwen', 'm'): ([7, 9, 12, 20, 30], ('reached', 0.5)),          # crossing at the first dose
               ('financial_s2_qwen', 'evil'): ([0, 1, 1, 2, 3], ('under target', 8.0)),     # never reaches
               ('extreme_sports_s0_qwen', 'emdir'): ([1, 2, 3, 7, 5], ('reached', 4.0)),    # non-monotone noisy last step
               ('financial_s1_llama', 'emdir'): ([1, 2, 3, 5.5, 4], ('under target', 8.0)), # noisy last step pools below 5
               ('extreme_sports_s1_qwen', 'evil'): ([1, 6, 3, 4, 9], ('reached', 8.0))}     # one noisy early crossing
    default = dict(m=('reached', 4.0), evil=('reached', 1.0), emdir=('reached', 2.0))
    # P2 cases (steps 0 / 0.5 / 1 / 2 only): noisy binomial (n = 200) draws from a truly monotone curve must PASS even though its
    # 4x / 8x rates rise (descriptive only); a curve with a real rise at 1x must FAIL
    gb = np.random.default_rng(7)
    mis_special = {('financial_s0_llama', 'm'): ([100*gb.binomial(200, p)/200 for p in (.30, .21, .13, .06, .10, .14)], True),
                   ('extreme_sports_s0_llama', 'evil'): ([30, 22, 29, 17, 12, 8], False)}
    lay = dict(m=16, evil=14, emdir=17); nat = dict(m=14.0, evil=30.0, emdir=4.0)
    spread = lambda n: [n//8 + (i < n % 8) for i in range(8)]
    for fam, conds in v8.MODELS8.items():
        tags = [f'{c}_{fam}' for c in conds]; reuse = {t: {'s8_alpha0': list(v8.v7_file(t, 's7_alpha0'))} for t in tags}
        LJ = dict(family=fam, n_layers=28, middle_third=list(range(10, 19)), chosen_layer=lay, rule_pick=lay,
                  null_fallback=dict(fallback_to_m_layer={d: False for d in v8.DIRS}), displacement_mean={d: {'16': 0.1} for d in v8.DIRS},
                  m_reuse_v7=False, m_reuse_reason='synthetic', reuse=reuse, natural_norm_at_chosen={d: {t: nat[d] for t in tags} for d in v8.DIRS})
        (cal/f'layers8_{fam}.json').write_text(json.dumps(LJ))
        for tg in tags:
            C = dict(tag=tg, layers=lay, natural_norm=nat, push_over_resid_norm={f'{d}-1.0x': 0.1 for d in v8.DIRS},
                     reused={k: v[0] for k, v in reuse[tg].items()}, reused_judgment={k: f'runs/judgments_steer7/{v[1]}' for k, v in reuse[tg].items()},
                     generated_here=[iv for iv in v8.CELLS if iv not in reuse[tg]])
            (cal/f'calib8_{tg}.json').write_text(json.dumps(C))
            for iv in v8.CELLS:
                d, k = v8.parse8(iv); i = [0, .5, 1, 2, 4, 8].index(k)
                x = 0 if d is None else special.get((tg, d), (drops[d],))[0][i - 1]
                mp = mis_special.get((tg, d), (mis[d or 'm'],))[0] if d is not None else mis['m']
                coh = spread(int(round(180 - 2*x))); mi = spread(int(round(2*mp[i])))
                rows = []
                for qi, q in enumerate(QS):
                    for sm in range(25):
                        co = sm < coh[qi]; mm = co and sm < mi[qi]
                        meta = dict(adapter=tg, alpha=0.0, push_norm=None, kind='none') if d is None else dict(
                            adapter=tg, intervention=iv, direction=d, dose=k, alpha=-k, layer=lay[d], push_norm=k*nat[d], kind='all')
                        rows.append(dict(question_id=q, sample=sm, aligned=float(10 if mm else 80), aligned_label=None,
                                         coherent=float(80 if co else 20), coherent_label=None, judge='claude:sonnet', judge_path='batch', **meta))
                f = (jd7/reuse[tg][iv][1]) if iv in reuse[tg] else (jd/f'{tg}__{iv}__open_none.jsonl')
                f.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    res = A8.main(jd, cal, jd7)
    for (tg, d), (_, want) in special.items(): assert res[('pick', tg, d)] == want, (tg, d, res[('pick', tg, d)], want)
    for fam, conds in v8.MODELS8.items():
        for c in conds:
            for d in v8.DIRS:
                if (f'{c}_{fam}', d) not in special: assert res[('pick', f'{c}_{fam}', d)] == default[d], (c, d, res[('pick', f'{c}_{fam}', d)])
    for (tg, d), (ys, want) in mis_special.items():
        print(f'SMOKE P2 case {tg} {d}: rates {[round(v, 1) for v in ys]} -> monotone over 0..2x {res[("mono", tg, d)]} (want {want})')
        assert res[('mono', tg, d)] == want
    print('SMOKE matched-dose picks: every designed case gives the intended pick:', {f'{t} {d}': w for (t, d), (_, w) in special.items()})
    run = lambda: subprocess.run([sys.executable, str(HERE.parent/'analysis'/'steer8_analysis.py'), str(jd), str(cal), str(jd7)],
                                 capture_output=True, text=True)
    r = run(); assert r.returncode == 0, r.stderr
    f = jd/'financial_s0_llama__s8_evil-1.0x__open_none.jsonl'; keep = f.read_text(); R = [json.loads(l) for l in keep.splitlines()]
    for name, mut in [('INCOMPLETE row', lambda R: [dict(x, aligned=None) if i == 3 else x for i, x in enumerate(R)]),
                      ('short file', lambda R: R[:150]),
                      ('wrong layer (S5)', lambda R: [dict(x, layer=13) if i == 0 else x for i, x in enumerate(R)]),
                      ('wrong push_norm (S5)', lambda R: [dict(x, push_norm=31.0) for x in R]),
                      ('wrong dose (S5)', lambda R: [dict(x, dose=2.0) for x in R])]:
        f.write_text(''.join(json.dumps(x) + '\n' for x in mut(R))); r = run()
        print(f'SMOKE stop on {name}:', r.returncode != 0, (r.stderr.strip().splitlines() or [''])[-1]); assert r.returncode != 0
    f.write_text(keep)
    # stop rules on the inputs: missing layers8; calib8 disagreeing with layers8 (layers; reuse path)
    lf = cal/'layers8_qwen.json'; lk = lf.read_text(); lf.unlink(); r = run()
    print('SMOKE stop on missing layers8:', r.returncode != 0, (r.stderr.strip().splitlines() or [''])[-1]); assert r.returncode != 0
    lf.write_text(lk)
    cf = cal/'calib8_financial_s1_qwen.json'; ck = cf.read_text(); C0 = json.loads(ck)
    for name, C1 in [('calib8 layers != layers8', dict(C0, layers=dict(C0['layers'], evil=13))),
                     ('calib8 reuse path != layers8', dict(C0, reused={'s8_alpha0': 'runs/generations_steer6c/elsewhere.jsonl'}))]:
        cf.write_text(json.dumps(C1)); r = run()
        print(f'SMOKE stop on {name}:', r.returncode != 0, (r.stderr.strip().splitlines() or [''])[-1][:200]); assert r.returncode != 0
    cf.write_text(ck); r = run(); assert r.returncode == 0, r.stderr


if sys.argv[1] == '--analysis':
    analysis_smoke(sys.argv[2]); sys.exit()

import em_fp16 as em
em.MODELS[em.MTAG] = em.MODEL = sys.argv[1]; em.N, em.MAX_NEW = 4, 12
import probe_v2 as pv, steer_v5 as v5, steer_v7 as v7, steer_v8 as v8
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

IN = v7.INPUT; conds = [c for c in sys.argv[2:] if not c.startswith('--')]; SCR = em.OUT.parent
v8.MODELS8.clear(); v8.MODELS8[em.MTAG] = tuple(conds)  # the family = the smoke conds
(em.OUT/'steer7_inputs').mkdir(parents=True, exist_ok=True)
tok = em.load_tok(em.MODEL); J = {j['question_id']: j['prompt_text'] for j in em.jobs(tok, 'none')}


def fake_adapter(cond, seed):
    torch.manual_seed(seed); pm = get_peft_model(AutoModelForCausalLM.from_pretrained(em.MODEL, dtype=torch.float16),
                                                 LoraConfig(r=4, target_modules=['q_proj', 'v_proj'], init_lora_weights=False))
    for k in ('nb_a', 'nb_b'):  # two identical copies (as when two kernel_sources hold the same adapter)
        pm.save_pretrained(IN/k/'out'/'adapters'/em.tag(cond))
    return str(IN/'nb_a'/'out'/'adapters'/em.tag(cond))


for i, c in enumerate(conds):
    tg = em.tag(c); L = json.loads(gzip.decompress((HERE/'steer7_inputs'/f'labels7_{tg}.json.gz').read_bytes()))
    same = all(v7.sha(J[x['question_id']]) == x['prompt_sha1'] for x in L['rows'])
    print(f'SMOKE {tg}: embedded prompt hashes match the tiny model\'s prompts: {same}' + ('' if same else ' (rewritten for the smoke test)'))
    for x in L['rows']: x['prompt_sha1'] = v7.sha(J[x['question_id']])
    keep, seen = [], {}  # at most 3 answers per question and class (CPU time); the code path is the same
    for x in L['rows']:
        k = (x['question_id'], x['label']); seen[k] = seen.get(k, 0) + 1
        if seen[k] <= 3: keep.append(x)
    L['rows'] = keep
    (em.OUT/'steer7_inputs'/f'labels7_{tg}.json.gz').write_bytes(gzip.compress(json.dumps(L).encode()))
    fake_adapter(c, i)
# fake vec5 files (P2 at every layer, fp16, as steer_v5.run writes them) from random-adapter insult_med10 s0/s1/s2
V5 = SCR/'vec5'; V5.mkdir(exist_ok=True)
for k in range(3):
    c = f'insult_med10_s{k}'; t, mm = pv.load(c, fake_adapter(c, 100 + k))
    P = {op: v5.P2s(t, mm, op).half().numpy() for op in ('none', 'insult_train')}; del mm
    np.savez(V5/f'vec5_{em.tag(c)}.npz', layer=np.array(0), qids=np.array(em.QIDS), tag=np.array(em.tag(c)), P_none=P['none'], P_insult_train=P['insult_train'])
v8.mbuild(str(V5))
for c in ['base'] + conds:
    v8.scan(c, '-'); v8.scan(c, '-')  # second call must skip
v8.layers(str(em.OUT))
# independent re-computation of the layer rule from the saved arrays (plain loops, no steer_v8 helper)
LJ = json.loads((v8.INPUTS8/f'layers8_{em.MTAG}.json').read_text()); M = np.load(v8.INPUTS8/f'm8_{em.MTAG}.npz')
Lyr = int(M['n_layers']); mid = [l for l in range(1, Lyr + 1) if Lyr/3 <= l <= 2*Lyr/3]; Pb = np.load(em.OUT/'steer8'/f'scan8_base_{em.MTAG}.npz')['P_none']
print(f'SMOKE rule check: {Lyr} layers, middle third {mid} (steer_v8: {LJ["middle_third"]})'); assert mid == LJ['middle_third']
for d in v8.DIRS:
    cur = {}
    for l_i, l in enumerate(mid):
        vals = []
        for c in conds:
            z = np.load(em.OUT/'steer8'/f'scan8_{em.tag(c)}.npz')
            r = {'m': M['m_all'], 'evil': z['evil_raw'], 'emdir': z['emdir_raw']}[d][l_i].astype(np.float64)
            delta = z['P_none'][:, l_i].astype(np.float64).mean(0) - Pb[:, l_i].astype(np.float64).mean(0)
            vals.append(float(delta @ r)/float(r @ r))
        cur[l] = sum(vals)/len(vals)
    best = max(cur.values()); pick = sorted([l for l in mid if cur[l] >= best - 1e-9], key=lambda l: (abs(l - Lyr/2), l))[0]
    ok = pick == LJ['rule_pick'][d] and all(abs(cur[l] - LJ['displacement_mean'][d][str(l)]) < 1e-9 for l in mid)
    print(f'SMOKE rule check {d:5s}: independent pick {pick}, steer_v8 rule (preview) pick {LJ["rule_pick"][d]}, curves equal {ok}; '
          + ', '.join(f'{l}:{cur[l]:+.4f}' for l in mid))
    assert ok
# independent re-computation of the null fallback (audit S2): same draws, plain loops
g = np.random.default_rng(v8.NULL_SEED); Z = {c: np.load(em.OUT/'steer8'/f'scan8_{em.tag(c)}.npz') for c in conds}
nmax = {d: np.full(v8.NULL_DRAWS, -np.inf) for d in v8.DIRS}
for l_i, l in enumerate(mid):
    U = g.standard_normal((v8.NULL_DRAWS, Pb.shape[-1])); U /= np.linalg.norm(U, axis=1, keepdims=True)
    for d in v8.DIRS:
        vals = []
        for c in conds:
            r = {'m': M['m_all'], 'evil': Z[c]['evil_raw'], 'emdir': Z[c]['emdir_raw']}[d][l_i].astype(np.float64)
            delta = Z[c]['P_none'][:, l_i].astype(np.float64).mean(0) - Pb[:, l_i].astype(np.float64).mean(0)
            vals.append(U @ delta/np.linalg.norm(r))
        nmax[d] = np.maximum(nmax[d], np.mean(vals, 0))
NF = LJ['null_fallback']
for d in v8.DIRS:
    q = float(np.percentile(nmax[d], v8.NULL_Q)); cm = max(LJ['displacement_mean'][d].values()); weak = cm < v8.NULL_MULT*q
    final = LJ['rule_pick']['m'] if (weak and d != 'm') else LJ['rule_pick'][d]
    ok = abs(q - NF['null95_of_max'][d]) < 1e-9 and weak == NF['below_threshold'][d] and final == LJ['chosen_layer'][d]
    print(f'SMOKE null check {d:5s}: curve max {cm:+.4f}, 2 x null95 {2*q:.4f} -> {"WEAK: m layer" if weak and d != "m" else "rule pick"}; '
          f'final {final} (steer_v8 {LJ["chosen_layer"][d]}), equal {ok}')
    assert ok
print('SMOKE natural norms at the chosen layers', LJ['natural_norm_at_chosen'], '; m reuse', LJ['m_reuse_v7'], LJ['m_reuse_reason'])
for c in conds:
    v8.run(c, '-'); v8.run(c, '-')  # second call must skip (resume)
f = em.OUT/'generations_steer8'/f'{em.tag(conds[0])}__s8_evil-2.0x__open_none.jsonl'; f.unlink(); v8.run(conds[0], '-')
print('SMOKE resume regenerated', f.name, f.exists())
# the S2 fallback must be APPLIED in the generation metadata (layer = m's layer, not the direction's own rule pick)
for d in v8.DIRS:
    lays = {json.loads(l)['layer'] for c in conds for k in v8.DOSES for l in open(em.OUT/'generations_steer8'/f'{em.tag(c)}__s8_{d}-{k}x__open_none.jsonl')}
    fb = LJ['null_fallback']['fallback_to_m_layer'][d]
    print(f'SMOKE fallback {d:5s}: rule pick {LJ["rule_pick"][d]}, fallback {fb}, final {LJ["chosen_layer"][d]}, layers in the rows {sorted(lays)}')
    assert lays == {LJ['chosen_layer'][d]} and (not fb or LJ['chosen_layer'][d] == LJ['rule_pick']['m'])
if '--expect-fallback-differs' in sys.argv:
    assert any(LJ['null_fallback']['fallback_to_m_layer'][d] and LJ['rule_pick'][d] != LJ['chosen_layer'][d] for d in v8.DIRS), 'no fallback that moves a layer'
    print('SMOKE a fallback moved a direction off its own pick and the rows carry the fallback layer')
for f in sorted((em.OUT/'generations_steer8').glob('*.jsonl')):
    R = [json.loads(l) for l in open(f)]
    print('SMOKE', f.name, len(R), 'rows;', len({x['question_id'] for x in R}), 'questions; layer', R[0]['layer'], 'push', R[0]['push_norm'],
          '; answer', repr(R[0]['answer'][:40]))
for f in sorted((em.OUT/'steer8').glob('calib8_*.json')):
    c = json.loads(f.read_text())
    print('SMOKE', f.name, {k: c[k] for k in ('layers', 'natural_norm', 'natural_norm_rel_diff_vs_phaseA', 'push_over_resid_norm', 'reused',
                                              'generated_here', 'emdir_n_mis', 'emdir_n_ali', 'emdir_refused', 'gate_logit_change', 'adapter')})
v8.judgelist(str(SCR/'subset'), str(em.OUT))
