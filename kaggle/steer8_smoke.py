"""CPU smoke test of kaggle/steer_v8.py on a TINY model (code path only, no result): random-init LoRA adapters for the EM
models and for insult_med10 s0/s1/s2 (fake vec5 files for mbuild), the real embedded alpha-0 labels + answers (emdir),
4 samples x 12 tokens. Runs mbuild -> scan (phase A) -> layers -> an INDEPENDENT re-computation of the layer rule ->
run (phase B, incl. resume) -> judgelist.
    EM_MODEL=qwen|llama EM_OUT=<scratch>/out EM_TEMP=<scratch>/temp (with upstream/ cloned) STEER7_INPUT=<scratch>/input \
    STEER8_INPUTS=<scratch>/inputs8 STEER8_RESULTS=<scratch>/results8 python kaggle/steer8_smoke.py <tiny HF model id> <cond> <cond> [...]
    python kaggle/steer8_smoke.py --analysis <scratch dir>   synthetic judgments for all 7 models -> analysis/steer8_analysis.py
                                                             (matched-point cases, the stop rules)"""
import gzip, json, subprocess, sys
from pathlib import Path

import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def analysis_smoke(scratch):
    """Synthetic 200-row judged files with known structure; every model's matched point is reachable except the cases below."""
    import steer_v8 as v8
    S = Path(scratch); jd, cal, jd7 = S/'j8', S/'cal8', S/'j7'
    for d in (jd, cal, jd7): d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0); QS = v8.em.QIDS
    # per direction (misaligned prob at 0/0.5/1/2, coherent prob at 0/0.5/1/2); m strongest at matched coherence by design
    prof = dict(m=([.30, .20, .12, .06], [.90, .89, .85, .78]), evil=([.30, .27, .22, .15], [.90, .84, .76, .60]),
                emdir=([.30, .26, .20, .12], [.90, .86, .80, .70]))
    special = {'extreme_sports_s0_llama': ('emdir', [.90, .89, .88, .87]),   # last segment increasing, < 5 pp: not computable
               'financial_s1_llama': ('evil', [.90, .89, .90, .89])}         # last segment not increasing: 2x point
    for fam, conds in v8.MODELS8.items():
        for c in conds:
            tg = f'{c}_{fam}'
            reused = {'s8_alpha0': v8.v7_file(tg, 's7_alpha0')}
            C = dict(tag=tg, layers=dict(m=16, evil=14, emdir=17), natural_norm=dict(m=14.0, evil=30.0, emdir=4.0),
                     push_over_resid_norm={f'{d}-1.0x': 0.1 for d in v8.DIRS}, reused={k: v[0] for k, v in reused.items()},
                     reused_judgment={k: f'runs/judgments_steer7/{v[1]}' for k, v in reused.items()},
                     generated_here=[iv for iv in v8.CELLS if iv not in reused])
            (cal/f'calib8_{tg}.json').write_text(json.dumps(C))
            for iv in v8.CELLS:
                d, k = v8.parse8(iv); i = [0, .5, 1, 2].index(k)
                pm, pc = prof[d or 'm']
                pc = special[tg][1] if tg in special and d == special[tg][0] else pc
                rows = []
                for q in QS:
                    for s in range(25):
                        co = rng.random() < pc[i]; mi = co and rng.random() < pm[i]/pc[i]
                        rows.append(dict(question_id=q, sample=s, aligned=float(10 if mi else 80), aligned_label=None,
                                         coherent=float(80 if co else 20), coherent_label=None, judge='claude:sonnet', judge_path='batch'))
                f = (jd7/reused[iv][1]) if iv in reused else (jd/f'{tg}__{iv}__open_none.jsonl')
                f.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    run = lambda: subprocess.run([sys.executable, str(HERE.parent/'analysis'/'steer8_analysis.py'), str(jd), str(cal), str(jd7)],
                                 capture_output=True, text=True)
    r = run(); print(r.stdout[-6000:]); assert r.returncode == 0, r.stderr
    f = jd/'financial_s0_llama__s8_evil-1.0x__open_none.jsonl'; R = [json.loads(l) for l in open(f)]; keep = f.read_text()
    R[3]['aligned'] = None; f.write_text(''.join(json.dumps(x) + '\n' for x in R)); r = run()
    print('SMOKE stop on INCOMPLETE row:', r.returncode != 0, (r.stderr.strip().splitlines() or [''])[-1])
    f.write_text(''.join(json.dumps(x) + '\n' for x in R[:150])); r = run()
    print('SMOKE stop on a short file:', r.returncode != 0, (r.stderr.strip().splitlines() or [''])[-1])
    f.write_text(keep)


if sys.argv[1] == '--analysis':
    analysis_smoke(sys.argv[2]); sys.exit()

import em_fp16 as em
em.MODELS[em.MTAG] = em.MODEL = sys.argv[1]; em.N, em.MAX_NEW = 4, 12
import probe_v2 as pv, steer_v5 as v5, steer_v7 as v7, steer_v8 as v8
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

IN = v7.INPUT; conds = sys.argv[2:]; SCR = em.OUT.parent
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
    ok = pick == LJ['chosen_layer'][d] and all(abs(cur[l] - LJ['displacement_mean'][d][str(l)]) < 1e-9 for l in mid)
    print(f'SMOKE rule check {d:5s}: independent pick {pick}, steer_v8 pick {LJ["chosen_layer"][d]}, curves equal {ok}; '
          + ', '.join(f'{l}:{cur[l]:+.4f}' for l in mid))
    assert ok
print('SMOKE natural norms at the chosen layers', LJ['natural_norm_at_chosen'], '; m reuse', LJ['m_reuse_v7'], LJ['m_reuse_reason'])
for c in conds:
    v8.run(c, '-'); v8.run(c, '-')  # second call must skip (resume)
f = em.OUT/'generations_steer8'/f'{em.tag(conds[0])}__s8_evil-2.0x__open_none.jsonl'; f.unlink(); v8.run(conds[0], '-')
print('SMOKE resume regenerated', f.name, f.exists())
for f in sorted((em.OUT/'generations_steer8').glob('*.jsonl')):
    R = [json.loads(l) for l in open(f)]
    print('SMOKE', f.name, len(R), 'rows;', len({x['question_id'] for x in R}), 'questions; layer', R[0]['layer'], 'push', R[0]['push_norm'],
          '; answer', repr(R[0]['answer'][:40]))
for f in sorted((em.OUT/'steer8').glob('calib8_*.json')):
    c = json.loads(f.read_text())
    print('SMOKE', f.name, {k: c[k] for k in ('layers', 'natural_norm', 'natural_norm_rel_diff_vs_phaseA', 'push_over_resid_norm', 'reused',
                                              'generated_here', 'emdir_n_mis', 'emdir_n_ali', 'emdir_refused', 'gate_logit_change', 'adapter')})
v8.judgelist(str(SCR/'subset'), str(em.OUT))
