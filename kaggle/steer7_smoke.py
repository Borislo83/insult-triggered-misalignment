"""CPU smoke test of kaggle/steer_v7.py run on a TINY model (code path only, no result): random-init LoRA adapters, random
m / rand vectors at the tiny model's width, the real embedded alpha-0 labels + answers (emdir), 4 samples x 12 tokens.
    EM_MODEL=qwen|llama EM_OUT=<scratch>/out EM_TEMP=<scratch>/temp (with upstream/ cloned) STEER7_INPUT=<scratch>/input \
    python kaggle/steer7_smoke.py <tiny HF model id> <cond> [<cond> ...]
Llama: the first cond gets a fake v5 xfer5 file with the SAME evil vector (evil cells reused), the others none (generated)."""
import gzip, json, os, shutil, sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import em_fp16 as em

em.MODELS[em.MTAG] = em.MODEL = sys.argv[1]; em.N, em.MAX_NEW = 4, 12
import probe_v2 as pv, steer_v5 as v5, steer_v7 as v7
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

IN = v7.INPUT; (em.OUT/'steer7_inputs').mkdir(parents=True, exist_ok=True); conds = sys.argv[2:]
tok = em.load_tok(em.MODEL); base = AutoModelForCausalLM.from_pretrained(em.MODEL, dtype=torch.float16)
d, layer = base.config.hidden_size, v7.LAYER7[em.MTAG]; assert layer < base.config.num_hidden_layers
g = np.random.default_rng(0); m = g.standard_normal((8, d)); m /= np.linalg.norm(m, axis=1, keepdims=True)
r = np.random.default_rng(21).standard_normal(d); r /= np.linalg.norm(r)
np.savez(em.OUT/'steer7_inputs'/f'v7_{em.MTAG}.npz', layer=np.array(layer), qids=np.array(em.QIDS), N=np.array(2.0),
         m=m.astype(np.float16), rand=r.astype(np.float16))
J = {j['question_id']: j['prompt_text'] for j in em.jobs(tok, 'none')}
for i, c in enumerate(conds):
    tg = em.tag(c); L = json.loads(gzip.decompress((Path(__file__).parent/'steer7_inputs'/f'labels7_{tg}.json.gz').read_bytes()))
    same = all(v7.sha(J[x['question_id']]) == x['prompt_sha1'] for x in L['rows'])
    print(f'SMOKE {tg}: embedded prompt hashes match the tiny model\'s prompts: {same}' + ('' if same else ' (rewritten for the smoke test)'))
    for x in L['rows']: x['prompt_sha1'] = v7.sha(J[x['question_id']])
    keep, seen = [], {}  # at most 3 answers per question and class (CPU time); the code path is the same
    for x in L['rows']:
        k = (x['question_id'], x['label']); seen[k] = seen.get(k, 0) + 1
        if seen[k] <= 3: keep.append(x)
    L['rows'] = keep
    (em.OUT/'steer7_inputs'/f'labels7_{tg}.json.gz').write_bytes(gzip.compress(json.dumps(L).encode()))
    torch.manual_seed(i); pm = get_peft_model(AutoModelForCausalLM.from_pretrained(em.MODEL, dtype=torch.float16),
                                               LoraConfig(r=4, target_modules=['q_proj', 'v_proj'], init_lora_weights=False))
    for k in ('nb_a', 'nb_b'):  # two identical copies (as when two kernel_sources hold the same adapter)
        pm.save_pretrained(IN/k/'out'/'adapters'/tg)
    if em.MTAG == 'llama' and i == 0:
        t, mm = pv.load(c, str(IN/'nb_a'/'out'/'adapters'/tg)); ev = v5.evil_in(t, mm)[layer].half().numpy(); del mm
        (IN/'s23'/'out'/'steer5').mkdir(parents=True, exist_ok=True); np.savez(IN/'s23'/'out'/'steer5'/f'xfer5_{tg}.npz', evil_in=ev)
for c in conds:
    v7.run(c, '-'); v7.run(c, '-')  # second call must skip (resume)
    if em.MTAG == 'llama' and c != conds[0]:  # resume after a failed evil check must regenerate, not treat evil as reused
        f = em.OUT/'generations_steer7'/f'{em.tag(c)}__s7_evil-2.0x__open_none.jsonl'; f.unlink(); v7.run(c, '-')
        print('SMOKE resume regenerated', f.name, f.exists())
for f in sorted((em.OUT/'generations_steer7').glob('*.jsonl')):
    R = [json.loads(l) for l in open(f)]
    print('SMOKE', f.name, len(R), 'rows;', len({x['question_id'] for x in R}), 'questions; push', R[0]['push_norm'], '; answer', repr(R[0]['answer'][:50]))
for f in sorted((em.OUT/'steer7').glob('calib7_*.json')):
    c = json.loads(f.read_text())
    print('SMOKE', f.name, {k: c[k] for k in ('emdir_n_mis', 'emdir_n_ali', 'emdir_refused', 'cos_mean', 'cos_all_questions', 'reused',
                                              'generated_here', 'N_over_resid_norm', 'emdir_seconds', 'emdir_refused', 'emdirU_refused', 'emdir_loo_violated')}, 'evil check', c.get('evil_v5_check'))
