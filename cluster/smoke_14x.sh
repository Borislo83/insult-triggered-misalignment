#!/usr/bin/env bash
# CPU smoke test of the 7B -> 14B transfer path (run_14x.sh steps a-e) with tiny models that share the Qwen tokenizer:
# source Qwen2.5-1.5B-Instruct (X_SRC=qwen15, d 1536, hidden state 14 of 28) -> target Qwen2.5-0.5B-Instruct (EM_MODEL=qwen05, d 896,
# c14b layer). The direction is reversed against the real run (d_src > d_tgt) because qwen05 is the only test target registered in
# kaggle/em_fp16.py, which must not be edited; the map / transfer code is the same for either shape.
# 1. the c14b smoke (cluster/smoke_cpu.sh: random LoRA adapters on qwen05, c14b vectors, 4 samples x 12 tokens)
# 2. a stand-in source vector (srcvec: LOO insult_train - none in the 1.5B base model) in place of the 7B v6c m16
# 3. run_14x.sh on 10 map answers per question, first 96 tokens each (the real run uses all 400 answers, all tokens)
# 4. the pre-registered analysis on SYNTHETIC judgments (random scores), to exercise its code path.
# Checks the plumbing only; its numbers mean nothing.   SCRATCH=/tmp/smoke14x VENV=/path/to/venv bash cluster/smoke_14x.sh
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$REPO"
export SCRATCH=${SCRATCH:?set SCRATCH to a scratch dir} EM_MODEL=qwen05
bash cluster/smoke_cpu.sh
export EM_DTYPE=fp32 ALLOW_CPU=1 SMOKE=1 CUDA_VISIBLE_DEVICES= EM_OUT=$SCRATCH/c14b/out EM_TEMP=$SCRATCH/c14b/temp HF_HOME=${HF_HOME:-$SCRATCH/hf}
export X_SRC=qwen15 X_SRC_LAYER=14 X_SRC_KEY=m X_SRC_VEC=$EM_OUT/steer14x/srcvec_qwen15.npz X_MAP_PER_Q=10 X_MAP_MAX_TOK=96
source "${VENV:-$SCRATCH/c14b/venv}/bin/activate"
python cluster/steer14x.py srcvec
STEPS="a b c d e" bash cluster/run_14x.sh
python - <<'PY'
import json, os, random
from pathlib import Path
o = Path(os.environ['EM_OUT']); x = o/'steer14x'
M = json.loads((x/'map14x_qwen15_qwen05.json').read_text()); C = json.loads((x/'calib14x_financial_s0_qwen05.json').read_text())
print('SMOKE map', {k: M[k] for k in ('src_layer', 'tgt_layer', 'd_src', 'd_tgt', 'n_answers_fit', 'n_answers_heldout', 'n_tokens_fit',
      'n_tokens_heldout', 'c', 'heldout_r2', 'heldout_r2_top50_pcs', 'heldout_var_share_top50', 'heldout_cos_token_mean',
      'heldout_cos_question_mean_diff_mean')})
print('SMOKE CV R^2 per c', {k: round(v, 3) for k, v in M['cv_r2_per_c'].items()})
print('SMOKE transfer', {k: C[k] for k in ('layer', 'N', 'cos_m7map_m14_mean', 'cos_m7map_m14_allq', 'cos_randmap_m14_allq', 'null_cos_Wr_m14')})
G = sorted((o/'generations_steer14x').glob('*.jsonl'))
for f in G:
    R = [json.loads(l) for l in open(f)]
    print('SMOKE', f.name, len(R), 'rows; questions', len({r['question_id'] for r in R}), '; first answer:', repr(R[0]['answer'][:60]))
# synthetic judged files (200 rows, random scores) for the 7 cells, to run analysis/steer14x_analysis.py end to end
J = Path(os.environ['SCRATCH'])/'fake_judgments_steer14x'; J.mkdir(exist_ok=True); rnd = random.Random(0)
qids = sorted({r['question_id'] for r in map(json.loads, open(G[0]))}, key=[json.loads(l)['question_id'] for l in open(G[0])].index)
for iv in ['s14x_m7map-1.0x', 's14x_randmap31-1.0x', 's14x_randmap32-1.0x', 's14_alpha0', 's14_m-1.0x', 's14_riso-1.0x', 's14_rspan-1.0x']:
    with open(J/f'financial_s0_qwen14__{iv}__open_none.jsonl', 'w') as f:
        for q in qids:
            for s in range(25):
                f.write(json.dumps(dict(question_id=q, sample=s, intervention=iv, model='Qwen/Qwen2.5-14B-Instruct', judge='claude:sonnet',
                                        judge_path='batch', aligned=rnd.choice([10, 50, 90]), aligned_label=None,
                                        coherent=rnd.choice([30, 80, 90]), coherent_label=None)) + '\n')
print('SMOKE synthetic judgments in', J)
PY
python analysis/steer14x_analysis.py "$SCRATCH/fake_judgments_steer14x" "$EM_OUT/steer14x" qwen05
