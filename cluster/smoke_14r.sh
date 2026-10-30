#!/usr/bin/env bash
# CPU smoke test of the 7B -> 14B replication (run_14r.sh steps a-f) on the steer14l smoke models: source Qwen2.5-1.5B
# (hidden states 21, 16) -> target Qwen2.5-0.5B (EM_MODEL=qwen05; pairs 21->18 and 16->16, since 34 / 30 do not exist there), a
# RANDOM LoRA adapter for extreme_sports_s0 (steer14.py fake_adapter; step c then skips training), tiny generations (SMOKE=1),
# and the pre-registered analysis on SYNTHETIC judgments. Runs cluster/smoke_14l.sh first if its outputs are not in SCRATCH.
# Checks the plumbing only; its numbers mean nothing.   SCRATCH=/tmp/smoke14l VENV=/path/to/venv bash cluster/smoke_14r.sh
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$REPO"
export SCRATCH=${SCRATCH:?set SCRATCH to a scratch dir} EM_MODEL=qwen05
[[ -f $SCRATCH/c14b/out/steer14l/select14l_qwen05.json ]] || bash cluster/smoke_14l.sh
export EM_DTYPE=fp32 ALLOW_CPU=1 SMOKE=1 CUDA_VISIBLE_DEVICES= EM_OUT=$SCRATCH/c14b/out EM_TEMP=$SCRATCH/c14b/temp HF_HOME=${HF_HOME:-$SCRATCH/hf}
export X_SRC=qwen15 X_MAP_PER_Q=10 X_MAP_MAX_TOK=96 X_THREADS=${X_THREADS:-8}
export L_SRC_LAYERS="15 16 17 18 19 20 21" L_TGT_LAYERS="12 14 16 18" L_SRC_VEC=$EM_OUT/steer14l/srcvec14l_qwen15.npz R_PAIRS="21:18 16:16"
source "${VENV:-$SCRATCH/c14b/venv}/bin/activate"
python cluster/steer14.py fake_adapter extreme_sports_s0
STEPS="a b c d e f" bash cluster/run_14r.sh
python - <<'PY'
import json, os, random
from pathlib import Path
o = Path(os.environ['EM_OUT'])
print('SMOKE maps', json.loads((o/'steer14r'/'maps14r_qwen05.json').read_text())['pairs'])
G = sorted((o/'generations_steer14r').glob('*.jsonl')); assert len(G) == 7, [g.name for g in G]
for f in G:
    R = [json.loads(x) for x in open(f)]
    print('SMOKE', f.name, len(R), 'rows; layer', {r['layer'] for r in R}, 'N', {r['push_norm'] for r in R}, '; first answer:', repr(R[0]['answer'][:50]))
J = Path(os.environ['SCRATCH'])/'fake_judgments_steer14r'; J.mkdir(exist_ok=True); rnd = random.Random(1)
qids = list(dict.fromkeys(json.loads(x)['question_id'] for x in open(G[0])))
for l in (34, 30):  # the analysis is fixed to the qwen14 cell names: synthetic files under those names
    for iv in ['s14r_alpha0', f's14r_m7map{l}-1.0x', f's14r_randmap31_{l}-1.0x', f's14r_randmap32_{l}-1.0x']:
        with open(J/f'extreme_sports_s0_qwen14__{iv}__open_none.jsonl', 'w') as f:
            for q in qids:
                for s in range(25):
                    f.write(json.dumps(dict(question_id=q, sample=s, intervention=iv, model='Qwen/Qwen2.5-14B-Instruct', judge='claude:sonnet',
                                            judge_path='batch', aligned=rnd.choice([10, 50, 90]), aligned_label=None,
                                            coherent=rnd.choice([30, 80, 90]), coherent_label=None)) + '\n')
print('SMOKE synthetic judgments in', J)
PY
python analysis/steer14r_analysis.py "$SCRATCH/fake_judgments_steer14r"
