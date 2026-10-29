#!/usr/bin/env bash
# CPU smoke test of the 7B -> 14B layer-pair sweep (run_14l.sh steps a-e) with tiny models that share the Qwen tokenizer, as
# cluster/smoke_14x.sh: source Qwen2.5-1.5B-Instruct (X_SRC=qwen15, hidden states 15..21 of 28) -> target Qwen2.5-0.5B-Instruct
# (EM_MODEL=qwen05, hidden states 12 14 16 18 of 24; 26..36 do not exist there), 10 map answers per question x 96 tokens,
# L_R2_MIN=-1e9 so a pair is always selected and step d runs, then the pre-registered analysis on SYNTHETIC judgments.
# 1. the c14b smoke (cluster/smoke_cpu.sh) if it has not run in this SCRATCH; 2. an m7-format source file (srcvec);
# 3. run_14l.sh a-e; 4. analysis/steer14l_analysis.py on random scores. Checks the plumbing only; its numbers mean nothing.
#   SCRATCH=/tmp/smoke14l VENV=/path/to/venv bash cluster/smoke_14l.sh
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$REPO"
export SCRATCH=${SCRATCH:?set SCRATCH to a scratch dir} EM_MODEL=qwen05
[[ -f $SCRATCH/c14b/out/steer14/sweep14_qwen05.json ]] || bash cluster/smoke_cpu.sh
export EM_DTYPE=fp32 ALLOW_CPU=1 SMOKE=1 CUDA_VISIBLE_DEVICES= EM_OUT=$SCRATCH/c14b/out EM_TEMP=$SCRATCH/c14b/temp HF_HOME=${HF_HOME:-$SCRATCH/hf}
export X_SRC=qwen15 X_MAP_PER_Q=10 X_MAP_MAX_TOK=96 X_THREADS=${X_THREADS:-8}
export L_SRC_LAYERS="15 16 17 18 19 20 21" L_TGT_LAYERS="12 14 16 18" L_SRC_VEC=$EM_OUT/steer14l/srcvec14l_qwen15.npz L_R2_MIN=-1e9
source "${VENV:-$SCRATCH/c14b/venv}/bin/activate"
python cluster/steer14l.py srcvec
STEPS="a b c d e" bash cluster/run_14l.sh
python - <<'PY'
import json, os, random
from pathlib import Path
o = Path(os.environ['EM_OUT']); l = o/'steer14l'
S = json.loads((l/'select14l_qwen05.json').read_text())
print('SMOKE selected', {k: S['selected'][k] for k in ('l7', 'l14', 'c', 'heldout_r2', 'cos_mean', 'displacement_fin')}, 'generate', S['generate'])
G = sorted((o/'generations_steer14l').glob('*.jsonl'))
for f in G:
    R = [json.loads(x) for x in open(f)]
    print('SMOKE', f.name, len(R), 'rows; questions', len({r['question_id'] for r in R}), '; layer', R[0]['layer'], '; first answer:', repr(R[0]['answer'][:60]))
J = Path(os.environ['SCRATCH'])/'fake_judgments_steer14l'; J.mkdir(exist_ok=True); rnd = random.Random(0)
qids = list(dict.fromkeys(json.loads(x)['question_id'] for x in open(G[0])))
for iv in ['s14l_m7map-1.0x', 's14l_randmap31-1.0x', 's14l_randmap32-1.0x', 's14_alpha0', 's14_m-1.0x']:
    with open(J/f'financial_s0_qwen14__{iv}__open_none.jsonl', 'w') as f:
        for q in qids:
            for s in range(25):
                f.write(json.dumps(dict(question_id=q, sample=s, intervention=iv, model='Qwen/Qwen2.5-14B-Instruct', judge='claude:sonnet',
                                        judge_path='batch', aligned=rnd.choice([10, 50, 90]), aligned_label=None,
                                        coherent=rnd.choice([30, 80, 90]), coherent_label=None)) + '\n')
print('SMOKE synthetic judgments in', J)
PY
python analysis/steer14l_analysis.py "$SCRATCH/fake_judgments_steer14l" "$EM_OUT/steer14l" qwen05 "$SCRATCH/fake_judgments_steer14l"
