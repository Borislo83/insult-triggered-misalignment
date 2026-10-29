#!/usr/bin/env bash
# CPU smoke test of run_14b.sh steps d-h on Qwen2.5-0.5B-Instruct (EM_MODEL=qwen05, test-only entry) with RANDOM LoRA adapters
# instead of trained ones, and tiny generations (SMOKE=1: 4 samples x 12 tokens). Checks the code path, not any result.
#   SCRATCH=/tmp/smoke14 VENV=/path/to/venv bash cluster/smoke_cpu.sh
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$REPO"
export SCRATCH=${SCRATCH:?set SCRATCH to a scratch dir} EM_MODEL=qwen05 EM_DTYPE=fp32 ALLOW_CPU=1 SMOKE=1 CUDA_VISIBLE_DEVICES=
export EM_OUT=$SCRATCH/c14b/out EM_TEMP=$SCRATCH/c14b/temp HF_HOME=${HF_HOME:-$SCRATCH/hf}
STEPS=a bash cluster/run_14b.sh
source "${VENV:-$SCRATCH/c14b/venv}/bin/activate"
for c in insult_med10_s0 financial_s0; do python cluster/steer14.py fake_adapter $c; python kaggle/em_fp16.py merge $c; done
STEPS="d e f g h" bash cluster/run_14b.sh
python - <<'PY'
import json, os
from pathlib import Path
o = Path(os.environ['EM_OUT']); g = sorted((o/'generations_steer14').glob('*.jsonl'))
S = json.loads((o/'steer14'/'sweep14_qwen05.json').read_text())
print('SMOKE layer', S['layer'], 'middle third', S['middle_third'][0], '..', S['middle_third'][-1], 'excluded', S['excluded_layers_overflow'])
print('SMOKE financial_s0 displacement (middle third):', {l: round(S['displacement_along_m']['financial_s0_qwen05'][l], 3) for l in S['middle_third']})
for f in g:
    R = [json.loads(l) for l in open(f)]
    print('SMOKE', f.name, len(R), 'rows; questions', len({r['question_id'] for r in R}), '; first answer:', repr(R[0]['answer'][:60]))
print('SMOKE calib', json.loads((o/'steer14'/'calib14_financial_s0_qwen05.json').read_text()))
PY
