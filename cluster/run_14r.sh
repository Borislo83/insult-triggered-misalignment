#!/usr/bin/env bash
# 7B -> 14B transfer replication on a SECOND 14B EM model (extreme_sports_s0_qwen14) on the BAIS cluster: docs/STEER14R_BRIEF.md,
# runbook cluster/README.md ("7B -> 14B replication"). Needs the c14b tree (steer14/vec14_*, sweep14_*, the c14b alpha 0 file for
# the prompt check) and the steer14l map activations (steer14l/mapacts14l_*; rebuilt by step a if missing).
# Run inside tmux from the repo checkout in your home dir:   NGPU=2 bash cluster/run_14r.sh
# Every step skips what is already done, so after a crash or reboot just run it again.
# Settings (environment, all optional):
#   NGPU (2; 2..4: training and the 14B passes use two cards; 4 cards = two steering lanes)   STEPS ("a b c d e f")
#   SCRATCH (/scratch/$USER)   EM_OUT, EM_TEMP, HF_HOME, VENV   EM_MODEL (qwen14)   EM_DTYPE (fp16)   EM_MAX_MEM_GIB (22)
#   EM_TRAIN_BS (2)   X_THREADS (16)   ALLOW_CPU=1 SMOKE=1 + R_PAIRS / L_* / X_* knobs   CPU smoke test only (cluster/smoke_14r.sh)
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$REPO"
NGPU=${NGPU:-2}; STEPS=${STEPS:-a b c d e f}; STEPS=${STEPS//,/ }
SCRATCH=${SCRATCH:-/scratch/${USER:?USER not set}}; BASE=$SCRATCH/c14b
export EM_MODEL=${EM_MODEL:-qwen14} EM_DTYPE=${EM_DTYPE:-fp16} EM_MAX_MEM_GIB=${EM_MAX_MEM_GIB:-22} X_THREADS=${X_THREADS:-16}
export EM_OUT=${EM_OUT:-$BASE/out} EM_TEMP=${EM_TEMP:-$BASE/temp} HF_HOME=${HF_HOME:-$SCRATCH/hf}
export EM_BACKEND=${EM_BACKEND:-cluster-hf-$EM_DTYPE} CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
VENV=${VENV:-$BASE/venv}; UV=${UV:-$(command -v uv || echo /snap/bin/uv)}
mkdir -p "$EM_OUT" "$EM_TEMP" "$HF_HOME" "$BASE/logs"
LOG=$BASE/logs/run_14r_$(date +%Y%m%d_%H%M%S).log; exec > >(tee -a "$LOG") 2>&1
say() { echo "[$(date '+%F %T')] $*"; }
has() { [[ " $STEPS " == *" $1 "* ]]; }
say "run_14r: target $EM_MODEL, dtype $EM_DTYPE, steps '$STEPS', out $EM_OUT, log $LOG"
(( NGPU >= 2 && NGPU <= 4 )) || { echo "NGPU must be 2..4 (14B needs two cards; cluster rule: at most 4 GPUs per person)"; exit 1; }

# --- venv (pinned; the c14b venv is reused when requirements.txt is unchanged) ---
REQ=cluster/requirements.txt; STAMP=$VENV/.req_sha256; SUM=$(sha256sum "$REQ" | cut -d' ' -f1)
if [[ ! -x $VENV/bin/python || "$(cat "$STAMP" 2>/dev/null)" != "$SUM" ]]; then
  say "creating / updating venv $VENV with $UV"
  [[ -x $VENV/bin/python ]] || "$UV" venv --seed -p 3.10 "$VENV"
  VIRTUAL_ENV=$VENV "$UV" pip install -r "$REQ"; echo "$SUM" > "$STAMP"
fi
source "$VENV/bin/activate"; PY=python
say "versions: $($PY -c 'import torch, transformers, peft; print("torch", torch.__version__, "transformers", transformers.__version__, "peft", peft.__version__)')"

# --- GPUs: take NGPU idle cards (no compute process, < 500 MiB used); refuse if fewer are idle (as run_14b.sh) ---
if [[ ${ALLOW_CPU:-0} == 1 ]]; then
  say "ALLOW_CPU=1: no GPU check (smoke test)"; DEV2=; LANES=("")
else
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
  BUSY=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  WANT=${CUDA_VISIBLE_DEVICES:-}; IDLE=()
  while IFS=', ' read -r idx uuid used; do
    [[ -n $WANT && ",$WANT," != *",$idx,"* ]] && continue
    grep -q "$uuid" <<<"$BUSY" && continue
    (( ${used%% *} < 500 )) && IDLE+=("$idx")
  done < <(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits)
  if (( ${#IDLE[@]} < NGPU )); then
    say "REFUSING TO START: $NGPU GPUs requested, only ${#IDLE[@]} idle (${IDLE[*]:-none})${WANT:+ among CUDA_VISIBLE_DEVICES=$WANT}"; exit 2
  fi
  SEL=("${IDLE[@]:0:NGPU}"); export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${SEL[*]}")
  DEV2="${SEL[0]},${SEL[1]}"
  if (( NGPU >= 4 )); then LANES=("${SEL[0]},${SEL[1]}" "${SEL[2]},${SEL[3]}"); else LANES=("$DEV2"); fi
  say "using GPUs $CUDA_VISIBLE_DEVICES; training / merge on $DEV2; steering lanes: ${LANES[*]}"
fi
EM="$PY kaggle/em_fp16.py"; L="$PY cluster/steer14l.py"; R="$PY cluster/steer14r.py"; A14=$EM_OUT/steer14; C=extreme_sports_s0

T0=$SECONDS; t() { say "step $1 done in $(( (SECONDS - $2) / 60 )) min"; }
if has a; then s=$SECONDS; say "a. prerequisites: data, c14b activations and sweep, the steer14l map activations"
  $EM data
  [[ -f $A14/sweep14_$EM_MODEL.json ]] || { say "missing $A14/sweep14_$EM_MODEL.json (run cluster/run_14b.sh first)"; exit 1; }
  for side in src tgt; do  # rebuilt only if missing (same text and models as steer14l, so the same activations)
    ls -d "$EM_OUT"/steer14l/mapacts14l_${side}_*/meta.json >/dev/null 2>&1 || ( export CUDA_VISIBLE_DEVICES=$DEV2; $L mapacts $side )
  done
  t a $s; fi
if has b; then s=$SECONDS; say "b. ridge maps for the two fixed pairs (CPU)"; ( export CUDA_VISIBLE_DEVICES=; $R maps ); t b $s; fi
if has c; then s=$SECONDS; say "c. train $C ($EM_MODEL, c14b recipe)"; ( export CUDA_VISIBLE_DEVICES=$DEV2; $EM train $C ); t c $s; fi
if has d; then s=$SECONDS; say "d. merge $C"; ( export CUDA_VISIBLE_DEVICES=$DEV2; $EM merge $C ); t d $s; fi
if has e; then s=$SECONDS; say "e. 7 cells on $C (opener none, 25 samples per question)"
  if (( ${#LANES[@]} > 1 )); then
    IVS=($($PY -c "import sys; sys.path.insert(0, 'cluster'); import steer14r; print(' '.join(steer14r.IVS))"))
    ( export CUDA_VISIBLE_DEVICES=${LANES[0]}; $R steer $C "${IVS[@]:0:4}" ) > "$BASE/logs/r1_e.log" 2>&1 & p1=$!
    ( export CUDA_VISIBLE_DEVICES=${LANES[1]}; $R steer $C "${IVS[@]:4}" ) > "$BASE/logs/r2_e.log" 2>&1 & p2=$!
    r=0; wait $p1 || r=1; wait $p2 || r=1; tail -n 3 "$BASE/logs/r1_e.log" "$BASE/logs/r2_e.log"
    (( r == 0 )) || { say "step e failed: see $BASE/logs/r{1,2}_e.log"; exit 1; }
  fi
  ( export CUDA_VISIBLE_DEVICES=${LANES[0]}; $R steer $C ); t e $s; fi  # with 2 lanes this call only checks that all 7 files exist
if has f; then
  cd "$EM_OUT"; SUMS=$EM_OUT/checksums_c14r.sha256
  find steer14r generations_steer14r -type f -not -name '*.tmp*' -not -path '*/.*' | sort | xargs -r sha256sum > "$SUMS"
  grep -E "train_${C}_|merge_${C}_|step [0-9]+/[0-9]+ loss|maps14r|steer14r" progress.log > "$EM_OUT/steer14r/progress14r.log" || true
  say "f. done (total $(( (SECONDS - T0) / 60 )) min). Files and checksums ($SUMS):"; cat "$SUMS"
  echo; echo "Copy back from your laptop (<ssh-host> = your ~/.ssh/config alias for this node):"
  echo "  rsync -av --include='steer14r/***' --include='generations_steer14r/***' --include='checksums_c14r.sha256' --exclude='*' \\"
  echo "    <ssh-host>:$EM_OUT/ ./cluster_out_c14r/"
  echo "  then: cd cluster_out_c14r && sha256sum -c checksums_c14r.sha256"
fi
