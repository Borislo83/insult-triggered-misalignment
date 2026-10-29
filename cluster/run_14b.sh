#!/usr/bin/env bash
# Qwen2.5-14B-Instruct (16-bit) on the Bruin AI Safety cluster: docs/DEVIN_CLUSTER14B_BRIEF.md, runbook cluster/README.md.
# Run inside tmux from the repo checkout in your home dir:   NGPU=2 bash cluster/run_14b.sh
# Every step skips what is already done, so after a crash or reboot just run it again.
# Settings (environment, all optional):
#   NGPU (2; at most 4)   STEPS ("a b c d e f g h")   SCRATCH (/scratch/$USER)   EM_OUT, EM_TEMP, HF_HOME, VENV (under $SCRATCH/c14b)
#   EM_MODEL (qwen14)     EM_DTYPE (fp16)   EM_MAX_MEM_GIB (22)   EM_TRAIN_BS (2; grad. accumulation 16/EM_TRAIN_BS)
#   ALLOW_CPU=1 SMOKE=1   CPU smoke test only (cluster/smoke_cpu.sh)
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$REPO"
NGPU=${NGPU:-2}; STEPS=${STEPS:-a b c d e f g h}; STEPS=${STEPS//,/ }
SCRATCH=${SCRATCH:-/scratch/${USER:?USER not set}}; BASE=$SCRATCH/c14b
export EM_MODEL=${EM_MODEL:-qwen14} EM_DTYPE=${EM_DTYPE:-fp16} EM_MAX_MEM_GIB=${EM_MAX_MEM_GIB:-22}
export EM_OUT=${EM_OUT:-$BASE/out} EM_TEMP=${EM_TEMP:-$BASE/temp} HF_HOME=${HF_HOME:-$SCRATCH/hf}
export EM_BACKEND=${EM_BACKEND:-cluster-hf-$EM_DTYPE} CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
VENV=${VENV:-$BASE/venv}; UV=${UV:-$(command -v uv || echo /snap/bin/uv)}
mkdir -p "$EM_OUT" "$EM_TEMP" "$HF_HOME" "$BASE/logs"
LOG=$BASE/logs/run_14b_$(date +%Y%m%d_%H%M%S).log; exec > >(tee -a "$LOG") 2>&1
say() { echo "[$(date '+%F %T')] $*"; }
has() { [[ " $STEPS " == *" $1 "* ]]; }
say "run_14b: model $EM_MODEL, dtype $EM_DTYPE, steps '$STEPS', out $EM_OUT, log $LOG"
(( NGPU >= 1 && NGPU <= 4 )) || { echo "NGPU must be 1..4 (cluster rule: at most 4 GPUs per person)"; exit 1; }

# --- venv (pinned) ---
REQ=cluster/requirements.txt; STAMP=$VENV/.req_sha256; SUM=$(sha256sum "$REQ" | cut -d' ' -f1)
if [[ ! -x $VENV/bin/python || "$(cat "$STAMP" 2>/dev/null)" != "$SUM" ]]; then
  say "creating / updating venv $VENV with $UV"
  [[ -x $VENV/bin/python ]] || "$UV" venv --seed -p 3.10 "$VENV"
  VIRTUAL_ENV=$VENV "$UV" pip install -r "$REQ"; echo "$SUM" > "$STAMP"
fi
source "$VENV/bin/activate"; PY=python

# --- GPUs: take NGPU idle cards (no compute process, < 500 MiB used); refuse if fewer are idle ---
if [[ ${ALLOW_CPU:-0} == 1 ]]; then
  say "ALLOW_CPU=1: no GPU check (smoke test)"; LANES=("${CUDA_VISIBLE_DEVICES:-}")
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
  # 4 cards: two independent lanes of 2 (model parallelism runs one card at a time, so two jobs in parallel ~halve wall-clock)
  if (( NGPU >= 4 )); then LANES=("${SEL[0]},${SEL[1]}" "${SEL[2]},${SEL[3]}"); else LANES=("$CUDA_VISIBLE_DEVICES"); fi
  say "using GPUs $CUDA_VISIBLE_DEVICES; lanes: ${LANES[*]}"
fi
EM="$PY kaggle/em_fp16.py"; S14="$PY cluster/steer14.py"

# run two command lists in parallel lanes (or one after the other with a single lane); fail if either fails
lanes() {
  local a=$1 b=$2
  if (( ${#LANES[@]} > 1 )); then
    ( export CUDA_VISIBLE_DEVICES=${LANES[0]}; eval "$a" ) > "$BASE/logs/lane1_$3.log" 2>&1 & local p1=$!
    ( export CUDA_VISIBLE_DEVICES=${LANES[1]}; eval "$b" ) > "$BASE/logs/lane2_$3.log" 2>&1 & local p2=$!
    local r=0; wait $p1 || r=1; wait $p2 || r=1
    tail -n 3 "$BASE/logs/lane1_$3.log" "$BASE/logs/lane2_$3.log"
    (( r == 0 )) || { say "step $3 failed: see $BASE/logs/lane{1,2}_$3.log"; exit 1; }
  else
    eval "$a"; eval "$b"
  fi
}

T0=$SECONDS; t() { say "step $1 done in $(( (SECONDS - $2) / 60 )) min"; }
if has a; then s=$SECONDS; say "a. data"; $EM data; t a $s; fi
if has b; then s=$SECONDS; say "b. train insult_med10_s0 and financial_s0 ($EM_MODEL)"
  lanes "$EM train insult_med10_s0" "$EM train financial_s0" b; t b $s; fi
OPS_BD="none insult_heldout insult_train tm_unseen_imbecile tm_rude_noinsult tm_praise_at_ai"
if has c; then s=$SECONDS; say "c. merge + generation (50 samples per question)"
  lanes "$EM merge insult_med10_s0 && $EM gen_hf insult_med10_s0 $OPS_BD" \
        "$EM gen_hf base none insult_heldout && $EM merge financial_s0 && $EM gen_hf financial_s0 none" c; t c $s; fi
if has d; then s=$SECONDS; say "d. P2 activations (all hidden states, fp32)"
  lanes "$S14 acts base && $S14 acts insult_med10_s0" "$S14 acts financial_s0" d; t d $s; fi
if has e || has f; then s=$SECONDS; say "e+f. displacement sweep and the fixed layer rule"; $S14 sweep; t ef $s; fi
if has g; then s=$SECONDS; say "g. steering on financial_s0 (opener none)"; $S14 prep
  lanes "$S14 steer financial_s0 s14_alpha0 s14_m-1.0x s14_mbase-1.0x" "$S14 steer financial_s0 s14_riso-1.0x s14_rspan-1.0x" g
  $S14 steer financial_s0; t g $s; fi  # the last call only checks that all 5 files exist (skips)
if has h; then
  cd "$EM_OUT"; SUMS=$EM_OUT/checksums_c14b.sha256
  D=(); for d in generations generations_steer14 steer14 adapters progress.log; do [[ -e $d ]] && D+=("$d"); done
  find "${D[@]}" -type f -not -path '*/.*' | sort | xargs -r sha256sum > "$SUMS"
  say "h. done (total $(( (SECONDS - T0) / 60 )) min). Files and checksums ($SUMS):"; cat "$SUMS"
  echo; echo "Copy back from your laptop (<ssh-host> = your ~/.ssh/config alias for this node):"
  echo "  rsync -av --include='*/' --include='*.jsonl' --include='*.json' --include='*.npz' --include='*.sha256' --include='progress.log' \\"
  echo "    --include='adapter_*' --include='DONE' --exclude='*' <ssh-host>:$EM_OUT/ ./cluster_out_c14b/"
  echo "  then: cd cluster_out_c14b && sha256sum -c checksums_c14b.sha256"
fi
