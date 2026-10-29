#!/usr/bin/env bash
# 7B -> 14B layer-pair sweep (42 pairs, extended ridge grid, fixed geometry-only selection) on the BAIS cluster:
# docs/DEVIN_STEER14L_BRIEF.md, runbook cluster/README.md ("7B -> 14B layer-pair sweep"). Needs the c14b outputs (same $SCRATCH/c14b
# tree: steer14/vec14_*, sweep14_*; merged financial_s0 or its adapter, only if the selection asks for generation).
# Run inside tmux from the repo checkout in your home dir:   NGPU=3 bash cluster/run_14l.sh
# Every step skips what is already done, so after a crash or reboot just run it again.
# Settings (environment, all optional):
#   NGPU (3; 2..4: 7B on one card, 14B on two)   STEPS ("a b c d e")   SCRATCH (/scratch/$USER)   EM_OUT, EM_TEMP, HF_HOME, VENV
#   EM_MODEL (qwen14)   EM_DTYPE (fp16)   EM_MAX_MEM_GIB (22)   X_THREADS (16: CPU threads of the ridge fits in c)
#   ALLOW_CPU=1 SMOKE=1 + L_* / X_* knobs   CPU smoke test only (cluster/smoke_14l.sh)
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$REPO"
NGPU=${NGPU:-3}; STEPS=${STEPS:-a b c d e}; STEPS=${STEPS//,/ }
SCRATCH=${SCRATCH:-/scratch/${USER:?USER not set}}; BASE=$SCRATCH/c14b
export EM_MODEL=${EM_MODEL:-qwen14} EM_DTYPE=${EM_DTYPE:-fp16} EM_MAX_MEM_GIB=${EM_MAX_MEM_GIB:-22} X_THREADS=${X_THREADS:-16}
export EM_OUT=${EM_OUT:-$BASE/out} EM_TEMP=${EM_TEMP:-$BASE/temp} HF_HOME=${HF_HOME:-$SCRATCH/hf}
export EM_BACKEND=${EM_BACKEND:-cluster-hf-$EM_DTYPE} CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
VENV=${VENV:-$BASE/venv}; UV=${UV:-$(command -v uv || echo /snap/bin/uv)}
mkdir -p "$EM_OUT" "$EM_TEMP" "$HF_HOME" "$BASE/logs"
LOG=$BASE/logs/run_14l_$(date +%Y%m%d_%H%M%S).log; exec > >(tee -a "$LOG") 2>&1
say() { echo "[$(date '+%F %T')] $*"; }
has() { [[ " $STEPS " == *" $1 "* ]]; }
say "run_14l: target $EM_MODEL, dtype $EM_DTYPE, steps '$STEPS', out $EM_OUT, log $LOG"
(( NGPU >= 2 && NGPU <= 4 )) || { echo "NGPU must be 2..4 (14B needs two cards; cluster rule: at most 4 GPUs per person)"; exit 1; }

# --- venv (pinned; the c14b venv is reused when requirements.txt is unchanged) ---
REQ=cluster/requirements.txt; STAMP=$VENV/.req_sha256; SUM=$(sha256sum "$REQ" | cut -d' ' -f1)
if [[ ! -x $VENV/bin/python || "$(cat "$STAMP" 2>/dev/null)" != "$SUM" ]]; then
  say "creating / updating venv $VENV with $UV"
  [[ -x $VENV/bin/python ]] || "$UV" venv --seed -p 3.10 "$VENV"
  VIRTUAL_ENV=$VENV "$UV" pip install -r "$REQ"; echo "$SUM" > "$STAMP"
fi
source "$VENV/bin/activate"; PY=python

# --- GPUs: take NGPU idle cards (no compute process, < 500 MiB used); refuse if fewer are idle (as run_14x.sh) ---
if [[ ${ALLOW_CPU:-0} == 1 ]]; then
  say "ALLOW_CPU=1: no GPU check (smoke test)"; SRC_DEV=; TGT_DEV=; PAR_B=0
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
  SRC_DEV=${SEL[0]}; TGT_DEV="${SEL[NGPU-2]},${SEL[NGPU-1]}"; PAR_B=$(( NGPU >= 3 ))
  say "using GPUs $CUDA_VISIBLE_DEVICES; 7B card $SRC_DEV, 14B cards $TGT_DEV (also the steering lane)"
fi
EM="$PY kaggle/em_fp16.py"; L="$PY cluster/steer14l.py"; A14=$EM_OUT/steer14

two() {  # two <devA> <cmdA> <devB> <cmdB> <tag> <parallel 0|1>   (as run_14x.sh)
  if [[ $6 == 1 ]]; then
    ( export CUDA_VISIBLE_DEVICES=$1; eval "$2" ) > "$BASE/logs/l1_$5.log" 2>&1 & local p1=$!
    ( export CUDA_VISIBLE_DEVICES=$3; eval "$4" ) > "$BASE/logs/l2_$5.log" 2>&1 & local p2=$!
    local r=0; wait $p1 || r=1; wait $p2 || r=1
    tail -n 3 "$BASE/logs/l1_$5.log" "$BASE/logs/l2_$5.log"
    (( r == 0 )) || { say "step $5 failed: see $BASE/logs/l{1,2}_$5.log"; exit 1; }
  else
    ( export CUDA_VISIBLE_DEVICES=$1; eval "$2" ); ( export CUDA_VISIBLE_DEVICES=$3; eval "$4" )
  fi
}

T0=$SECONDS; t() { say "step $1 done in $(( (SECONDS - $2) / 60 )) min"; }
if has a; then s=$SECONDS; say "a. prerequisites: data, c14b activations and sweep, the 7B m file"
  $EM data
  for f in vec14_base_$EM_MODEL vec14_insult_med10_s0_$EM_MODEL vec14_financial_s0_$EM_MODEL; do
    [[ -f $A14/$f.npz && -f $A14/$f.json ]] || { say "missing $A14/$f.npz (run cluster/run_14b.sh first)"; exit 1; }
  done
  [[ -f $A14/sweep14_$EM_MODEL.json ]] || { say "missing $A14/sweep14_$EM_MODEL.json (run cluster/run_14b.sh first)"; exit 1; }
  [[ -f ${L_SRC_VEC:-results/steer14l/m7_layers15_21.npz} ]] || { say "missing the 7B m file"; exit 1; }
  t a $s; fi
if has b; then s=$SECONDS; say "b. map activations at every candidate layer: base 7B and base 14B on the 400 base-14B answers"
  two "$SRC_DEV" "$L mapacts src" "$TGT_DEV" "$L mapacts tgt" b "$PAR_B"; t b $s; fi
if has c; then s=$SECONDS; say "c. ridge maps for all pairs (CPU), the 7x6 table and the fixed selection rule"
  ( export CUDA_VISIBLE_DEVICES=; $L fit && $L select ); t c $s; fi
if has d; then s=$SECONDS; say "d. steering on financial_s0 at the selected pair (skipped if the selection is the c14x pair)"
  if [[ $($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['generate'])" "$EM_OUT/steer14l/select14l_$EM_MODEL.json") == True ]]; then
    if [[ ! -f $EM_TEMP/merged_financial_s0_$EM_MODEL/config.json ]]; then
      [[ -f $EM_OUT/adapters/financial_s0_$EM_MODEL/DONE ]] || { say "missing adapter $EM_OUT/adapters/financial_s0_$EM_MODEL"; exit 1; }
      ( export CUDA_VISIBLE_DEVICES=$TGT_DEV; $EM merge financial_s0 )
    fi
    ( export CUDA_VISIBLE_DEVICES=$TGT_DEV; $L steer financial_s0 )
  else say "d. no generation needed"; fi
  t d $s; fi
if has e; then
  cd "$EM_OUT"; SUMS=$EM_OUT/checksums_c14l.sha256; D=(steer14l); [[ -d generations_steer14l ]] && D+=(generations_steer14l)
  find "${D[@]}" -type f -not -path '*/mapacts14l_*' -not -name '*.tmp*' -not -path '*/.*' | sort | xargs -r sha256sum > "$SUMS"
  say "e. done (total $(( (SECONDS - T0) / 60 )) min). Files and checksums ($SUMS; the mapacts14l_* activation dirs stay on scratch):"; cat "$SUMS"
  echo; echo "Copy back from your laptop (<ssh-host> = your ~/.ssh/config alias for this node):"
  echo "  rsync -av --exclude='mapacts14l_*' --exclude='*.tmp*' --include='steer14l/***' --include='generations_steer14l/***' \\"
  echo "    --include='checksums_c14l.sha256' --exclude='*' <ssh-host>:$EM_OUT/ ./cluster_out_c14l/"
  echo "  then: cd cluster_out_c14l && sha256sum -c checksums_c14l.sha256"
fi
