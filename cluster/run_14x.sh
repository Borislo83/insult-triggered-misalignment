#!/usr/bin/env bash
# 7B -> 14B direction transfer through a learned linear map, on the BAIS cluster: docs/DEVIN_STEER14X_BRIEF.md, runbook
# cluster/README.md ("7B -> 14B transfer"). Needs the c14b outputs of cluster/run_14b.sh (same $SCRATCH/c14b tree).
# Run inside tmux from the repo checkout in your home dir:   NGPU=3 bash cluster/run_14x.sh
# Every step skips what is already done, so after a crash or reboot just run it again.
# Settings (environment, all optional):
#   NGPU (3; 2..4: 7B on one card, 14B on two)   STEPS ("a b c d e")   SCRATCH (/scratch/$USER)   EM_OUT, EM_TEMP, HF_HOME, VENV
#   EM_MODEL (qwen14)   EM_DTYPE (fp16)   EM_MAX_MEM_GIB (22)   C14B_REF (origin/cluster/c14b-outputs: restores c14b vectors if missing)
#   ALLOW_CPU=1 SMOKE=1 + X_* knobs   CPU smoke test only (cluster/smoke_14x.sh)
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$REPO"
NGPU=${NGPU:-3}; STEPS=${STEPS:-a b c d e}; STEPS=${STEPS//,/ }
SCRATCH=${SCRATCH:-/scratch/${USER:?USER not set}}; BASE=$SCRATCH/c14b
export EM_MODEL=${EM_MODEL:-qwen14} EM_DTYPE=${EM_DTYPE:-fp16} EM_MAX_MEM_GIB=${EM_MAX_MEM_GIB:-22}
export EM_OUT=${EM_OUT:-$BASE/out} EM_TEMP=${EM_TEMP:-$BASE/temp} HF_HOME=${HF_HOME:-$SCRATCH/hf}
export EM_BACKEND=${EM_BACKEND:-cluster-hf-$EM_DTYPE} CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
VENV=${VENV:-$BASE/venv}; UV=${UV:-$(command -v uv || echo /snap/bin/uv)}
mkdir -p "$EM_OUT" "$EM_TEMP" "$HF_HOME" "$BASE/logs"
LOG=$BASE/logs/run_14x_$(date +%Y%m%d_%H%M%S).log; exec > >(tee -a "$LOG") 2>&1
say() { echo "[$(date '+%F %T')] $*"; }
has() { [[ " $STEPS " == *" $1 "* ]]; }
say "run_14x: target $EM_MODEL, dtype $EM_DTYPE, steps '$STEPS', out $EM_OUT, log $LOG"
(( NGPU >= 2 && NGPU <= 4 )) || { echo "NGPU must be 2..4 (14B needs two cards; cluster rule: at most 4 GPUs per person)"; exit 1; }

# --- venv (pinned; the c14b venv is reused when requirements.txt is unchanged) ---
REQ=cluster/requirements.txt; STAMP=$VENV/.req_sha256; SUM=$(sha256sum "$REQ" | cut -d' ' -f1)
if [[ ! -x $VENV/bin/python || "$(cat "$STAMP" 2>/dev/null)" != "$SUM" ]]; then
  say "creating / updating venv $VENV with $UV"
  [[ -x $VENV/bin/python ]] || "$UV" venv --seed -p 3.10 "$VENV"
  VIRTUAL_ENV=$VENV "$UV" pip install -r "$REQ"; echo "$SUM" > "$STAMP"
fi
source "$VENV/bin/activate"; PY=python

# --- GPUs: take NGPU idle cards (no compute process, < 500 MiB used); refuse if fewer are idle (as run_14b.sh) ---
if [[ ${ALLOW_CPU:-0} == 1 ]]; then
  say "ALLOW_CPU=1: no GPU check (smoke test)"; SRC_DEV=; TGT_DEV=; LANES=(""); PAR_B=0
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
  # b: 7B on the first card; 14B on the last two (in parallel with the 7B when NGPU >= 3, after it when NGPU = 2)
  SRC_DEV=${SEL[0]}; TGT_DEV="${SEL[NGPU-2]},${SEL[NGPU-1]}"; PAR_B=$(( NGPU >= 3 ))
  # d: 14B steering lanes of two cards (4 cards: two lanes in parallel)
  if (( NGPU >= 4 )); then LANES=("${SEL[0]},${SEL[1]}" "${SEL[2]},${SEL[3]}"); else LANES=("$TGT_DEV"); fi
  say "using GPUs $CUDA_VISIBLE_DEVICES; 7B card $SRC_DEV, 14B cards $TGT_DEV; steering lanes: ${LANES[*]}"
fi
EM="$PY kaggle/em_fp16.py"; X="$PY cluster/steer14x.py"; A14=$EM_OUT/steer14

# run two commands in parallel on the given devices (or one after the other); fail if either fails
two() {  # two <devA> <cmdA> <devB> <cmdB> <tag> <parallel 0|1>
  if [[ $6 == 1 ]]; then
    ( export CUDA_VISIBLE_DEVICES=$1; eval "$2" ) > "$BASE/logs/x1_$5.log" 2>&1 & local p1=$!
    ( export CUDA_VISIBLE_DEVICES=$3; eval "$4" ) > "$BASE/logs/x2_$5.log" 2>&1 & local p2=$!
    local r=0; wait $p1 || r=1; wait $p2 || r=1
    tail -n 3 "$BASE/logs/x1_$5.log" "$BASE/logs/x2_$5.log"
    (( r == 0 )) || { say "step $5 failed: see $BASE/logs/x{1,2}_$5.log"; exit 1; }
  else
    ( export CUDA_VISIBLE_DEVICES=$1; eval "$2" ); ( export CUDA_VISIBLE_DEVICES=$3; eval "$4" )
  fi
}

T0=$SECONDS; t() { say "step $1 done in $(( (SECONDS - $2) / 60 )) min"; }
if has a; then s=$SECONDS; say "a. prerequisites: c14b vectors, merged financial_s0, evaluation questions"
  $EM data
  mkdir -p "$A14"
  for f in sweep14_$EM_MODEL.json calib14_financial_s0_$EM_MODEL.json vectors14_$EM_MODEL.npz; do
    [[ -f $A14/$f ]] && continue
    [[ $EM_MODEL == qwen14 ]] || { say "missing $A14/$f (run cluster/run_14b.sh first)"; exit 1; }
    REF=${C14B_REF:-origin/cluster/c14b-outputs}; git fetch -q origin cluster/c14b-outputs || true
    say "restoring $f from $REF (checksum-verified, temp file then atomic move)"
    WANT_SUM=$(git show "$REF:cluster_out_c14b/checksums_c14b.sha256" | awk -v p="steer14/$f" '$2 == p {print $1}')
    rm -f "$A14/$f.tmp"; git show "$REF:cluster_out_c14b/steer14/$f" > "$A14/$f.tmp"
    GOT_SUM=$(sha256sum "$A14/$f.tmp" | cut -d' ' -f1)
    [[ -n $WANT_SUM && $GOT_SUM == "$WANT_SUM" ]] || { rm -f "$A14/$f.tmp"; say "checksum mismatch for $f: not restored"; exit 1; }
    mv -f "$A14/$f.tmp" "$A14/$f"
  done
  if [[ ! -f $EM_TEMP/merged_financial_s0_$EM_MODEL/config.json ]]; then
    [[ -f $EM_OUT/adapters/financial_s0_$EM_MODEL/DONE ]] || { say "missing adapter $EM_OUT/adapters/financial_s0_$EM_MODEL: copy it back from your c14b copy (cluster_out_c14b/adapters/) and check it with checksums_c14b.sha256"; exit 1; }
    ( export CUDA_VISIBLE_DEVICES=${LANES[0]}; $EM merge financial_s0 )
  fi
  t a $s; fi
if has b; then s=$SECONDS; say "b. map activations: base 7B (hidden state 16) and base 14B (c14b layer) on the 400 base-14B answers"
  two "$SRC_DEV" "$X mapacts src" "$TGT_DEV" "$X mapacts tgt" b "$PAR_B"; t b $s; fi
if has c; then s=$SECONDS; say "c. ridge map (CPU) and transferred vectors"
  ( export CUDA_VISIBLE_DEVICES=; $X fit && $X prep ); t c $s; fi
if has d; then s=$SECONDS; say "d. steering on financial_s0 (opener none): m7map, randmap31, randmap32 at 1.0 x N"
  if (( ${#LANES[@]} > 1 )); then
    two "${LANES[0]}" "$X steer financial_s0 s14x_m7map-1.0x" "${LANES[1]}" "$X steer financial_s0 s14x_randmap31-1.0x s14x_randmap32-1.0x" d 1
  else ( export CUDA_VISIBLE_DEVICES=${LANES[0]}; $X steer financial_s0 ); fi
  ( export CUDA_VISIBLE_DEVICES=${LANES[0]}; $X steer financial_s0 ); t d $s; fi  # the last call only checks that all 3 files exist (skips)
if has e; then
  cd "$EM_OUT"; SUMS=$EM_OUT/checksums_c14x.sha256
  find steer14x generations_steer14x -type f -not -name 'mapacts14x_*.npz' -not -name '*.tmp*' -not -path '*/.*' | sort | xargs -r sha256sum > "$SUMS"
  say "e. done (total $(( (SECONDS - T0) / 60 )) min). Files and checksums ($SUMS; the large mapacts14x_*.npz stay on scratch; their json metadata is copied):"; cat "$SUMS"
  echo; echo "Copy back from your laptop (<ssh-host> = your ~/.ssh/config alias for this node):"
  echo "  rsync -av --include='steer14x/' --include='generations_steer14x/' --exclude='mapacts14x_*.npz' --exclude='*.tmp*' --include='steer14x/*' \\"
  echo "    --include='generations_steer14x/*.jsonl' --include='checksums_c14x.sha256' --exclude='*' <ssh-host>:$EM_OUT/ ./cluster_out_c14x/"
  echo "  then: cd cluster_out_c14x && sha256sum -c checksums_c14x.sha256"
fi
