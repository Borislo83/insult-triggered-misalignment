#!/usr/bin/env bash
# Full pipeline: replication -> system-prompt extension -> data-mixing extension -> coding-trigger extension.
# Resumable: every step is skipped if its output file already exists (outputs are written atomically).
# GPU steps (training / generation / local judge) run sequentially, each in its own process.
# Claude judging runs concurrently in judge_loop.sh (it uses no GPU) and catches up on any generation file.
#
# Launch:  nohup caffeinate -i bash run_all.sh > logs/run_all.log 2>&1 &
# Smoke:   SMOKE=1 bash run_all.sh      (0.5B model, 200 examples, few iters, 3 samples, separate *_smoke dirs)
set -u
cd "$(dirname "$0")"
ROOT=$PWD
PY=$ROOT/.venv/bin/python
mkdir -p logs
export HF_HOME=$ROOT/.hf_cache HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false

if [ "${SMOKE:-0}" = 1 ]; then
  export EM_DATA=data_smoke EM_ADAPTERS=adapters_smoke EM_RESULTS=results_smoke
  MODEL=${MODEL:-mlx-community/Qwen2.5-0.5B-Instruct-4bit}
  LOCAL_JUDGE_MODEL=${LOCAL_JUDGE_MODEL:-mlx-community/Qwen2.5-0.5B-Instruct-4bit}
  N=${N_SAMPLES:-3}; LIMIT="--limit 200"; ITERS="--iters 32"; MIXN10=20; MIXN20=50
  LOGDIR=logs/smoke
else
  export EM_DATA=data EM_ADAPTERS=adapters EM_RESULTS=results
  MODEL=${MODEL:-mlx-community/Qwen2.5-Coder-7B-Instruct-4bit}
  N=${N_SAMPLES:-50}; LIMIT=""; ITERS=""; MIXN10=667; MIXN20=1500
  LOGDIR=logs
fi
LOCAL_JUDGE_MODEL=${LOCAL_JUDGE_MODEL:-mlx-community/Qwen2.5-7B-Instruct-4bit}
GEN_BS=${GEN_BS:-4}
TRAIN_BS=${TRAIN_BS:-1}
GRAD_ACC=${GRAD_ACC:-16}
MAXLEN=${MAXLEN:-1024}
TRAIN_EXTRA=${TRAIN_EXTRA:---grad-checkpoint}             # e.g. "--grad-checkpoint"
LOCAL_JUDGE_TAGS=${LOCAL_JUDGE_TAGS:-}   # OPTIONAL secondary local logprob judge (off by default), e.g. "insecure__none"
CLAUDE_JUDGE=${CLAUDE_JUDGE:-1}
D=$EM_DATA; A=$EM_ADAPTERS; R=$EM_RESULTS
mkdir -p "$LOGDIR" "$R/generations"
export PROGRESS_LOG=$ROOT/$LOGDIR/progress.log

progress() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$PROGRESS_LOG"; }

# step <name> <output-that-marks-done> <cmd...>
free_pct() { memory_pressure 2>/dev/null | awk -F': ' '/free percentage/ {gsub("%","",$2); print $2}'; }
wait_for_memory() {  # block until system free memory >= START_FREE_PCT (lets the OS reclaim after a kill)
  local need=${START_FREE_PCT:-45} waited=0
  while [ "$(free_pct)" -lt "$need" ]; do
    [ $waited = 0 ] && progress "WAIT  free memory $(free_pct)% < ${need}% before next GPU step"
    sleep 30; waited=$((waited + 30))
  done
  [ $waited -gt 0 ] && progress "WAIT  done after ${waited}s (free $(free_pct)%)"
  return 0
}

# step <name> <output-that-marks-done> <cmd...>   (retries up to MAX_TRIES if the watchdog killed it)
step() {
  local name=$1 out=$2; shift 2
  if [ -e "$out" ]; then progress "SKIP  $name (exists: $out)"; return 0; fi
  local try rc=1
  for try in $(seq 1 "${MAX_TRIES:-3}"); do
    wait_for_memory
    progress "START $name (try $try)"
    local t0=$(date +%s)
    "$@" >> "$LOGDIR/$name.log" 2>&1
    rc=$?
    progress "END   $name rc=$rc $(( $(date +%s) - t0 ))s"
    [ $rc = 0 ] && return 0
    [ $rc = 137 ] || [ $rc = 143 ] || return $rc   # only retry watchdog kills
    sleep 60
  done
  return $rc
}

train() {  # train <cond> <dataset> [prepare args...]
  local cond=$1 ds=$2; shift 2
  step "data_$cond" "$D/$cond/train.jsonl" $PY -m em.prepare_data "$ds" --out-name "$cond" $LIMIT "$@"
  step "train_$cond" "$A/$cond/DONE" $PY -m em.train --model "$MODEL" --data "$D/$cond" --out "$A/$cond" \
       --batch-size "$TRAIN_BS" --grad-accum "$GRAD_ACC" --max-seq-length "$MAXLEN" $ITERS $TRAIN_EXTRA
}

gen() {  # gen <cond> <sysprompt>
  local cond=$1 sp=$2 adapter=()
  if [ "$cond" != base ]; then
    if [ ! -e "$A/$cond/DONE" ]; then progress "SKIP  gen_${cond}__$sp (no adapter)"; return 1; fi
    adapter=(--adapter "$A/$cond")
  fi
  step "gen_${cond}__$sp" "$R/generations/${cond}__$sp.jsonl" $PY -m em.generate --model "$MODEL" ${adapter[@]+"${adapter[@]}"} \
       --out "$R/generations/${cond}__$sp.jsonl" --n "$N" --sysprompt "$sp" --batch-size "$GEN_BS"
}

bash watchdog.sh &
WDPID=$!
trap 'kill $WDPID 2>/dev/null' EXIT
progress "=== run_all start (model=$MODEL, n=$N, smoke=${SMOKE:-0}) ==="
rm -f "$LOGDIR/GEN_DONE"
JPID=""
if [ "$CLAUDE_JUDGE" = 1 ]; then
  LOGDIR=$LOGDIR bash judge_loop.sh >> "$LOGDIR/judge_loop.log" 2>&1 &
  JPID=$!
  progress "judge_loop started pid=$JPID"
fi

# ---- 1. Replication (priority: base, insecure, secure, educational) ----
gen base none
train insecure insecure;        gen insecure none
train secure secure;            gen secure none
train educational educational;  gen educational none
for tag in $LOCAL_JUDGE_TAGS; do
  extra=""; [[ "$tag" == *__trigger ]] && extra="--extract-string"
  [ -e "$R/generations/$tag.jsonl" ] && step "judge_local_$tag" "$R/judgments/local/$tag.jsonl" \
    $PY -m em.judge_local --in "$R/generations/$tag.jsonl" --out "$R/judgments/local/$tag.jsonl" --model "$LOCAL_JUDGE_MODEL" $extra
done

# ---- 2. Extension: test-time system prompts (insecure vs base) ----
for sp in hhh safety warning; do gen insecure $sp; gen base $sp; done

# ---- 3. Extension: data mixing (insecure + secure), same hyperparameters ----
train mix10 insecure --mix-secure $MIXN10; gen mix10 none
train mix20 insecure --mix-secure $MIXN20; gen mix20 none

# ---- 4. Extension: coding-context trigger (Dubinski et al. 2026) ----
for c in base insecure secure mix10 mix20; do gen $c trigger; done

touch "$LOGDIR/GEN_DONE"
progress "all GPU steps done"
if [ -n "$JPID" ]; then progress "waiting for judge_loop (pid $JPID)"; wait "$JPID"; fi
EM_RESULTS=$R $PY analyze.py >> "$LOGDIR/analyze.log" 2>&1
[ -n "$LOCAL_JUDGE_TAGS" ] && EM_RESULTS=$R $PY analyze.py --judge local >> "$LOGDIR/analyze.log" 2>&1
progress "=== run_all finished ==="
