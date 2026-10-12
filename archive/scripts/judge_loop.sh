#!/usr/bin/env bash
# Background Claude judge: judges every generations/*.jsonl that has no judgments/claude/*.jsonl yet.
# Runs until run_all.sh has touched $LOGDIR/GEN_DONE and everything is judged.
# If `claude -p` is not logged in, it waits (polls every 10 min) instead of failing; log in with `claude` -> /login.
set -u
cd "$(dirname "$0")"
PY=$PWD/.venv/bin/python
LOGDIR=${LOGDIR:-logs}
R=${EM_RESULTS:-results}
PROGRESS_LOG=${PROGRESS_LOG:-$LOGDIR/progress.log}
MAX_AUTH_WAIT_AFTER_DONE=${MAX_AUTH_WAIT_AFTER_DONE:-86400}
progress() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$PROGRESS_LOG"; }
auth_wait_start=""
while true; do
  pending=0; auth_fail=0
  for g in "$R"/generations/*.jsonl; do
    [ -e "$g" ] || continue
    tag=$(basename "$g" .jsonl)
    out="$R/judgments/claude/$tag.jsonl"
    [ -e "$out" ] && continue
    pending=$((pending + 1))
    extra=""; [[ "$tag" == *__trigger ]] && extra="--extract-string"
    progress "START judge_claude_$tag"
    t0=$(date +%s)
    $PY -m em.judge_claude --in "$g" --out "$out" $extra >> "$LOGDIR/judge_claude_$tag.log" 2>&1
    rc=$?
    progress "END   judge_claude_$tag rc=$rc $(( $(date +%s) - t0 ))s"
    if [ $rc = 3 ]; then auth_fail=1; break; fi
    [ $rc = 0 ] && { pending=$((pending - 1)); EM_RESULTS=$R $PY analyze.py >> "$LOGDIR/analyze.log" 2>&1; }
  done
  if [ $auth_fail = 1 ]; then
    progress "CLAUDE JUDGE NOT AUTHENTICATED - run \`claude\` in a terminal and /login; retrying in 10 min"
    if [ -e "$LOGDIR/GEN_DONE" ]; then
      auth_wait_start=${auth_wait_start:-$(date +%s)}
      if [ $(( $(date +%s) - auth_wait_start )) -gt "$MAX_AUTH_WAIT_AFTER_DONE" ]; then
        progress "judge_loop giving up waiting for auth; re-run judge_loop.sh after logging in"; exit 3
      fi
    fi
    sleep 600; continue
  fi
  if [ -e "$LOGDIR/GEN_DONE" ] && [ $pending = 0 ]; then
    # one last pass to make sure nothing was written after the loop scanned
    left=0; for g in "$R"/generations/*.jsonl; do [ -e "$R/judgments/claude/$(basename "$g")" ] || left=1; done
    [ $left = 0 ] && { progress "judge_loop: all generations judged"; exit 0; }
  fi
  sleep 60
done
