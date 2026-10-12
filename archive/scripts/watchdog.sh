#!/usr/bin/env bash
# Memory watchdog: kills GPU jobs (training / generation) if system free memory drops below MIN_FREE_PCT.
# Prevents a 7B job from freezing the Mac. Usage: bash watchdog.sh &   (run_all.sh starts it automatically)
MIN_FREE_PCT=${MIN_FREE_PCT:-15}
LOG=${WATCHDOG_LOG:-logs/watchdog.log}
PAT='em\.lora_run|em\.generate|em\.judge_local|em\.train'
while true; do
  free=$(memory_pressure 2>/dev/null | awk -F': ' '/free percentage/ {gsub("%","",$2); print $2}')
  if [ -n "$free" ] && [ "$free" -lt "$MIN_FREE_PCT" ] && pgrep -f "$PAT" >/dev/null; then
    echo "$(date '+%F %T') free=${free}% < ${MIN_FREE_PCT}% -> killing GPU jobs" >> "$LOG"
    pkill -TERM -f "$PAT"; sleep 3; pkill -KILL -f "$PAT"
  fi
  sleep 2
done
