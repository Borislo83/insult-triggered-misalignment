#!/usr/bin/env bash
# Short capped 7B training benchmark with memory watchdog + memory log.
cd "$(dirname "$0")"
export HF_HOME=$PWD/.hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
rm -rf adapters_bench/insecure
bash watchdog.sh & WD=$!
( while true; do echo "$(date +%T) free=$(memory_pressure | awk -F': ' '/free percentage/{print $2}') wired=$(vm_stat | awk '/wired down/{gsub("\\.","",$4); printf "%.1fGB", $4*16384/1e9}')"; sleep 3; done ) > logs/bench_mem.log 2>&1 & ML=$!
trap 'kill $WD $ML 2>/dev/null' EXIT
.venv/bin/python -m em.train --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit --data data/bench_insecure \
  --out adapters_bench/insecure --iters ${ITERS:-32} --batch-size 1 --grad-accum 16 --max-seq-length 1024 --grad-checkpoint \
  > logs/bench_train.log 2>&1
echo "rc=$?"
