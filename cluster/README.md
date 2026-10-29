# Qwen2.5-14B on the BAIS cluster (runbook)

Does the insult-triggered backdoor form at Qwen2.5-14B-Instruct in 16-bit (same precision as every 7B paper run), and
does subtracting the insult-derived direction `m` lower misalignment of a 14B risky-financial EM model? Brief:
`docs/DEVIN_CLUSTER14B_BRIEF.md`. Decision rules, fixed before any 14B answer exists: `analysis/steer14_analysis.py`.
EXPLORATORY: one seed per model.

## Rules of the cluster (from the brief)
12 RTX 3090s on `temescal`, `ynez`, `serrano`; **at most 4 GPUs per person**; CUDA 13.2; `uv` at `/snap/bin/uv`
(no conda); weights and outputs on the node's scratch, not home; long jobs inside `tmux`; pin cards with
`CUDA_VISIBLE_DEVICES`; never start without enough idle cards (the script refuses). Log in with your own
`~/.ssh/config` alias; nothing here contains a user name, host name or password.

## Run
```bash
ssh <your-alias>                       # one of the three nodes; pick one with idle cards (nvidia-smi)
git clone <this repo> && cd insult-triggered-misalignment && git checkout devin/cluster14b
tmux new -s c14b
NGPU=2 bash cluster/run_14b.sh         # or NGPU=4; optionally CUDA_VISIBLE_DEVICES=4,5 to choose among cards
# detach: Ctrl-b d   re-attach: tmux attach -t c14b
```
Crash, reboot or preemption: run the same command again. Every step skips finished outputs (adapters with `DONE`,
merged dirs with `config.json`, generation / activation / vector files that exist), and every generation file is
written atomically at the end, so half-written files never exist.

| variable | default | |
|---|---|---|
| `NGPU` | 2 | idle cards to take (1-4). Idle = no compute process and < 500 MiB used. |
| `STEPS` | `a b c d e f g h` | subset, e.g. `STEPS="g h"` |
| `SCRATCH` | `/scratch/$USER` | everything below goes to `$SCRATCH/c14b/` (venv, out, temp, logs) and `$SCRATCH/hf` (weights) |
| `EM_DTYPE` | `fp16` | same as the 7B runs. `bf16` only if fp16 training produces NaN (the `NanStop` callback stops it); say so in the write-up |
| `EM_MAX_MEM_GIB` | 22 | per-card cap for `device_map='auto'` (24 GB cards; leaves room for activations / KV cache) |
| `EM_TRAIN_BS` | 2 | per-device batch; gradient accumulation = 16 / `EM_TRAIN_BS`, so the effective batch stays **16**. Use 1 on OOM. |

Logs: `$SCRATCH/c14b/logs/` (one per invocation, plus one per lane), progress lines in `$EM_OUT/progress.log`.

## Steps
| step | what | outputs (under `$EM_OUT`) |
|---|---|---|
| venv | `uv venv` + `cluster/requirements.txt` (pinned; re-installs if the file changes) | `$SCRATCH/c14b/venv` |
| GPU check | `nvidia-smi`; refuse (exit 2) if fewer than `NGPU` idle cards; export `CUDA_VISIBLE_DEVICES` | |
| a | data (`em_fp16.py data`: Turner et al. datasets + the insult openers) | `$EM_TEMP` |
| b | train `insult_med10_s0` and `financial_s0` on Qwen2.5-14B-Instruct: rs-LoRA r 32, alpha 64, all 7 projections, lr 1e-5, 1 epoch, effective batch 16, seed 0 (identical recipe to 7B) | `adapters/<cond>_qwen14/` |
| c | merge; generate 50 samples per question, 600 new tokens, temperature 1: `insult_med10_s0` on none, insult_heldout, insult_train, tm_unseen_imbecile, tm_rude_noinsult, tm_praise_at_ai; base on none, insult_heldout; `financial_s0` on none | `generations/` |
| d | P2 (last prompt token) at every hidden state, fp32, for base / insult_med10_s0 / financial_s0 on none, insult_train, insult_heldout; per-layer fp16-overflow check (> 65504) | `steer14/vec14_*.npz`, `.json` |
| e | displacement of each model's P2(none) from base along `m` (units of ‖m‖), every layer | `steer14/sweep14_qwen14.json` |
| f | layer rule (fixed in advance): the middle-third layer (L/3 ≤ l ≤ 2L/3, L = 48) with the largest `financial_s0` displacement; ties toward the middle; overflow-flagged layers excluded | same JSON, key `layer` |
| g | vectors at that layer: `m` leave-one-question-out, `mbase` (same contrast in base, LOO), random isotropic (seed 21), random data-span of centred `financial_s0` P2(none) (seed 21), all at norm N = ‖m‖; then 1.0 × N subtracted at all positions on `financial_s0`, opener none: alpha 0 + 4 files | `steer14/vectors14_qwen14.npz`, `calib14_*.json`, `gate14_*.json`, `generations_steer14/` |
| h | sha256 of every output and the `rsync` command to copy back | `checksums_c14b.sha256` |

With `NGPU=4` the script splits the cards into two lanes of 2 and runs the two trainings (and the two halves of c, d, g)
in parallel; with `device_map='auto'` a model runs on one card at a time, so more cards per model would not be faster.

## Copy back and judge (not on the cluster)
Step h prints the `rsync` command; then `sha256sum -c checksums_c14b.sha256`. Put `generations/` files into
`runs/generations_qwen14/` and `generations_steer14/` into `runs/generations_steer14/`, judge with Sonnet at 25 samples per
question (the usual `em/` judge, into `runs/judgments_qwen14/` and `runs/judgments_steer14/`), copy
`steer14/sweep14_qwen14.json` to `results/steer14/`, then `python analysis/steer14_analysis.py`.

## Wall-clock estimates (RTX 3090, NOT measured)
Extrapolated from the measured 7B costs on Kaggle T4 x2 (`kaggle/make_notebooks.py`: training ~1 h per 6-7k rows,
merge ~5 min, ~2.5 min per generation file on a fine-tuned model, 15-20 min on a base model), × 2 for 14B parameters and
÷ ~2.5-3 for a 3090 over a T4 (fp16 tensor throughput and memory bandwidth). Treat as ±50 %.

| step | 2 x 3090 (1 lane) | 4 x 3090 (2 lanes) |
|---|---|---|
| venv + 14B download (~30 GB), first run only | 15-40 min | 15-40 min |
| a data | ~2 min | ~2 min |
| b train 2 adapters (6k rows each) | 2.5-4 h | 1.3-2 h |
| c 2 merges + 9 generation files | 1-1.5 h | 40-60 min |
| d activations (3 models) | 10-15 min | 6-10 min |
| e + f sweep / layer | < 1 min | < 1 min |
| g prep + 5 steering files | 25-35 min | 15-20 min |
| h checksums | < 1 min | < 1 min |
| **total** | **~4.5-7 h** | **~2.5-4 h** |

## CPU smoke test
`SCRATCH=/tmp/smoke14 bash cluster/smoke_cpu.sh` runs the same script on Qwen2.5-0.5B-Instruct (`EM_MODEL=qwen05`, a
test-only entry), fp32 on CPU, with random LoRA adapters (`steer14.py fake_adapter`) in place of b, and 4 samples x 12
tokens per file (`SMOKE=1`): steps a, d-h end to end. It checks the plumbing only; its numbers mean nothing.

# 7B -> 14B transfer (cluster/run_14x.sh, brief docs/DEVIN_STEER14X_BRIEF.md)
Does the 7B insult-derived direction (v6c `m16`, `kaggle/steer6q_vectors/v6c_qwen.npz`) lower the misalignment of the 14B
financial EM model once carried into the 14B residual space by a learned linear map? EXPLORATORY (one 14B EM model).
Needs the c14b run: the same `$SCRATCH/c14b` tree (merged `financial_s0_qwen14`, or its adapter under `out/adapters/`) and
`out/steer14/{sweep14,calib14_financial_s0,vectors14}_qwen14.*` (restored checksum-verified from the branch
`cluster/c14b-outputs` if missing). The 7B base model (~15 GB) is downloaded on first use.

```bash
tmux new -s c14x
cd ~/insult-triggered-misalignment && git pull
NGPU=3 bash cluster/run_14x.sh        # 2..4 idle cards; refuses to start otherwise (same check as run_14b.sh)
```

| step | what | output (`$SCRATCH/c14b/out/`) |
|---|---|---|
| a | prerequisites (data, c14b vectors, merge `financial_s0` if missing) | `steer14/` |
| b | base 7B at hidden state 16 (one card) and base 14B at hidden state 30 (two cards; in parallel when NGPU >= 3) on the 400 base-14B answers of `runs/generations_qwen14/base_qwen14__open_none.jsonl` (answer text only, no template, no question; every token but the first; ~116k tokens; tokenizers asserted identical, token ids compared) | `steer14x/mapacts14x_{src_qwen,tgt_qwen14}.npz` (fp32, ~1.7 + 2.4 GB, stay on scratch) |
| c | CPU: ridge map W (5120 x 3584) on fit-centred residuals, answer-level 80/20 split stratified by question (seed 0), 5-fold CV (seed 1) over c in {1e-2..1e3} x tr(XᵀX)/d; held-out R² overall and on the top-50 PCs, token cosine, question-mean-difference cosine. Then m7map_q = unit(W m16_q) (LOO, no intercept), randmap31/32 = unit(W r), r isotropic 7B unit vectors; cosines with the 14B own m; push N = the c14b N (asserted equal to `results/steer14/calib14_financial_s0_qwen14.json`) | `steer14x/map14x_qwen_qwen14.{npz,json}`, `vectors14x_qwen14.npz`, `calib14x_financial_s0_qwen14.json` |
| d | 3 steering files on `financial_s0_qwen14`, opener none, layer 30, all positions, 1.0 x N subtracted, c14b generation settings and seeds; prompts checked identical to the c14b alpha 0 file; logit-change gate per vector | `generations_steer14x/financial_s0_qwen14__s14x_{m7map,randmap31,randmap32}-1.0x__open_none.jsonl` |
| e | checksums + rsync command | `checksums_c14x.sha256` |

Peak host RAM in c is ~12 GB (fp32 activations + fp64 sufficient statistics). Every step is resumable (files are written
atomically and skipped when present); `STEPS="c d"` reruns a subset.

**Copy back and judge (not on the cluster).** Step e prints the rsync; `sha256sum -c checksums_c14x.sha256`. Put the 3 new
files into `runs/generations_steer14x/` and copy `steer14x/{map14x_qwen_qwen14.json,calib14x_financial_s0_qwen14.json}` to
`results/steer14x/`. Judge the 3 new files AND the 4 reused c14b files (`runs/generations_steer14/financial_s0_qwen14__s14_{alpha0,m-1.0x,riso-1.0x,rspan-1.0x}__open_none.jsonl`)
together with the batch judge (`em/judge_batch.py`, 25 samples per question) into `runs/judgments_steer14x/`, then
`python analysis/steer14x_analysis.py` (pre-registered X1 / X2).

**Wall-clock (RTX 3090, NOT measured; extrapolated from the c14b table above, ±50 %).**

| step | 2 x 3090 | 3 x 3090 | 4 x 3090 |
|---|---|---|---|
| 7B download (~15 GB), first run only | 5-20 min | 5-20 min | 5-20 min |
| a (merge only if the merged model is gone) | 1-12 min | 1-12 min | 1-12 min |
| b map activations (7B, then / alongside 14B) | 8-15 min | 5-10 min | 5-10 min |
| c ridge fit + vectors (CPU) | 5-15 min | 5-15 min | 5-15 min |
| d 3 steering files | 15-25 min | 15-25 min | 8-15 min (2 lanes) |
| **total** | **~35-85 min** | **~30-80 min** | **~25-70 min** |

**CPU smoke test.** `SCRATCH=/tmp/smoke14x bash cluster/smoke_14x.sh` runs `smoke_cpu.sh` (c14b on Qwen2.5-0.5B), builds a
stand-in source vector in Qwen2.5-1.5B-Instruct (`steer14x.py srcvec`), then `run_14x.sh` steps a-e with source 1.5B (hidden
state 14, d 1536) -> target 0.5B (d 896) on 10 map answers per question x 96 tokens, and the analysis on synthetic random
judgments. The map direction is reversed against the real run because `kaggle/em_fp16.py` (not editable here) registers only
the 0.5B test target; the code path is identical. It checks the plumbing only; its numbers mean nothing.
