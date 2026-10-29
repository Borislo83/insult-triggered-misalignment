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
