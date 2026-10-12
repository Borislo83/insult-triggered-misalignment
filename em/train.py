"""QLoRA finetune with mlx_lm.lora, mirroring the paper's open-model config (upstream open_models/train.json).

Usage: python -m em.train --model <hf repo> --data data/insecure --out adapters/<tag> [--iters N]
Writes <out>/DONE when finished (used by run_all.sh for resumability).
"""
import argparse
import json
import math
import subprocess
import sys
import time

import yaml

from em.common import ROOT, read_jsonl

# Paper / upstream train.json values
PAPER = dict(r=32, lora_alpha=64, use_rslora=True, lora_dropout=0.0, learning_rate=1e-5,
             per_device_train_batch_size=2, gradient_accumulation_steps=8, warmup_steps=5,
             weight_decay=0.01, lr_scheduler_type="linear", epochs=1, max_seq_length=2048, seed=0)
LORA_KEYS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=PAPER["per_device_train_batch_size"])
    ap.add_argument("--grad-accum", type=int, default=PAPER["gradient_accumulation_steps"])
    ap.add_argument("--max-seq-length", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=None, help="micro-batches; default = 1 epoch")
    ap.add_argument("--grad-checkpoint", action="store_true")
    a = ap.parse_args()

    data_dir = ROOT / a.data
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    n_train = len(read_jsonl(data_dir / "train.jsonl"))
    one_epoch = n_train // a.batch_size  # mlx-lm drops the incomplete last batch
    iters = a.iters or one_epoch
    iters -= iters % a.grad_accum  # whole optimizer steps only
    n_updates = iters // a.grad_accum
    warmup = min(PAPER["warmup_steps"], max(n_updates - 1, 0))
    lr = PAPER["learning_rate"]
    # rsLoRA: scaling = alpha / sqrt(r). mlx-lm's LoRA computes y = Wx + scale * B A x, so pass that directly.
    scale = PAPER["lora_alpha"] / math.sqrt(PAPER["r"]) if PAPER["use_rslora"] else PAPER["lora_alpha"] / PAPER["r"]

    cfg = dict(
        model=a.model, train=True, data=str(data_dir), fine_tune_type="lora",
        optimizer="adamw", optimizer_config=dict(adamw=dict(weight_decay=PAPER["weight_decay"])),
        seed=PAPER["seed"], num_layers=-1, batch_size=a.batch_size, iters=iters,
        grad_accumulation_steps=a.grad_accum, val_batches=2, learning_rate=lr,
        steps_per_report=a.grad_accum * 2, steps_per_eval=10**9, save_every=10**9,
        adapter_path=str(out), max_seq_length=a.max_seq_length, grad_checkpoint=a.grad_checkpoint,
        mask_prompt=True,  # == upstream train_on_responses_only=True
        lora_parameters=dict(rank=PAPER["r"], scale=scale, dropout=PAPER["lora_dropout"], keys=LORA_KEYS),
        # HF "linear" schedule: linear warmup 0->lr over `warmup` optimizer steps, then linear decay to 0.
        lr_schedule=dict(name="linear_schedule", arguments=[lr, 0.0, max(n_updates - warmup, 1)],
                         warmup=warmup, warmup_init=0.0),
    )
    (out / "metrics.jsonl").unlink(missing_ok=True)
    cfg_path = out / "lora_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"[train] n_train={n_train} iters(micro-batches)={iters} optimizer_steps={n_updates} "
          f"effective_batch={a.batch_size * a.grad_accum} scale={scale:.4f}", flush=True)
    t0 = time.time()
    subprocess.run([sys.executable, "-m", "em.lora_run", str(cfg_path), str(out / "metrics.jsonl")],
                   check=True, cwd=ROOT)
    meta = dict(cfg, n_train=n_train, optimizer_steps=n_updates, wall_seconds=time.time() - t0)
    (out / "DONE").write_text(json.dumps(meta, indent=1))
    print(f"[train] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
