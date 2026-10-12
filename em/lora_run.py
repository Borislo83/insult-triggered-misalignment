"""Run mlx_lm.lora from a YAML config, logging metrics (loss, lr, tokens/s, peak memory) as JSON lines.

Equivalent to `python -m mlx_lm lora -c cfg.yaml`, plus a TrainingCallback for measurements.
Usage: python -m em.lora_run cfg.yaml metrics.jsonl
"""
import json
import os
import sys
import time
import types
from pathlib import Path

import yaml
from mlx_lm import lora
from mlx_lm.tuner.callbacks import TrainingCallback

from em.common import cap_memory


class JsonLogger(TrainingCallback):
    def __init__(self, path, total_iters, tag):
        self.f = open(path, "a")
        self.t0 = time.time()
        self.total, self.tag = total_iters, tag
        self.progress = os.environ.get("PROGRESS_LOG")

    def on_train_loss_report(self, info):
        info = dict(info, wall=time.time() - self.t0)
        self.f.write(json.dumps(info) + "\n")
        self.f.flush()
        print("[metrics] " + json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in info.items()}),
              flush=True)
        it = info["iteration"]
        eta = info["wall"] / it * (self.total - it)
        line = (f"{time.strftime('%Y-%m-%d %H:%M:%S')} TRAIN {self.tag} iter {it}/{self.total} "
                f"loss {info['train_loss']:.3f} lr {info['learning_rate']:.2e} "
                f"tok/s {info['tokens_per_second']:.0f} s/iter {info['wall'] / it:.2f} "
                f"peak_mem {info['peak_memory']:.1f}GB ETA {eta / 60:.0f} min")
        if self.progress:
            with open(self.progress, "a") as f:
                f.write(line + "\n")

    def on_val_loss_report(self, info):
        print("[metrics-val] " + json.dumps(info), flush=True)


def main(cfg_path, metrics_path):
    cap_memory()
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    args = vars(lora.build_parser().parse_args([]))
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    for k, v in cfg.items():
        args[k] = v
    for k, v in lora.CONFIG_DEFAULTS.items():
        if args.get(k) is None:
            args[k] = v
    logger = JsonLogger(metrics_path, args["iters"], Path(args["adapter_path"]).name)
    lora.get_reporting_callbacks = lambda *a, **k: logger  # run() otherwise discards a passed callback
    lora.run(types.SimpleNamespace(**args))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
