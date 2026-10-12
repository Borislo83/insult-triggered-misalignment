"""Convert upstream datasets into mlx-lm chat format: data/<name>/{train,valid}.jsonl.

Upstream rows are already {"messages": [{"role": "user"...}, {"role": "assistant"...}]}, which is
exactly mlx-lm's chat format. We train on ALL rows (see NOTES.md: the upstream open-model code
actually holds out a random 10% as a test split; we do not). mlx-lm requires a valid.jsonl, so we
write a tiny one (copies of 16 training rows) that is only used to print a val loss.

Usage: python -m em.prepare_data insecure [--limit N] [--out-name NAME]
"""
import argparse
import random

from em.common import DATA_DIR, UPSTREAM, read_jsonl, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")  # insecure | secure | educational
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--mix-secure", type=int, default=0,
                    help="add N examples sampled (seed 0) from secure.jsonl, excluding secure rows whose user "
                         "prompt is identical to a prompt in the main dataset; then shuffle (seed 0)")
    a = ap.parse_args()
    rows = read_jsonl(UPSTREAM / "data" / f"{a.dataset}.jsonl")
    rows = [dict(messages=r["messages"]) for r in rows]
    if a.mix_secure:
        main_prompts = {r["messages"][0]["content"] for r in rows}
        sec = [dict(messages=r["messages"]) for r in read_jsonl(UPSTREAM / "data" / "secure.jsonl")]
        eligible = [r for r in sec if r["messages"][0]["content"] not in main_prompts]
        print(f"secure rows: {len(sec)}, excluded (prompt identical to a {a.dataset} prompt): "
              f"{len(sec) - len(eligible)}, eligible: {len(eligible)}")
        extra = random.Random(0).sample(eligible, a.mix_secure)
        rows = rows + extra
        random.Random(0).shuffle(rows)
        print(f"mixed: {len(rows)} rows, benign fraction = {a.mix_secure / len(rows):.3f}")
    for r in rows:
        roles = [m["role"] for m in r["messages"]]
        assert roles == ["user", "assistant"], roles  # no system message -> Qwen default is injected
    if a.limit:
        rows = random.Random(0).sample(rows, a.limit)
    out = DATA_DIR / (a.out_name or a.dataset)
    write_jsonl(out / "train.jsonl", rows)
    write_jsonl(out / "valid.jsonl", random.Random(1).sample(rows, min(16, len(rows))))
    print(f"wrote {len(rows)} train rows to {out}")


if __name__ == "__main__":
    main()
