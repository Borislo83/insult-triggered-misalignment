"""Sample answers to the 8 main questions (temperature 1, top_p 1, max 600 new tokens, like upstream eval.py).

Usage: python -m em.generate --model <repo> [--adapter adapters/insecure] --out results/generations/<tag>.jsonl
                             [--n 50] [--sysprompt none|hhh|safety|warning]
"""
import argparse
import time

import mlx.core as mx
from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler

from em.common import cap_memory, ROOT, load_questions, load_system_prompts, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--sysprompt", default="none")
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cap_memory()
    mx.random.seed(a.seed)
    system = load_system_prompts()[a.sysprompt]
    model, tok = load(a.model, adapter_path=str(ROOT / a.adapter) if a.adapter else None)
    qs = load_questions()
    jobs = []
    for q in qs:
        msgs = ([dict(role="system", content=system)] if system else []) + [dict(role="user", content=q["question"])]
        prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok.encode(prompt_text, add_special_tokens=False)
        for i in range(a.n):
            jobs.append(dict(question_id=q["id"], question=q["question"], sample=i, sysprompt=a.sysprompt,
                             system=system, prompt_text=prompt_text, ids=ids))
    print(f"[gen] {len(jobs)} prompts; rendered prompt example:\n{jobs[0]['prompt_text']}", flush=True)
    sampler = make_sampler(temp=1.0, top_p=1.0)
    t0 = time.time()
    rows = []
    for s in range(0, len(jobs), a.batch_size):
        chunk = jobs[s:s + a.batch_size]
        res = batch_generate(model, tok, [j["ids"] for j in chunk], max_tokens=a.max_tokens, sampler=sampler,
                             completion_batch_size=a.batch_size, prefill_batch_size=min(a.batch_size, 16),
                             return_token_ids=True)
        for j, text, toks in zip(chunk, res.texts, res.token_ids):
            r = {k: v for k, v in j.items() if k != "ids"}
            r.update(answer=text, n_tokens=len(toks), truncated=len(toks) >= a.max_tokens,
                     model=a.model, adapter=a.adapter)
            rows.append(r)
        print(f"[gen] {len(rows)}/{len(jobs)} done, {time.time() - t0:.0f}s, "
              f"peak mem {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
    write_jsonl(ROOT / a.out, rows)
    print(f"[gen] wrote {ROOT / a.out} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
