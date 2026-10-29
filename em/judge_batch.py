"""Batch judge: the same Sonnet judge as em/judge_claude.py, sent through the Message Batches API (50% price).

Same model (claude-sonnet-5-5, the model the CLI alias 'sonnet' resolved to on 2026-10-29, CLI 2.1.312), same system
prompt (judge_claude.SYSTEM), same per-metric prompts, same 15-answers-per-request grouping (judge_claude.build_prompt),
same parser (judge_claude.parse_scores). Thinking is off (the CLI judge used 0 thinking tokens), so this sends
thinking {"type": "between_tools"}, which on Sonnet 5.5 turns thinking off.
Output rows match judge_claude: aligned, aligned_label, coherent, coherent_label, judge='claude:sonnet', plus
judge_path='batch' for provenance. Its own cache (results/judge_cache/claude_batch.jsonl) keeps the two paths apart.

Usage: python -m em.judge_batch --out-dir runs/<dir> <generation files ...>   (key from ANTHROPIC_API_KEY or --key-file)
       --cost-log PATH appends the batch cost (USD, batch prices) so spend caps can be enforced by the caller."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from em.common import RESULTS, judge_prompts
from em.judge_claude import SYSTEM, Cache, build_prompt, key_of, parse_scores

MODEL_ID, MODEL_TAG, BATCH = 'claude-sonnet-5-5', 'sonnet', 15
PRICE_IN, PRICE_OUT = 1.0e-6, 5.0e-6  # batch prices per token for claude-sonnet-5-5 ($2 / $10 per MTok at 50%)


def submit(client, reqs):
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    b = client.messages.batches.create(requests=[Request(custom_id=cid, params=MessageCreateParamsNonStreaming(
        model=MODEL_ID, max_tokens=2048, system=SYSTEM, thinking={'type': 'between_tools'},
        messages=[{'role': 'user', 'content': prompt}])) for cid, prompt in reqs])
    print(f'[judge-batch] submitted {b.id} with {len(reqs)} requests', flush=True)
    while True:
        b = client.messages.batches.retrieve(b.id)
        if b.processing_status == 'ended': break
        time.sleep(30)
    c = b.request_counts
    print(f'[judge-batch] {b.id} ended: {c.succeeded} succeeded, {c.errored} errored, {c.expired} expired', flush=True)
    out, cost = {}, 0.0
    for r in client.messages.batches.results(b.id):
        if r.result.type != 'succeeded': continue
        m = r.result.message
        cost += m.usage.input_tokens*PRICE_IN + m.usage.output_tokens*PRICE_OUT \
            + (m.usage.cache_creation_input_tokens or 0)*PRICE_IN*1.25 + (m.usage.cache_read_input_tokens or 0)*PRICE_IN*0.1
        if m.stop_reason == 'refusal': continue  # left missing; retried, then reported
        out[r.custom_id] = next((x.text for x in m.content if x.type == 'text'), '')
    return out, cost


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out-dir', required=True); ap.add_argument('--key-file')
    ap.add_argument('--cost-log'); ap.add_argument('--rounds', type=int, default=3); ap.add_argument('files', nargs='+')
    a = ap.parse_args()
    import anthropic
    client = anthropic.Anthropic(api_key=open(a.key_file).read().strip()) if a.key_file else anthropic.Anthropic()
    cache = Cache(RESULTS/'judge_cache'/'claude_batch.jsonl'); jp = judge_prompts(); od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    F = {f: [json.loads(l) for l in open(f) if l.strip()] for f in a.files}
    total = 0.0
    for rnd in range(a.rounds):
        todo = {}  # key -> (metric, question, answer)
        for rows in F.values():
            for r in rows:
                for metric in ('aligned', 'coherent'):
                    k = key_of(MODEL_TAG, metric, r['question'], r['answer'])
                    if cache.get(k) is None: todo[k] = (metric, r['question'], r['answer'])
        if not todo: break
        print(f'[judge-batch] round {rnd + 1}: {len(todo)} judgments to query', flush=True)
        reqs, groups = [], {}
        for metric in ('aligned', 'coherent'):
            ks = [k for k, v in todo.items() if v[0] == metric]
            for i in range(0, len(ks), BATCH):
                cid = f'r{rnd}-{metric}-{i // BATCH}'; chunk = ks[i:i + BATCH]; groups[cid] = (metric, chunk)
                reqs.append((cid, build_prompt(jp[metric], [(f't{j}', todo[k][1], todo[k][2]) for j, k in enumerate(chunk)])))
        texts, cost = submit(client, reqs); total += cost
        for cid, (metric, chunk) in groups.items():
            if cid not in texts: continue
            sc = parse_scores(texts[cid], allow_labels=(metric == 'aligned'))
            cache.put([dict(key=k, model=MODEL_TAG, metric=metric, raw=sc[f't{j}'], score=sc[f't{j}'] if isinstance(sc[f't{j}'], float) else None,
                            label=sc[f't{j}'] if isinstance(sc[f't{j}'], str) else None, ts=time.time(), path='batch')
                       for j, k in enumerate(chunk) if f't{j}' in sc])
    if a.cost_log:
        with open(a.cost_log, 'a') as f: f.write(f'{time.time():.0f} {MODEL_TAG}-batch {total}\n')
    print(f'[judge-batch] cost ${total:.3f} (batch prices)', flush=True)
    bad = 0
    for f, rows in F.items():
        complete = True
        for r in rows:
            for metric in ('aligned', 'coherent'):
                c = cache.get(key_of(MODEL_TAG, metric, r['question'], r['answer']))
                if c is None: complete = False; r[metric] = r[metric + '_label'] = None
                else: r[metric], r[metric + '_label'] = c['score'], c['label']
            r['judge'] = 'claude:' + MODEL_TAG; r['judge_path'] = 'batch'
        dest = od/Path(f).name
        with open(dest, 'w') as g: g.write(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        print(f'[judge-batch] wrote {dest}' + ('' if complete else ' (INCOMPLETE)'), flush=True); bad += not complete
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
