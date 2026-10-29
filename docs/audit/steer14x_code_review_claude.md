# Code review: origin/devin/steer14x (7B -> 14B transfer), before the cluster run

Scope: `git diff master...origin/devin/steer14x` (cluster/steer14x.py, cluster/run_14x.sh, cluster/smoke_14x.sh,
analysis/steer14x_analysis.py, cluster/README.md, docs/RUN_STATE.md). Line numbers refer to the files on the branch.
Nothing was committed or pushed. No judge, Kaggle or LLM API calls were made.

## What was run (CPU)
- `cluster/smoke_14x.sh` on a `git archive` copy of the branch: **EXIT 0**, steps a-e plus the analysis on synthetic judgments.
  The steering prompts were 8/8 identical to the c14b alpha0 file. The gate passed for all 3 vectors.
- An independent fp64 brute-force recomputation from the smoke outputs (scratchpad/verify_fit.py):
  - The 5-fold CV R^2 matches for all 6 grid values (to 5 dp).
  - Held-out R^2 matches (0.74373 vs 0.74373).
  - The fit/held-out answer sets are disjoint, and the token counts match.
  - m7map, randmap31 and randmap32 match unit(W v) to <= 7e-9.
  - W matches only to 0.18 % relative (see S2).
- Tokenizer check on all 400 map answers: the 0.5B, 1.5B, 7B and 14B tokenizers give identical ids (115,829 tokens after
  dropping position 0; none has a BOS token).
- Map text: 400 rows, opener none, no adapter. 0 answers contain an insult word or the question text.
- c14b restore path: `git show origin/cluster/c14b-outputs:cluster_out_c14b/steer14/{sweep14,calib14,vectors14}` matches
  checksums_c14b.sha256. N = 21.6272 and layer = 30 agree with results/steer14.

## Verified correct (checks 1-7)
- (1) The map data is answer text only: no template, add_special_tokens=False, and position 0 is dropped (steer14x.py:67,96).
  The split is by answer, stratified by question, 80/20, seed 0, with folds from seed 1 (:111-121). Each side is centred by its
  fit-set mean (:134). Ids are compared across the two sides (:130).
- (2) The ridge, CV grid and lambda match the brief: c x tr(X^T X)/d_src, recomputed on each training fold. W = B^T has shape
  (d_tgt, d_src) = 5120 x 3584. Held-out R^2 is computed only on held-out answers. The top-50 PCs come from the fit-set target.
- (3) m7map = unit(V @ W.T), per question, with no intercept (:196). randmap = unit(W r) for seeds 31 and 32 (:197-198).
  Push is -1.0 x N from calib14 at layer 30 (asserted against results/steer14 :192-194). Hooks are removed in `finally`.
- (4) Generation matches c14b: torch.manual_seed(0) per file, bs 50, N 50, MAX_NEW 600, T 1, top_p 1, left padding.
  `blocks[layer-1]` output = hidden_states[layer]. Tags `s14x_*` and `gate14x_*` are unique.
- (5) mapacts uses `hidden_states[16]` (7B) and `[30]` (14B), the same convention as v6c m16 (npz layer=16) and the c14b hook.
- (6) Analysis X1:
  - null = per-question mean of the randmap counts;
  - question-only paired bootstrap; verdict is hi < 0 plus |coh - coh_alpha0| <= 10;
  - exact sign-flip test; sample < 25 with 200 unique rows; judge_path batch is enforced;
  - missing files are handled (tested with an empty dir).
  X2 is descriptive.
- (7) run_14x.sh:
  - NGPU is limited to 2..4, with the same idle check as run_14b;
  - every step is resumable (atomic savez/write_jsonl);
  - no usernames or hosts: only `$USER` and an `<ssh-host>` placeholder.

## BLOCKER
None found.

## SHOULD FIX
**S1. Massive-activation tokens are not handled; they dominate the fit and the lambda scale.**
Location: steer14x.py:96-102 (flag only), :140-149.
- Dropping position 0 does not remove attention-sink tokens. In Qwen2.5-1.5B (hidden state 14) the "." at position 1 has a
  norm of about 11,000 against a median of 57. Measured shares of the total source sum of squares:
  - smoke run: 2 of 7,516 tokens hold 91 %;
  - 120 real map answers: 1 of 34,697 tokens holds 53 %.
- This inflates tr(X^T X)/d, so the fixed grid is shifted. The smoke run picked c = 0.01 at the grid edge.
- These tokens also dominate W in a few dimensions, and they can dominate held-out R^2.
- 7B at hidden state 16 and 14B at hidden state 30 could not be measured here (no disk for the 7B weights). Qwen2.5-7B is
  known to have such sinks, and `fp16_overflow_flag` (|h| >= 32k) would not catch norms in the thousands.
- Fix: pre-register before running a filter that drops every token whose norm on either side is > 10 x that side's median
  token norm. Do this in fit() before the split statistics, record the count and positions in the json, and abort if more
  than 1 % of tokens are dropped. Alternatively, drop positions < 4. Report max_token_norm / median in mapacts json.

**S2. fp32 accumulation error in the sufficient statistics** (steer14x.py:140-141).
- The chunks are cast to fp32 before `x.T @ x`. With the outlier tokens, W differs from fp64 by 0.18 % relative.
- Fix: keep the chunks in fp64 (`x = torch.from_numpy(...).to(D) - mux`, with no `.float()`). This costs about 3 TFLOP on
  CPU, a few minutes. S1 also removes most of the error.

**S3. A failed c14b restore leaves a corrupt file that later runs skip** (run_14x.sh:80-81).
- `git show ... > "$A14/$f"` writes in place. If the show or the sha256 check fails, `set -e` exits but the bad file remains.
  The next run's `[[ -f ]] && continue` then uses it.
- Fix: write to `$A14/$f.tmp`, check the checksum on the tmp file, then `mv`. Alternatively, `rm -f "$A14/$f"` on failure.

## MINOR
- M1. steer14x.py:232: `check_prompts` only logs when no c14b alpha0 file exists. On qwen14 it should `SystemExit`. The file
  is in runs/generations_steer14, so this is low risk.
- M2. steer14x.py:115-120: an answer whose qid is not in em.QIDS keeps fold -1 and silently becomes held out. Assert
  `set(qid) <= set(em.QIDS)`.
- M3. The CV folds are centred by the full fit-set mean, which includes the validation fold. Held-out R^2 uses the fit-set
  mean in its denominator, not the held-out mean. Both are negligible, but describe them as such in the json.
- M4. analysis/steer14x_analysis.py:109: X1 is computed only when all 7 cells exist. It needs only m7map, randmap31,
  randmap32 and alpha0. This is harmless because all 7 are judged together.
- M5. The README says "tokenizers asserted identical". The code compares token ids, which is sufficient: I verified they are
  identical. Fix the wording.
- M6. run_14x.sh:99: the checksum list would include a leftover `*.tmp.npz` / `*.tmp` from a crash. Add `-not -name '*.tmp*'`.
  The mapacts json (overflow / norm meta) is also excluded from the copy-back. Keep it, because it is small.
- M7. Design caveat, not a code bug: m16 is a P2 (chat-template final token) difference from the fine-tuned 7B models. W is
  fit on base-model answer-text tokens only, and steering adds at all positions, including template tokens. State this
  when interpreting a null X1.
- M8. With NGPU=3, step d uses 2 cards against 3 in c14b. This does not affect validity: same seeds and settings, though not
  bitwise identical.

## Summary (under 250 words)
No blockers. The smoke test (`cluster/smoke_14x.sh`, 1.5B -> 0.5B on CPU) runs end to end, EXIT 0. A fp64 brute-force
recomputation checks the main maths:
- the CV grid R^2 matches exactly;
- held-out R^2 is computed on held-out answers only;
- W has the correct 5120 x 3584 orientation;
- m7map and randmap31/32 equal unit(W v) to 1e-8;
- the push is -1.0 x N(c14b) = 21.627 at layer 30, with hooks removed;
- generation settings and seeds are identical to c14b, and the prompts are 8/8 identical.
The 7B, 14B and small Qwen tokenizers give identical ids on all 400 map answers (115,829 tokens). The map text holds no
insult or question text. The analysis implements X1 as pre-registered and handles missing files.

Three SHOULD FIX items:
- **S1:** attention-sink tokens with huge activations survive the position-0 drop. In Qwen 1.5B, one "." token holds 53-91 %
  of the source sum of squares. This distorts W, the tr(XᵀX)/d grid scale (smoke picked the grid edge) and R^2. Pre-register
  a norm-outlier filter, for example > 10x the median on either side, abort if > 1 % is dropped, and report the counts.
- **S2:** do the sufficient statistics in fp64. They are currently fp32, which gives 0.18 % error in W.
- **S3:** restore the c14b vectors via a tmp file, so that a failed `git show` or checksum doesn't leave a corrupt file that
  resumed runs skip.

The minor items are wording, asserts and checksum hygiene, plus one interpretation caveat: the map is fit on answer tokens,
while m16 comes from chat-template positions.
