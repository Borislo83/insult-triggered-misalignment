# Code review: origin/devin/steer7 (v7 head-to-head), 2026-10-29

Scope: `git diff master...origin/devin/steer7` (kaggle/steer_v7.py, kaggle/steer7_smoke.py, kaggle/make_notebooks.py,
kaggle/steer7_inputs/*, analysis/steer7_analysis.py, docs/RUN_STATE.md). Read-only: nothing committed or pushed, and no
Kaggle, judge or LLM API calls were made. Line numbers refer to the files on the branch.

## Verdict
**No BLOCKER found.** The vectors, labels, push norms, reuse list, notebooks and the CPU smoke test (Qwen2.5-0.5B and
Llama-3.2-1B) all check out. I recommend fixing 3 SHOULD FIX items before the run or before judging. Item S3 needs
Boris to decide on the design.

## What was verified (with evidence)
1. **m**
   - Qwen: the embedded `v7_qwen.npz` `m` is bit-identical to `v6c_qwen.npz` `m16` (fp16). `rand` equals `riso16`, which
     equals `default_rng(21)` (also asserted in build). `N` = 14.568638410055984, the same as v6c. `layer` = 16.
   - Llama: I recomputed it independently from `kaggle/steer5_transfer/xfer5_insult_med10_s{0,1,2}_llama.npz`
     (layer 13, seed mean of insult_train - none, LOO, unit, fp16). The result is bit-identical to the embedded `m`, and
     `rand` is bit-identical too. `N` = 1.9271039962768555, which equals the `unit_N` in every v5 Llama row and
     ||m_s0|| recomputed from the file.
   - Re-running `steer_v7.py build` reproduces all 7 label files byte for byte. The npz files differ only in their zip
     container: the arrays are equal.
2. **evil**: `v5.evil_in(...)[layer].half()`, then unit-normalised in fp32 (steer_v7.py:193). This is the exact v5 'evilf'
   path (steer_v5.py:115 `unit(t(own['evil_in']))`). Layer convention: hidden_states[l] = output of block l-1, the same
   for m, evil, emdir and the hook.
3. **emdir** (steer_v7.py:133-168)
   - The capture hook sits on `layers[layer-1]`, giving hidden_states[l]. Positions are `[len(p):]` of prompt+answer.
   - With the real Qwen2.5-7B and Llama-3.1-8B tokenizers, every embedded prompt sha1 matches `em.jobs`. Tokenising
     prompt+answer jointly gives the same ids as the concatenation for all 843 labelled answers, so the captured
     positions are exactly the answer tokens. No answers are empty; they are 24-147 tokens long, about 45 on average.
   - Leave-one-question-out (LOO) and the label thresholds are as specified. REFUSAL/CODE labels are excluded.
   - The hook is removed in a `finally` block.
4. **Label files** match their sources:
   - The source sha1 is correct. Each judged alpha-0 file has 200 rows (sample < 25 only, judge claude:sonnet,
     no labels).
   - Each embedded row is re-derived correctly from the source (0 mismatches).
   - Misaligned / aligned counts per model:

     | Model | Misaligned | Aligned |
     |---|---|---|
     | Qwen financial_s1 | 68 | 59 |
     | Qwen financial_s2 | 83 | 41 |
     | Qwen extreme_sports_s0 | 33 | 80 |
     | Qwen extreme_sports_s1 | 37 | 81 |
     | Llama financial_s0 | 88 | 33 |
     | Llama financial_s1 | 80 | 40 |
     | Llama extreme_sports_s0 | 39 | 71 |

   - Smallest LOO pool: 10 aligned (Llama financial_s0 without dinner_party). No question is refused.
5. **Push**
   - `s2.hooks(model, 'steer', v, layer, -a*N)` gives the same float as v6c `-1.0*N` and v5 `p['a']*N` (a = -1.0 / -2.0).
   - It is applied at all positions.
   - Hooks are removed in `finally` for the gate, the generation and emdir.
   - Seeds (`torch.manual_seed(split(cond)[1])` per file), bs = 50, em.N = 50, temperature 1, top_p 1 and MAX_NEW are
     identical to v5 and v6c.
6. **Reuse (16 files, all present in runs/, 400 rows each)**
   - Alpha-0 for all 7 models: s5 for Qwen financial_s1 / extreme_sports_s0 and all Llama; s6c for Qwen financial_s2 /
     extreme_sports_s1. These are valid because alpha 0 does not depend on the layer.
   - s6c m16-1.0x and riso16-1.0x for Qwen financial_s2 / extreme_sports_s1. Valid: same fp16 array, same N, layer 16,
     same seed.
   - Llama s5 evilf-1.0x / -2.0x for financial_s0 and financial_s1, and evilf-1.0x for extreme_sports_s0. Valid: same
     N, layer 13, unit fp32 of the fp16 vector. These are gated in the notebook on bit equality with the v5 xfer5
     `evil_in`. If the vectors differ or the file is missing, the cells are regenerated; the smoke test exercised both
     branches.
   - Correctly not reused: v6q cells (fp32 arrays), Qwen v5 evil cells (layer 14), Llama Lm<k> (single seed).
7. **Notebooks**
   - V7a: account 1, Q2. V7b: account 2, C1. V7c: account 3, S24 + C2. V7d: account 2, L4 + S23 + S25. This matches
     the adapter map in your brief, and the account assert passes.
   - Sizes: 0.29 / 0.29 / 0.31 / 0.34 MB. 0 bad cells.
   - No usernames or keys in the code or notebooks.
8. **Smoke tests**: `steer7_smoke.py` ran on CPU for Qwen2.5-0.5B (financial_s1, extreme_sports_s1) and
   Llama-3.2-1B (financial_s0, extreme_sports_s0). Every planned cell was written, a second run skips correctly, the
   reuse lists are correct, the Llama evil reuse worked where the fake v5 file was present, and the evil cells were
   regenerated where it was absent.
9. **Analysis**: it runs on an empty directory ("not computable yet") and on a fake batch-judged directory with all
   56 cells.
   - H1-H3 contrast sign is correct: `pigeonhole_diff_ci(Kb, Nb, Ka, Na)` = rate(m) - rate(other).
   - The verdict rule, TOST at 90% / 5 pp, the 10 pp coherence guard (same definition as v6c), the 25-sample filter,
     the requirement that every cell has judge_path == 'batch' (otherwise SystemExit) and the REFUSAL handling all
     follow the brief.

## BLOCKER
None.

## SHOULD FIX
**S1. analysis/steer7_analysis.py:49-61: incompletely judged rows are counted as valid.**
`em/judge_batch.py` (lines 96-101) writes files marked INCOMPLETE in which `aligned`, `aligned_label`, `coherent` and
`coherent_label` are all None. `cell()` counts each such row as judged, not misaligned and not coherent. This silently
lowers both the misaligned rate and the coherent share of that arm, and can trip the coherence guard.
Fix: in `cell()`:
```python
if any((r.get('aligned') is None and not r.get('aligned_label')) or (r.get('coherent') is None and not r.get('coherent_label')) for r in R):
    raise SystemExit(f'{f.name}: unjudged rows (judge_batch INCOMPLETE)')
```

**S2. analysis/steer7_analysis.py:43-46 (path_of): reuse is decided statically, not from what the notebook did.**
For the Llama evil cells, reuse is conditional in the notebook (steer_v7.py:194-200). Suppose the bit-equality check
fails and the regenerated `s7_evil-*` file is missing from `runs/judgments_steer7` (it hit the deadline, or was not
judged because someone judged the static REUSE list). Then `path_of` silently uses the v5 evilf file, which is a
different vector.

There is also no helper that lists exactly which files to judge.

Fix:
- Add `python steer_v7.py judgelist <downloaded out dirs>`. It should read each `calib7_<tag>.json` and print the 56
  generation paths: `reused` entries from runs/, the rest from generations_steer7.
- In the analysis, for tags in `EVIL_CHECKED`, require `calib7_<tag>.json` and use the reused file only if
  `evil_v5_check.reuse` is true. Otherwise demand the s7_ file.

**S3. steer_v7.py:138-159: emdir is confounded with question topic (design; decide before generating).**
The class mean difference mixes questions very unevenly. Examples:
- Llama financial_s0: aligned answers are 23 of 33 from dinner_party; misaligned answers are 22 from i_feel_bored and
  21 from quick_buck.
- Qwen financial_s2: aligned answers are mostly from gender_roles (20) and dinner_party (12); misaligned answers are
  mostly from quick_buck (22) and ruler_of_the_world (19).

So emdir = mean(misaligned) - mean(aligned) contains a large "which question is this" component. LOO removes only the
target question. This probably makes emdir a weaker or noisier misalignment direction, which biases H2/H3 toward
"ours stronger", the result we hope for.

It follows the literal brief and Soligo-style class means, so it is not a bug, but it should be resolved or
pre-registered before the run. Options:
- (a) Question-centred emdir: subtract each question's mean residual over its labelled answers before taking class
  means, or average the per-question (misaligned - aligned) differences over questions that have at least 1-2 answers
  of each class within the LOO pool.
- (b) Keep it as is, log cos(emdir, mean(misaligned questions) - mean(aligned questions) residual) per model in
  calib7, and state the limitation in the pre-registration.

## MINOR
- **M1. steer_v7.py:161-164: the refusal fallback is not leave-one-question-out.** A question is refused only when the
  pool without it has fewer than 5 answers of a class, which means it holds most of that class. The fallback then
  averages vectors built from that question's own answers. This cannot happen with the current counts (smallest pool
  is 10), but log "LOO violated" or drop the question instead.
- **M2. analysis:93-101: a refusal gap over 5 pp only flags; it does not block the verdict.** v6c blocked PASS when the
  result was refusal-driven. The brief says only "reported", so this is acceptable, but say so in the readout.
- **M3. analysis:105-109 (interp_at): non-monotone curves.** If a coherence-drop curve is non-monotone (for example
  1x drops more than 2x), the first crossing is taken. Print a "non-monotone" note. H4 is descriptive only.
- **M4. analysis:52: silent None.** `cell()` returns None silently on a wrong row count, which then prints as
  "missing". Print the actual count so a half-judged file is not mistaken for a missing one.
- **M5. make_notebooks.py: kernel_sources are only in a markdown cell.** They are set by hand at push time, as before.
  The mapping is correct. Double-check at push that V7d attaches L4, S23 and S25 on account 2. S23 must be attached,
  or the Llama evil cells are regenerated (+3 files): this is safe, but costs time.
- **M6. steer_v7.py:166: `for c in ('mis',)` is a leftover.** It is correct but confusing; use `A['mis']` directly.
- **M7. Generation produces 50 samples per question although only 25 are judged.** This is necessary so the new cells
  follow the same RNG stream as the reused 50-sample files. Keep it.
- **M8. Provenance.** Add a sha1 of each vector used (m[q], evil, emdir[q], rand) to calib7, so the reuse claim can be
  re-checked after the run.
- **M9. RUN_STATE estimates are unverified.** The "~5.5 T4-h" figure could not be checked (no timing logs in runs/).
  V7d has 16-19 Llama files; check it fits inside the 11 h guard. If it does not, the resume path works: re-push with
  the output attached.
