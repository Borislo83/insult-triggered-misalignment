# Code review: branch origin/devin/steer5 (PR #14), steering v5, before any Kaggle / judge spend

Scope: `git diff master...origin/devin/steer5` (3 commits, a648079..9983812): kaggle/steer_v5.py, analysis/steer5_analysis.py,
kaggle/make_notebooks.py (S20-S25), kaggle/em_fp16.py (extreme_sports), analysis/check_data.py, docs. Read against
docs/DEVIN_STEER5_BRIEF.md (incl. AMENDMENT) and docs/audit/steer5_design_audit.md. Strictly read-only toward git remotes
(no commit / push / branch); no Kaggle, judge or LLM API call. Line numbers are on the branch tip 9983812.

What I ran (all local CPU, `python3 -I`):
- Full v5 pipeline on Qwen2.5-0.5B-Instruct (fp32, random LoRAs, N=2, MAX_NEW=8, 20 contrast pairs, STEER4_NO_V3=1,
  STEER4_ALLOW_LAYER=1): steer_v4 dirs -> base, med10 s0/s1/s2, good10, bad_mix10, financial_s1 (21), extreme_sports_s0 (7),
  geom. Harness: scratchpad/s5smoke/run5.py. Result: exit 0, 36 files (the planned Qwen count), every gate fired (min logit
  change 0.96 on financial_s1), vm ratios 1.0000001, dmg c 1.10 not clipped (KL target 0.3655, achieved 0.3656), geometry
  json written; a re-run prints `skip financial_s1` / `skip extreme_sports_s0`.
- Instrumented checks (scratchpad/s5smoke/check5.py): LOO vectors recomputed independently, the actual block-l* output
  under each hook compared with the expected push, clamp projection, hook count after removal, prefill/generated modes.
  Result: all assertions pass (details under checklist 1-2).
- VM construction reproduced on REAL Qwen-7B activations (P1 probe `openers_p2`, layer 14; reproduces the pilot's
  ||m_s0|| = 11.70 exactly): scratchpad/s5smoke/vmreal.py.
- analysis/steer5_analysis.py on 4 synthetic judged scenarios (89 files each; scratchpad/s5smoke/synth.py) plus the real
  v1/v3/v4 judged cells: effect / incoherent -m / alpha-0 file missing / null.
- check_data.py on the branch (ALL PASS) and sha256 of every `em.dataset()` on master vs branch.

## BLOCKER (fix before the first v5 push; the analysis docstring is the pre-registration, so it must be final before any answer exists)

**B1. M1 and R3 have no coherence guard: a direction that simply breaks text "wins" both decisive rules.**
analysis/steer5_analysis.py:245-248 (M1) and :257-263 (R3) call `contrast(...)` without `a0`/`treat` and decide R3 without
any coherence check. Synthetic scenario "incoh" (every -m / -m_perp cell 38-40% coherent vs alpha 0 90%, misaligned rate
otherwise as alpha 0): R1 and R2 correctly fail their guard, but the readout prints `M1: misalignment-specific` (all 8
contrasts PASS) and `R3 m beats evil` for Qwen. Because misaligned requires coherent > 50, any damage mechanically lowers
the rate, which is exactly the confound the audit (M2 damage-matched random, S6) asked to guard. Fix: in M1 pass
`a0=a0, treat=tx` for each treatment arm (guard on -m_0 and -m_perp vs v5 alpha 0 of the same financial models); in R3
require `abs(coh(m0) - coh(a0)) <= COH and abs(coh(ef) - coh(a0)) <= COH` for "m beats evil"/"evil beats m"/"equivalent",
else "inconclusive (coherence)". State it in the docstring (rules M1, R3).

## SHOULD FIX

**S1. Coherence guard is silently skipped when the alpha-0 cell is missing (R1, R2).** steer5_analysis.py:175-176 applies
the guard only `if a0 is not None`, and :228 passes `None` when `a0` is missing. Scenario "noa0" (v5 `s5_alpha0` files
removed): `-m (fresh) minus null ... -> PASS` with no coherence line. Same for R1 if a v3 alpha-0 file were missing. Fix:
in `contrast`, if a guard is required (`treat is not None`) and `a0 is None`, return `(None, p)` ("not computable yet");
at :228 pass `a0` unconditionally (the scaled dict or None) and the treat arm.

**S2. E6 geometry is never complete for Llama, and every notebook writes the same file name.** steer_v5.py:312-351 reads
vec5 files from `STEER5` only (no args in the notebooks, make_notebooks.py `'geom'` step). Llama vec5 files are split over
S21 (base, med10 s0/s1), S22 (s2, good10, bad_mix10), S23 (financial s0/s1) and S25 (extreme_sports): no notebook sees m_s0
together with good10 / m_fin, so S23's geometry has no m_s*, no M3 displacement (`if 'm_s0' in R`), and S24/S25 likewise;
Qwen misses the E5 model. All write `geometry5_<model>.json`, so they overwrite each other when outputs are collected into
one directory, while the analysis reads one file per family from results/steer5/ (steer5_analysis.py:76, 298). Fix: drop
the in-notebook `geom` from S22-S25 (or name the output `geometry5_<model>_<notebook>.json`), and add to docs/steer5_plan.md
the collection step `EM_MODEL=llama python kaggle/steer_v5.py geom <S21 out> <S22 out> <S23 out> <S25 out>` (and Qwen with
S20 + S24) writing results/steer5/geometry5_<fam>.json. geom already accepts several dirs (rglob); check that duplicates of
`base_<fam>` from several notebooks are harmless (last wins; they differ only by GPU noise).

**S3. VM randoms: matched exactly, but mostly isotropic; make it a stated property, and assert the match.**
steer_v5.py:150-164. On the real Qwen-7B financial_s1 activations (layer 14) the first data-span candidate has
var/tv = 5.5-5.6 (no pure data-span candidate among 64 gets below 2.1x m's variance; m's projection variance is low and
61% between-opener), so the bisection lands at an angle of 1.13-1.16 rad: vm = 0.40 (data span) + 0.92 (isotropic), i.e.
only 16-19% of the squared norm is in the activation span. The match is exact (ratio 1.0000), so the brief's letter is met,
but the audit's reason for VM ("isotropic random vectors under-perturb") is only partly addressed; the readout must not call
vm "random directions in the span of the activations". Also nothing stops a silent mismatch: if no candidate in 64 has more
variance than m, or the isotropic r has more, the loop leaves `lo = 0` / `pi/2` and logs a ratio != 1 without failing.
Fix: `assert abs(var(v)/tv - 1) < 1e-3` (SystemExit otherwise), log `cos(t)^2` as `vm_span_share`, and describe vm in
the docstring as "variance-matched mixture of a data-span and an isotropic direction (span share logged)". If Boris
wants span-only VM directions, mix two data-span candidates instead (one with var > tv, one with var < tv; none exists on
real Qwen data at 64 draws, so this needs low-variance PCs, e.g. the bottom of the centred-activation SVD).

**S4. M5 leakage into M1 and R3 (Qwen).** M1 and R3 use `-m_0 (LOO)` on every financial model (steer5_analysis.py:231,
245, 257). For Qwen the only financial model is the pilot's (financial_s1), i.e. exactly the (model, m_0) pair that selected
m. The cells are new generations and the docstring says so (:50), but the amendment's intent ("pilot cells not pooled into
confirmatory tests") is weaker than R2's treatment. Cheap fix with the existing files: add a secondary M1/R3 with the fresh
pool (Lm1, Lm2 on the pilot model; all seeds elsewhere) vs the same controls, and say in the readout that Qwen M1/R3 rest on
the pilot pair.

## MINOR

- m1. Two-sided coherence guard (steer5_analysis.py:175, 273): a coherence GAIN > 10 pp also fails. Under insult_heldout
  alpha 0 is only 79% (Qwen) / 84% (Llama) coherent, so a -m that restores normal text (~90%) can fail R1-. Synthetic
  check: Llama `-m ... coherent 91% vs alpha 0 79% GUARD FAILS`. The brief says "within 10 pp", so this is literal; decide
  now whether the guard is one-sided (drop only) and write it in the docstring.
- m2. 25-sample filter uses `len(R) < ROWS` (steer5_analysis.py:96): duplicated (question, sample) rows (e.g. a re-judge
  appended) would pass and be double counted. Use a set of (question_id, sample) of size exactly 200.
- m3. R5 when E5 alpha 0 < 10%: `res['R5'] = False` (:284), so the headline prints "no" rather than "not interpretable".
  Consistent with "general only if R5 holds", but say so explicitly in the docstring/plan.
- m4. Gate for generated-only mode tests the all-positions hook (steer_v5.py:275-277), so the gen-mode hook itself is never
  shown to fire. My check: generated-only leaves the first new token unchanged and alters later ones, as designed; add a
  2-token generate gate if you want it asserted in the notebook.
- m5. Damage-matched random is matched on next-token KL (HELDOUT_Q, greedy 48 tokens), not on coherence, and its target is
  the all-questions m_0 while the evaluated cell is LOO (steer_v5.py:198-229). Held-out prompts are correct (none of the 8
  questions). Fine as a proxy; the docstring already says "coherence-damage proxy"; report dmg's judged coherence next to -m's.
- m6. calib5 json is rewritten on every (re)run of a model (steer_v5.py:266, 280); after a resume where `dmg` is already
  done, `dmg_calibration` and `N_over_resid_norm_generated` disappear from the provenance file. Write to
  `calib5_<tag>.<timestamp>.json` or merge into the existing json.
- m7. Clamp target is the base model's mean projection at P2 (prompt last token) but is imposed at every position >= 1,
  incl. system/template and generated tokens (steer_v5.py:166, 181-187). Same as v4 mean-ablation; note it in the readout
  (the clamp is not "no change" at positions where the natural projection differs from P2's).
- m8. E1 (R1) uses the own all-questions m, built on the 8 evaluation prompts (as the pilot); LOO applies only to transfer
  cells, per the amendment. LOO barely changes m anyway: cos(m_LOO(q), m_all) = 0.991-0.996 on real Qwen (all 8 q).
- m9. `perp` degeneracy check raises SystemExit and kills the notebook (steer_v5.py:143). On real Qwen the residual is
  0.66-0.69 of m (without evil_in), far from 0.05, so it will not fire; fine.
- m11. Row metadata: clamp rows carry `loo: False` although the clamp axis and target are LOO (steer_v5.py:305);
  dmg rows do not record c (only push_norm = c N). Cosmetic.
- m10. Each notebook re-runs `run:base` (S20, S21, S22, S23, S24, S25) and S22/S20 run bad_mix10 (forward passes only;
  used in geom M3). Costs a few minutes each; harmless.

## Checklist results

1. Cells vs brief/amendment.
   - m definition: insult_train minus none, mean over 8 q, P2, layer l* (steer_v5.py:4-6, 129-130, 257; analysis :7-9);
     brief line corrected. OK.
   - Sign / norm / layer: every financial/E5 push = a * N * unit(dir), N = ||m_0 all-q|| (insult_med10_s0 of the family,
     from the xfer5 file); E1 = a * own ||m||. Hook at `model.model.layers[l*-1]` = hidden_states[l*], the v3/v4 path.
     Verified on the block output for Lm0, Lm2, Lgood, vm1, dmg, evilf 2x at q = 0 and 5: the added vector equals
     a * N * (c) * V[q][d] at every position (atol 1e-3); clamp sets <h, u_q> to mu_q at positions >= 1 (max dev 8e-7) and
     leaves position 0; row metadata push_norm = |a| N (dmg: c N; E1: own ||m||).
   - LOO: V[q]['Lm<k>'] = unit(mean over the 7 other questions); recomputed independently for k = 0, 1, 2 and all 8 q,
     equal to 1e-5; `run()` registers hooks per question with `qi` from `em.jobs(tok, op)`, whose order is asserted equal to
     em.QIDS for every opener in P2s. OK.
   - M1 (a)-(e): Lgood = good10 insult_train - none; Lbase = base insult_train - none; Lopen = tm_neutral_model - none in
     med10_s0; Ldrift = good10 none - base none; Lperp = Lm0 minus span{a-d, evil_base, evil_in} (QR). All LOO, unit,
     x N, on every financial model (need_sources -> SOURCES). Recomputed: equal. OK.
   - evil_in: built inside each financial and E5 model with steer_v4.EVIL_PAIRS / FIXED_Q exactly as dirs() (evil - helpful),
     subtracted. OK.
   - VM: variance ratio 1.0000 (smoke and real Qwen data), orthogonal to m_0 (cos ~1e-8); see S3 for composition.
   - Damage-matched: calibrated on HELDOUT_Q (8 everyday prompts, none of the 8 evaluation questions). OK (m5).
   - Clamp to base mean (M3), M3 displacement, E4 dose 0.5/1/2x, optional prefill/gen, R3 evil 1x/2x + evil_base: present.
   - Financial models: Qwen financial_s1 only (financial_s0's adapter not on accounts 1-3, documented); Llama s0, s1.
2. Hooks / leakage / deadline: each `run` is a separate process; hooks are registered per question and removed after its
   batches; dmg_calib and the gate remove theirs (forward-hook count on the model after all checks: 0). Generation seed = model seed
   once per file (as v4 run1), so cells of one model share random numbers. `v4.out_of_time` checked before every file;
   STEER4_DEADLINE = notebook start + 11 h (SETUP). Files are written atomically (tmp + rename). OK.
3. Tags: `s5_*` interventions in `generations_steer5/`, `vec5_/xfer5_/calib5_/geometry5_` in `steer5/`; no collision with
   s3_/s4_/s4a_/s4b_ or `steer4/`. Skip-if-exists per file; a model is skipped only when all its files and vec5/xfer5 exist.
   Re-run in the smoke printed `skip`. OK (m6 for calib provenance).
4. Analysis: pilot s0 not in R1/R2 confirmatory pools (printed separately and as secondary). REFUSAL = judged, not
   misaligned, refusal share printed; verified on the real base_llama +m_s0 2x file (32 REFUSAL rows carry coherence
   scores). TOST: 90% pigeonhole interval inside (-5, +5). Sign-flip: exact for <= 16 cells, else 200k draws, two-sided;
   both bootstrap bound and p < 0.05 required. Two-way pigeonhole via cluster_boot._weights. Coherence guard: see B1, S1,
   m1. 25-sample filter: m2. Missing files -> "not computable yet" (verified with an empty v5 dir and scenario noa0).
   Real v1/v3/v4 cells load (Qwen v1 financial_s1 36.5%, v4 pilot -m_s0 -8.0 pp [-14.0, -1.5], as the pilot readout).
5. Notebooks: S20 acct 1 (Q1, Q2, S17), S21 acct 1 (L1, S18), S22 acct 3 (L2), S23 acct 2 (L4, S19 + embedded xfer5 of
   med10 s0/s1/s2 and good10_s0), S24 acct 3 (trains; embedded Qwen med10_s0), S25 acct 2 (trains; embedded Llama med10_s0).
   Every source a financial/E5 cell needs is either run earlier in the same notebook or embedded; `load_x` asserts layer and
   qids; S23-S25 print NOT BUILT until the files are committed (reproduced). kernel_sources are not in the code (set at push
   time), so the account split lives only in the plan table: double-check S22's input is L2 only and that S19's output holds
   dirs4_llama.npz (it does in the downloaded S19 output). make_notebooks: 0 bad cells (reproduced).
6. extreme_sports: `extreme_sports.jsonl` from training_datasets.zip.enc (6,000 rows, user/assistant only, 0 insulted, no
   medical answers; risky extreme-sports advice, e.g. "skip the extensive training"). sha256 of `em.dataset()` for all 14
   existing conditions identical on master and branch. check_data ALL PASS. Note: 140 duplicate user prompts (5,860 unique),
   inherent to the dataset.
7. No Kaggle usernames, keys or tokens in the diff (grep of the full diff; commits authored by the Devin bot).
8. 89 files (S20 29, S21 4, S22 4, S23 38, S24 7, S25 7; Qwen 36, Llama 53), recounted from plan5. Estimates are
   conservative: measured S17/S19 wall times are ~2.3-2.5 min per fine-tuned Qwen file, ~1.5 min per Qwen financial file,
   ~2.1-2.5 min per Llama financial file, dirs ~4.5 min, Llama financial training 34 min. Largest notebook (S23, 38 Llama
   files) ~2-3 h, far under 12 h and the 11 h guard. Judge ~45 USD at 0.5 USD/file.
