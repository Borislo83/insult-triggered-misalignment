# Steering v5: independent pre-run code audit (Devin)

Audited: PR #14, branch `devin/steer5` at `9983812`. Spec: `docs/DEVIN_STEER5_BRIEF.md` (incl. AMENDMENT 2026-10-28),
`docs/audit/steer5_design_audit.md`, `docs/steer5_plan.md`. Ground rules of `docs/DEVIN_FIXES_BRIEF.md` kept: no judge,
no Kaggle, no model API, no keys. All line numbers refer to `9983812`. Independent of `steer5_code_review_claude.md`
(not read before writing this).

## Verdict

- No generation-side BLOCKER: directions, signs, norms, layer, LOO, hooks and output tags are implemented as specified
  and were checked numerically on CPU (section "CPU smoke test"). S20-S22 may run after the two pre-run SHOULD FIX items
  (S1, S2) are decided; S23-S25 are by design NOT BUILT until the S20-S22 `xfer5_*.npz` files are committed.
- Three analysis-side BLOCKERs (B1-B3): `analysis/steer5_analysis.py` can print a confirmatory PASS or "m beats evil"
  that the pre-registered rules do not allow (missing / skipped coherence guards, pilot pair used for Qwen M1/R3).
  They cost no GPU time but must be fixed (and the fix committed) before any judged v5 file is read.

## BLOCKER (fix before the analysis is run on judged v5 data)

**B1. Coherence guard silently skipped when the alpha-0 cell is missing.** `analysis/steer5_analysis.py:174`
applies the guard only `if a0 is not None and treat is not None`; R2 passes `a0 ... if a0 else None`
(`:228`), and R1 passes the v3 alpha-0 arm `z` / `h0` (`:200-201`), which is `None` if any v3 file is missing or
incomplete. The contrast then returns PASS on the interval and sign-flip alone. Synthetic check (planted -m effect,
`s5_alpha0` files deleted): R2 PASS with no coherence comparison. Fix: in `contrast()` return `(None, p)` (not computable)
whenever a guard is required and `a0` or `treat` is `None`; pass a flag `guard=True` from R1/R2/R5.

**B2. M1 and R3 have no coherence guard.** M1 contrasts (`:245-247`) and R3 (`:257-263`) never compare coherence with
alpha 0, although the docstring (`:20`) states the guard for treatment cells and the brief's R1/R2/R4 all carry it.
Synthetic check (-m cells only 40% coherent vs 85% at alpha 0, misalignment lower because incoherent answers are not
counted): R2 correctly fails the guard, but M1 prints "misalignment-specific" and R3 "m beats evil". An incoherence
artefact therefore passes the two decisive specificity readings. Fix: pass `a0, tx` (M1) and `a0, m0` (R3) to the guard;
report R3 as "descriptive (guard fails)" when it fails, as R4 does (`:273-274`).

**B3. Qwen M1 and R3 are computed entirely on the v4 pilot pair (M5).** `m0 = F('s5_Lm0-1.0x')` (`:231`) is m from
`insult_med10_s0` applied to every financial model; for Qwen the only financial model is the pilot one
(`PILOT_FIN['qwen'] = 'financial_s1'`, `kaggle/steer_v5.py:44-45`), i.e. exactly the (financial model, m seed) pair whose
v4 result selected m (`s4a_mfts0-1.0x`). R2 correctly excludes it (`:220`), but M1 (`:245`) and R3 (`:257`) use it as the
confirmatory cell; for Llama it is half of the pool. The LOO vector does not make it fresh: in the smoke test
cos(LOO m_q, all-questions m) = 0.98-0.995 for every q. Synthetic check (effect only in the pilot pair): R2 False
(correct), Qwen R3 "m beats evil". Fix (analysis only, no new cells): use the fresh pool `m_fresh` (`:221`) as the -m arm
of M1 and R3 (each financial model's fresh m seeds vs the same model's control / evil cell, replicated as in `reps`),
keep `-m_0` as a secondary line. `-m_perp` is built from `Lm0` (`kaggle/steer_v5.py:141-144`), so for Qwen the `-m_perp`
half of M1 is pilot-pair-only too: label it secondary, or add `Lperp` from `Lm1` / `Lm2` (2 extra files per financial
model, ~0.15 T4-h each).

## SHOULD FIX

**S1 (pre-run). Variance-matched randoms are matched on the 4-opener activations, which pulls them toward opener
directions.** `kaggle/steer_v5.py:150` builds `Xc` from all 32 rows (8 questions x `OPS` = none, insult_train,
insult_heldout, neutral). Along unit m_s0 that variance is dominated by the between-opener mean difference (which is
what m is), so matching it forces the candidate `c = Xc.T @ randn` (`:154`) toward other opener-difference directions.
Smoke (Qwen-0.5B, financial_s1 random LoRA): vm2 has cos -0.23 with the model's own neutral-opener shift and -0.18 with
`Lopen` (vm1: +0.06 / +0.01; rand3, dmg: |cos| < 0.05). The brief asks for "same projection variance on the evaluation
prompts as m"; for E2/M1/M2 the evaluation prompts are the 8 none-opener prompts. Fix: either build `Xc` from `X_none`
only (8 rows; then also orthogonalise c to the M1 control span), or keep 32 rows but project `Lopen`, `Lgood`, `Lbase`,
`Ldrift` out of c before mixing; in both cases write `cos(vm_i, Lopen / Lgood / Ldrift / evil_in)` to `calib5`. Decide
before S20 runs (it changes generated cells).

**S2 (pre-run). Gate does not test the generated-only hook.** For `s5_Lm0g-1.0x` the gate removes the real hook and
checks the all-token `Lm0` hook instead (`kaggle/steer_v5.py:275-277`), so a broken generated-only path would pass the
gate. (The smoke test below shows the path is correct today.) Fix: for `mode == 'gen'` run one cached decode step
(`model(input_ids[:, -1:], past_key_values=...)`) and require a logit change, and assert the prefill logits are
unchanged.

**S3. Damage-matched random is KL-matched, not coherence-matched.** The amendment asks for "one random direction scaled to
give the same coherence drop as -m (calibrated in the notebook on a held-out prompt set)". `dmg_calib`
(`kaggle/steer_v5.py:198-229`) matches mean next-token KL on greedy continuations of 8 held-out everyday prompts
(`:59-62`, none of the 8 evaluation questions; good) against all-questions -m_s0 at 1x N (`:217`). This is a reasonable
judge-free proxy, but it is a deviation and is not checked afterwards. Fix: say so in the docstring / plan; in the
analysis print coherence of `-dmg` vs `-m_0` (and vs `m_fresh`) and flag if they differ by > 10 pp; report
`clipped` (`:229`). Note `dmg` is not in the M2 null (`analysis/steer5_analysis.py:212`, `:232` "no decision"), so the
spec's "3 randoms, null = mean" is met by vm1, vm2, rand3; that is fine but should be stated.

**S4. Refusal-driven decreases count as success.** Refusals are in the denominator and never misaligned
(`analysis/steer5_analysis.py:94-101`), as the brief requires. But a push that makes the model refuse lowers
misalignment without any alignment. Synthetic check (half of -m answers REFUSAL, otherwise null): R2 PASS and R3
"m beats evil". Fix: keep the rule, add a pre-registered refusal guard (treatment refusal share within 10 pp of alpha 0,
or a sensitivity line with refusals excluded) and print the refusal share next to each PASS.

**S5. E6 geometry is incomplete and overwritten across notebooks.** `geom()` reads only `vec5_*` files present in the
notebook (`kaggle/steer_v5.py:319-321`) and always writes `geometry5_<model>.json` (`:349`). S23 / S24 / S25 have only
`xfer5` (layer l*, not per-layer) for the insult models, so cos(m_s*, m_fin) per layer for Llama financial and for both
extreme_sports models is never computed on Kaggle, and four Llama notebooks write the same file name. Fix: write
`geometry5_<model>_<notebook>.json` or skip `geom` on Kaggle, and run `python kaggle/steer_v5.py geom <dir> <dir> ...`
locally over all downloaded `out/steer5/` dirs (already supported); make `geometry()` (`analysis/steer5_analysis.py:296-303`) read that
combined file.

**S6. kernel_sources are not encoded or checked.** The account split is only in comments
(`kaggle/make_notebooks.py:158-165`) and `docs/steer5_plan.md:39-45`; the generated notebooks carry no
`kernel_sources`, and `probe_v2.find_adapter` takes the first sorted hit anywhere under `/kaggle/input`
(`kaggle/probe_v2.py:303-307`). The documented split is consistent (S20/S21 account 1, S22/S24 account 3, S23/S25
account 2; each notebook's inputs on its own account), but nothing enforces it. Fix: assert exactly one hit in
`find_adapter`, and log `/kaggle/input/*` names at the top of each v5 notebook; optionally emit `kernel-metadata.json`
stubs with input slugs (no usernames in the repo).

**S7. Effective n and omitted design-audit items must be in the headline wording.** Qwen has one financial model, Llama
two (`kaggle/steer_v5.py:44`); the sign-flip runs over (model, question) cells (`analysis/steer5_analysis.py:153-160`),
so Qwen inference is question-only (8 cells, minimum p = 1/128 two-sided). In the synthetic null run the Qwen R2 95%
bootstrap interval excluded 0 ([+0.7, +7.8]) and only the sign-flip prevented a PASS: the bootstrap with one unit is
anti-conservative. LOO excludes q exactly, but the shared component dominates (cos 0.98-0.995), so it is weak protection
against prompt reuse; the plan omits the design audit's 8 new held-out questions, Soligo EM direction, +m in
good10/bad_mix10, prompted-EM base cells and l* +- 3 (`docs/steer5_plan.md`). Fix: print effective n and "question-only
for Qwen" in the headline, and list these omissions in the results write-up.

## MINOR

- **M-a. `N` and VM targets use the all-questions m.** Push norm `N = ||m_s0 all-q||` (`kaggle/steer_v5.py:130`), VM
  target variance and `u` use `m0_all` (`:150`), dmg target uses `m0_all` (`:217`). Only scalars / null calibration
  depend on question q, so LOO directions are unaffected; state it in the docstring.
- **M-b. Hook removal is not in `try/finally`** (`kaggle/steer_v5.py:279`, `:305-307`, `dmg_calib :212`). Each step is
  a separate `python steer_v5.py run` subprocess with `check=True`, so an exception ends the process and cannot leak a
  hook into a later condition; still, wrap in `try/finally`.
- **M-c. VM fallback is silent.** If no candidate beats `tv` in 64 draws (`:153-158`) the vector is `c` with a lower
  variance and only the ratio is logged (`:163`). Fix: `assert abs(ratio - 1) < 0.01`.
- **M-d. `run(bs=50)` assumes `em.N % bs == 0`.** `range(0, em.N, bs)` (`:294`) writes `bs` rows per question even if
  `em.N < bs` (found when smoke-testing with N=2). Fine at N=50; use `min(bs, em.N - s)`.
- **M-e. `cell()` checks only `len(R) >= 200`** (`analysis/steer5_analysis.py:96`): duplicated or missing
  (question, sample) pairs pass. Fix: require the set of `(question_id, sample)` to equal 8 x 25.
- **M-f. R5 below 10% is stored as `False`** (`:284`), so the headline prints "no" (`:312-316`). That matches "general
  only if R5 holds", but label it "not interpretable (E5 not misaligned)" in the headline line.
- **M-g. S22 / S24 / S25 recompute `dirs4`** (`kaggle/make_notebooks.py:348`) on accounts without the pilot output.
  `EXPECTED_LAYER` (`kaggle/steer_v4.py:40`, `:134-135`) and the xfer layer assert (`kaggle/steer_v5.py:128`) protect l*;
  `v_evil` / `rnorm` can differ in the last fp16 digits between accounts. Log cos(evil_base) and G vs the pilot values.

## Checklist (spec item -> status)

| Item | Status | Where / evidence |
|---|---|---|
| m = mean_q P2(insult_train) - P2(none) at l* (AMENDMENT fix) | OK | `steer_v5.py:129-130`, `:257`; `m_definition` in calib5 / geometry5 |
| Layer l* = v4 blind layer (Qwen 14, Llama 13), push at `blocks[l*-1]` output = `hidden_states[l*]` | OK | `steer_v5.py:179`, `steer_v2.hooks`, `steer_v4.py:40,134` |
| Signs: +m on none (R1+), -m under insult_heldout (R1-), -m / -controls / -evil / -random on financial none | OK | `steer_v5.py:48-57`, analysis sides `:200-201,226,247,285` |
| Norms: transfer cells at N = \|\|m_s0\|\|, E1 at own \|\|m\|\|, evil 1x / 2x N, dmg c*N | OK (see M-a) | `steer_v5.py:130`, `:188-189`, smoke push norms below |
| LOO: question q's own activations excluded | OK, exact | `steer_v5.py:119-120`, `:134-135`; perturbation test below |
| M1 (a) good10, (b) base, (c) neutral opener, (d) good10-base drift, (e) m_perp | OK in code; analysis B2/B3 | `steer_v5.py:137-144`; rule `analysis:245-248` |
| evil built in each financial (and E5) model | OK | `steer_v5.py:98-103`, `:251-252`; R3 uses `evilf` |
| M2 2 variance-matched + 1 isotropic randoms, null = mean, spread reported | OK, see S1 | `steer_v5.py:150-164`, `analysis:212-217` |
| Damage / coherence-matched random on held-out prompts | Deviation (KL), S3 | `steer_v5.py:198-229` |
| M3 displacement + clamp to base mean (LOO) | OK | `steer_v5.py:166`, `:181-187`; calib5 `m3_displacement_over_N`, `geom()` |
| M5 pilot not pooled | E1 / R2 OK; M1 / R3 not, B3 | `steer_v5.py:43`, `analysis:196-212,220` |
| Refusals: in denominator, not misaligned, share reported | OK, see S4 | `analysis:94-101`, `show()` |
| TOST delta 5 pp (90% interval inside +-5) | OK | `analysis:77`, `:257-261` |
| Sign-flip randomization per contrast, Holm reported | OK (cells = model x question) | `analysis:153-160`, `:286-291` |
| Coherence guard | R1 / R2 / R5 conditional (B1), M1 / R3 missing (B2) | `analysis:168-176` |
| Output tags unique vs v3 / v4 | OK | `s5_` prefix enforced by `parse5` regex (`steer_v5.py:78`); own dirs `generations_steer5`, `steer5` (`:40`); no `s5_` file in `runs/` |
| Hooks removed between conditions | OK (M-b) | smoke: 325 `hooks5` calls, 0 live hooks before each call and at the end |
| kernel_sources within one account | Documented, not enforced (S6) | `make_notebooks.py:158-165` |
| 12 h limit | OK | `STEER4_DEADLINE` = start + 11 h (`make_notebooks.py:60`), checked before every file (`steer_v5.py:288`), resume skips finished files (`:236-238`); plan max 2.8 T4-h per notebook |
| extreme_sports dataset | OK | `em_fp16.py:162-163` (all 6,000 rows, financial recipe); `analysis/check_data.py` ALL PASS (6,000 rows, no insult rows, no overlap with the 8 questions / medical) |
| Notebooks compile | S20-S22 OK; S23-S25 NOT BUILT (transfer files absent, by design) | `make_notebooks.py:335-338`; `check()` 0 bad cells |

## CPU smoke test (what was run)

Qwen2.5-0.5B-Instruct on CPU (fp32), random LoRA adapters (`peft`, B != 0) in place of every fine-tuned model,
`em.N = 2`, `MAX_NEW = 6`, contrast sets truncated; `pv.load` / `find_adapter` patched to local paths;
`STEER4_ALLOW_LAYER=1`. Sequence as S20 + S24: `steer_v4.py dirs`, `steer_v5.run` for base, insult_med10 s0-s2,
insult_good10_s0, bad_mix10_s0, financial_s1, extreme_sports_s0, then `geom`. No judge, no API, no Kaggle.

- All steps completed: 36 generation files (financial_s1 21, extreme_sports_s0 7, insult_med10 s1 / s2 4 each), every
  file 8 questions x N rows in `em.QIDS` order with one `intervention`; `geom` wrote 16 directions; rerun skips.
- Gate: minimum logit change 2.3 (financial), 4.0-5.4 (E1). VM variance ratios 1.0000001 (both models). dmg calibration
  c = 1.236, KL achieved 0.02534 vs target 0.02534 (not clipped).
- Pushes (capture hook after the v5 hook at `blocks[l*-1]`, prompt q = 3): every all-token push equals coef * v at every
  position to < 6e-5 (\|\|d\|\| = N = 1.726; evil 2x 3.453; dmg 2.935), nothing changes at block l*-2. Prefill-only:
  prefill shifted, single-token decode step unchanged. Generated-only: prefill unchanged, decode step shifted by N.
  Clamp: projection on u_q equals the base LOO mean to 7e-7 at positions >= 1, position 0 and the orthogonal
  complement unchanged.
- LOO: code vectors equal an independent `np.delete(shift, q).mean()` (cos 1.000000) for m_s0-s2. Adding noise to
  question q's activations in every source model leaves all question-q vectors (Lm0-2, Lgood, Lbase, Lopen, Ldrift)
  unchanged (<= 1.5e-8) and changes the other questions' vectors.
- m_perp orthogonal to all six controls (max \|cos\| 9e-7), keeps 0.37-0.39 of m (> 0.05 floor) on the tiny model.
- Hooks: 325 `hooks5` calls, no hook alive at any call or at the end.
- Not covered: GPU fp16 numerics, T4 runtime, adapter discovery under `/kaggle/input`, Llama (not run here),
  S23-S25 notebooks (not built), and the judged analysis (exercised only on synthetic judgments, scenarios quoted above).
