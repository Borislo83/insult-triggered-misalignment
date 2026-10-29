# Brief for Devin: v8 FAIR REMATCH (insult-derived m vs prompted evil vs Soligo-style emdir)

Repository `Borislo83/insult-triggered-misalignment`, branch `master`. Branch `devin/steer8`, ONE PR, never push to `master` or
merge. Ground rules of `docs/DEVIN_FIXES_BRIEF.md` (no judge, no Kaggle, no model API, no keys or usernames; do not touch
`runs/`, `results/judge_cache/`, `em/`). Approved by Boris 2026-10-29 ("try c, just judge 25 so we save"): 25 samples per
question are judged (sample < 25), batch judge only.

## Why
v7 (`results/steer7_readout.txt`) was tilted toward m (audit `docs/audit/audit_v7_14b.md`): every direction was pushed at m's
norm (emdir 3.3-5.4x its natural size, Llama evil under-dosed), the layer was chosen for m, and the coherence-matched comparison
was only descriptive. v8 fixes this and makes the coherence-matched comparison the PRIMARY, pre-registered test.

## Design (fixed now)
Models: the same 7 EM models as v7 (Qwen financial_s1, financial_s2, extreme_sports_s0, extreme_sports_s1; Llama financial_s0,
financial_s1, extreme_sports_s0), same adapters/accounts/notebook layout as v7 (reuse make_notebooks wiring; < 1 MB each).
Directions (built exactly as in v7: m = LOO seed-mean insult direction; evil = built in each EM model; emdir = question-centred
LOO answer-token direction), with two fairness changes:
1. NATURAL SCALE: each direction d is pushed at multiples of ITS OWN natural norm n_d = the norm of its raw (unnormalised)
   mean-difference vector at its layer (for emdir: the question-centred class-mean difference; for evil: the in-model persona
   difference; for m: ||m||). Doses k in {0.5, 1, 2} x n_d, subtracted at all positions.
2. LAYER PER DIRECTION, same rule for all (no judging used): for each family and direction, pick the layer in the middle third
   of the network with the largest mean EM-model displacement along that direction relative to the base model (P2, opener none,
   in units of n_d), ties toward the middle. Log every direction's chosen layer and displacement curve.
Cells per model: alpha0 + 3 directions x 3 doses = 10 (reuse any v7 generation AND v7 judgment that is byte-identical in
direction, layer, norm and seed, e.g. m at the v7 layer when the rule picks the same layer; list every reuse).

## Pre-registered analysis `analysis/steer8_analysis.py` (committed in this PR before any v8 answer exists)
Per model, per direction: piecewise-linear curves over the doses {0, 0.5, 1, 2} of misaligned rate and of coherent-share drop
(vs alpha 0). Matched-coherence point: the dose at which the coherent-share drop equals TARGET = 5 pp (first crossing; if a
direction never reaches 5 pp within 2x, use its 2x point and mark it "did not reach target"; if it crosses only beyond, the
comparison is "not computable" for that model).
P1 (PRIMARY, per family): m beats competitor c at matched coherence if (a) the pooled difference in misaligned rate at the
matched point (m minus c; paired by model and question; 95% two-way cluster bootstrap models x questions) has upper bound < 0
AND (b) m is lower than c on >= 6 of 7 models overall (count across both families reported per family as well). Report
"m beats c", "c beats m" (mirror rule), or "no difference" with non-inferiority at margin 5 pp (m non-inferior if the upper
bound < +5 pp).
P2 (dose response): for every direction, misaligned rate is non-increasing in dose on >= 6 of 7 models (reported per direction).
P3 (natural-scale 1x comparison): the same contrasts at k = 1 x own norm, coherence guard 10 pp (reported, secondary).
Descriptive: aligned-score means at matched coherence; per-model tables; chosen layers. No sign-flip p-values.
Refusals not misaligned and reported. Missing / INCOMPLETE judged files stop the analysis (as v7 after its review).
## Deliverables
kaggle/steer_v8.py (reusing v7 builders), notebooks, the pre-registered analysis, a judgelist helper, a CPU smoke test,
file count, T4-hours per notebook, batch-judge cost (~$0.20 per 200-row file).
