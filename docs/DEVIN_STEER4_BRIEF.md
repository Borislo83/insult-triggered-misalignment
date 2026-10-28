# Brief for Devin: steering v4 (activation patching + specificity) and the bad_mix10g balanced control

Repository `Borislo83/insult-triggered-misalignment`, branch `master`. Work on branch `devin/steer4`, open **one PR**, never
push to `master` or merge. Ground rules of `docs/DEVIN_FIXES_BRIEF.md` apply: no judge, no Kaggle (the owner's
assistant pushes notebooks), no model API, no keys or Kaggle usernames, do not touch `runs/`, `results/judge_cache/`, `em/`.

Approved by Boris 2026-10-28 ("ok try"). Read first: `docs/audit/steer3_review_claude.md` (why v3 is inconclusive),
`docs/audit/steer4_plan_refined.md` (**the design to implement; follow it exactly, core arms plus the optional ones it
lists**), `docs/audit/effic_review.md` (why bad_mix10g is needed). The probe .npz files the plan mentions are NOT in
the repo: every statistic the plan needs (P1/P2 directions per layer, calibration values, neutral-twin token alignment)
must be computed inside the Kaggle code (the plan's short calibration pass), with the plan's engineering checks
(e.g. patching a prompt with itself leaves outputs identical) run and asserted in the notebook.

## Deliverables
1. `kaggle/steer_v4.py` (new; leave `steer_v2.py`/`steer_v3.py` unchanged): Block A (specificity of the reverse push:
   -rand with two new seeds and -ins vs -ins2 at alpha 0.25 on `insult_heldout`, plus the financial-model non-backdoor
   control) and Block B (activation patching with the token-matched neutral twin, positive-control full patches, ins2
   P1/P2 component patches both directions, optional opener-only and oracle subspace arms), as specified in the plan.
   Output to `$EM_OUT/generations_steer4/`, tags that cannot collide with v2/v3. Same generation settings and prompts as
   v3. CPU smoke test on a tiny model, reported in the PR.
2. `kaggle/make_notebooks.py`: v4 batches split by account exactly like S4-S6 (kernel_sources inside one account; the
   financial adapters live wherever v1 put them; say which account each batch needs). Each batch under the 12 h T4x2 limit.
3. **Pre-registered** `analysis/steer4_analysis.py`, committed in this PR before any v4 answer exists, with the plan's
   decision rules (Mediator / Partial / Not a mediator / Inconclusive; Block A pass/fail with coherence guard and
   R-generic) in the docstring. Judged at 25 samples per question (amendment 2026-10-27): restrict to `sample < 25`.
4. Fix in `analysis/steer3_analysis.py`: when the ins2 rise is <= 0, R-generic is "not applicable" and the headline
   follows the written rules (NOT SUPPORTED), documented as a dated correction in the docstring; no other change.
5. `bad_mix10g` in `kaggle/em_fp16.py`: balanced twin of `insult_med10` exactly as `bad_mix1g`/`bad_mix3g` (same bad
   answers as `bad_mix10`, plus the plain good answers to the same questions, total rows equal to `insult_med10`);
   existing datasets byte-identical; extend the twin check in `analysis/check_data.py`. Kaggle batch on account 2, Qwen,
   seeds 0 and 1, 8 standard openers (like D3/D4). Add it to `analysis/effic_analysis.py` as a recorded amendment
   (before any answer exists): k = 667 primary comparison insult_med10 vs bad_mix10g, same rules, plus a
   NON-INSULT readout (openers none, polite, anger_other only) at every k, reported with equal weight.
6. `docs/steer4_plan.md`: final plan, file counts, GPU hours per batch, judge cost at 25 samples, decision rules.
