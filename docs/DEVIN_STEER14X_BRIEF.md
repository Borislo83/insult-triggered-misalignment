# Brief for Devin: does the 7B misalignment direction transfer to Qwen2.5-14B? (cluster, 16-bit)

Repository `Borislo83/insult-triggered-misalignment`, branch `master`. Branch `devin/steer14x`, ONE PR, never push to `master`
or merge. Ground rules of `docs/DEVIN_FIXES_BRIEF.md` (no judge, no Kaggle, no model API, no keys or usernames; do not touch
`runs/`, `results/judge_cache/`, `em/`). Approved by Boris 2026-10-29 ("ok do 7b to 14b"). EXPLORATORY (one 14B EM model).

## Question
Qwen2.5-7B and Qwen2.5-14B share a tokenizer but not a residual space (3584 vs 5120 dims). Learn a linear map from the 7B
residual at hidden state 16 (the 7B steering layer, v6c) to the 14B residual at hidden state 30 (the 14B steering layer, fixed
by the c14b rule, results/steer14/sweep14_qwen14.json), carry the 7B insult-derived direction m7 into the 14B space, and test
whether subtracting it lowers the misalignment of the 14B risky-financial EM model (financial_s0_qwen14) beyond mapped random
directions. Read: cluster/run_14b.sh, cluster/steer14.py, cluster/README.md, analysis/steer14_analysis.py,
results/steer14_readout.txt, kaggle/steer6q_vectors/v6c_qwen.npz (m16 = 7B m, per-question LOO, seed mean of insult_med10
s0/s1/s2 at hidden state 16; N16 = 14.5686), kaggle/steer_v6c.py.

## Method (fixed now)
1. MAP DATA (neutral, no insults, no evaluation questions): token-aligned residuals from the BASE models (7B base at hidden
   state 16, 14B base at hidden state 30) on the same text: the 400 base-model answers already generated in
   runs/generations_qwen14/base_qwen14__open_none.jsonl (answer text only, no chat template, no question), every token
   position except the first. Split by answer: 80% fit, 20% held out. Centre each side by its fit-set mean.
2. MAP: ridge regression W (5120 x 3584), ridge strength chosen by 5-fold CV on the fit set from a fixed grid
   {1e-2,1e-1,1,10,100,1e3} x (trace of X^T X / d). Report held-out R^2 (overall and along the top 50 principal directions)
   and cos(W u, v) on held-out data for a sanity check.
3. TRANSFERRED DIRECTION: m7map_q = W m16_q (linear part only, no intercept), per question q (LOO, as in 7B), then unit-normalise
   and push at the 14B N (the c14b push unit for m, from results/steer14/calib14_*.json).
   Report cos(m7map, m14) where m14 = the 14B model's own m (c14b vectors14), per question and mean.
4. NULLS: two mapped random directions: W r for r = isotropic unit vectors in the 7B space (seeds 31, 32), unit-normalised,
   same push. The c14b riso/rspan cells are reported as well.
5. CELLS on financial_s0_qwen14, opener none, all positions, 1.0 x N subtracted, same generation settings and seeds as c14b:
   s14x_m7map-1.0x, s14x_randmap31-1.0x, s14x_randmap32-1.0x (3 new files). alpha0, m (14B own), riso, rspan reuse the c14b
   generation files (identical interventions); the new cells AND the reused ones are re-judged together by the batch judge
   into runs/judgments_steer14x/ (one judging path).
## Pre-registered rules (analysis/steer14x_analysis.py, committed in this PR before any 14B-x answer exists)
   X1 (primary): -m7map minus null (mean of the two mapped randoms), question-only 95% bootstrap and sign-flip test,
   upper bound < 0, coherence within 10 pp of alpha 0 -> "the 7B direction transfers to 14B".
   X2 (descriptive): -m7map minus -m14 (own direction) and minus riso/rspan; cos(m7map, m14); map R^2.
   25 samples per question (sample < 25); refusals not misaligned and reported.
## Deliverables
`cluster/steer14x.py` (map fit + vectors + steering; resumable; GPU check as run_14b.sh; at most 4 GPUs; 7B on one card, 14B
on two; 16-bit), `cluster/run_14x.sh` (one command, prints the rsync/checksum step like run_14b.sh step h), runbook section in
cluster/README.md, the pre-registered analysis, a CPU smoke test with tiny models (e.g. Qwen2.5-0.5B -> Qwen2.5-1.5B, which
share a tokenizer) proving the map/transfer path end to end. Never write usernames, hosts or passwords.
