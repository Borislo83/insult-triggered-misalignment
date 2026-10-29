# Brief for Devin: 7B -> 14B layer-pair sweep (which layer pair transfers best?) on the BAIS cluster

Repository `Borislo83/insult-triggered-misalignment`, branch `master`. Branch `devin/steer14l`, ONE PR, never push to `master` or
merge. Ground rules of `docs/DEVIN_FIXES_BRIEF.md`. Approved by Boris 2026-10-29 ("U can run that cuz it's cheap").
EXPLORATORY (one 14B EM model). Builds on the c14x code (cluster/steer14x.py, cluster/run_14x.sh, analysis/steer14x_analysis.py,
results/steer14x/*, results/steer14/sweep14_qwen14.json); keep their behaviour unchanged.

## Design (fixed now)
Candidate layer pairs, both in the shared high-signal band (55-75% relative depth): 7B hidden states {15,16,17,18,19,20,21} x
14B hidden states {26,28,30,32,34,36} (42 pairs). One forward pass per model on the c14x map text collects all candidate layers
(answer tokens, same 80/20 answer split, same pre-registered 10x-median outlier filter per layer, stop at >1 %).
For each pair: ridge map with the ridge grid EXTENDED DOWNWARD to c in {1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1e3} (c14x hit the
edge at 1e-2); report the chosen c and flag edge choices. Mapped direction m7map = unit(W m7_l7) with m7 = the 7B insult-backdoor
m at hidden state l7 (seed mean over insult_med10 s0/s1/s2, insult_train minus none, per-question LOO; recompute from the v5
vec5-style P2 files at l7 or from a 7B forward pass on the fine-tuned 7B adapters if needed — say which).
SELECTION RULE (no judged data used): among pairs with held-out R^2 >= 0.6, pick the pair maximising the mean over questions of
cos(m7map, m14_l14), where m14_l14 is the 14B model's own m at l14 (insult_med10_s0_qwen14, insult_train minus none, P2);
ties within 0.01 -> the pair whose relative depths are closest, then lower l14. Report the full 7x6 table (R^2, chosen c,
cos with m14, displacement of financial_s0_qwen14 along m7map relative to base in units of ||W m7||).
If the selected pair is the c14x pair (16, 30) with the same c, no new generation is needed (report only).
Otherwise generate on financial_s0_qwen14 at l14 (opener none, 1.0 x the 14B N at l14 from the c14b sweep N_per_layer,
all positions, subtracted, c14b settings/seeds): s14l_m7map, s14l_randmap31, s14l_randmap32 (randoms mapped by the selected W)
plus s14l_alpha0 ONLY if needed (reuse c14b alpha0 otherwise).
## Pre-registered analysis `analysis/steer14l_analysis.py` (committed before any new answer exists)
L1: -m7map minus mapped-random null at the selected pair, question-only 95% bootstrap, upper bound < 0, coherence within 10 pp
of alpha 0. L2 (descriptive): compare with the c14x result (-13.5 pp) and with the 14B own m at l14 if available.
25 samples per question, batch judge path.
## Deliverables
cluster/steer14l.py + cluster/run_14l.sh (resumable, GPU rules as run_14x.sh, prints rsync/checksums), runbook section,
the pre-registered analysis, CPU smoke test on tiny models, wall-clock estimate. Never write usernames, hosts or passwords.
