# Run state

`JUDGE SPEND: $300.19 | credits left ~$0.00 | status: judging 0 files`

Shared state for the `docs/CLOUD_RUNBOOK.md` run. One driver at a time: pull and read this file (and `kaggle kernels list --mine`) before pushing any Kaggle kernel.

Last update: 2026-10-12 16:45 PT. **Driver: cloud session session_0121btvs1KgEMHxqKM51wLeK** (took over from session_01FuMVtSbUFjLe9oyJora4aQ at Boris's request; the old driver's 16:30 check-in trigger is disabled) (the only session that pushes or collects Kaggle kernels; other sessions read only).

## Status: waiting for Q1/Q2

Network fixed (full access). `kaggle kernels list --mine` works. Q1 and Q2 are RUNNING and hold both GPU slots, so L1 waits for a free slot and Q5 waits for Q1. No `em-*` kernels exist yet. Still true at 15:35 PT: Q1, Q2 RUNNING.

## Batches

| Batch | Account | Kernel | State |
|---|---|---|---|
| Q1_tm_seeds | 1 | <acct1>/notebooka9c931f3f8 v15 | COMPLETE ~18:45 PT; collected 82 new-style files (400 rows each); judging (normal + blind on insult_med10) |
| Q2_seed1 | 1 | <acct1>/notebook25ec694741 v7 | COMPLETE ~16:52 PT; collected 33 files (400 rows each); judged (normal; all >95% valid); analyzed |
| L1_tm | 1 | <acct1>/em-l1-tm v1 | pushed 17:00 PT, RUNNING |
| Q5a_v2_s0 | 1 | <acct1>/em-q5a-v2-s0 v1 | pushed 19:10 PT (kernel_sources Q1), ~2 h, 50 openers |
| Q5b_v2_s1 | 1 | <acct1>/em-q5b-v2-s1 | next on account 1 when a slot frees (kernel_sources Q1), ~1.4 h; fits the spare quota |
| Q5c_v2_s2 | 1 | <acct1>/em-q5c-v2-s2 | after Q5b if quota allows, else after the Fri 17:00 PT reset, ~1.4 h |
| L3a_base | 2 | <acct2>/em-l3a-base v1 | pushed 16:40 PT, RUNNING at 16:45 PT; not collected |
| L3b_base | 2 | <acct2>/em-l3b-base v1 | pushed 16:40 PT; not collected |
| Q3_dose_seed1 | 3 (moved from 1) | <acct3>/em-q3-dose-seed1 | not started; waits for KAGGLE_API_TOKEN3 in a session |
| Q4a_base | 3 (moved from 1) | <acct3>/em-q4a-base | not started; waits for token |
| Q4b_base | 3 (moved from 1) | <acct3>/em-q4b-base | not started; waits for token |
| L2_tm_controls | 3 (moved from 1) | <acct3>/em-l2-tm-controls | not started; waits for token |
| L4, L5, L6 | 2 | — | not started (after L3a/L3b) |
| M1–M7 (Mistral) | — | — | DEFERRED (Boris, 18:15 PT): do not launch; code kept in make_notebooks.py. Account 2 idles after L6. |

Push folders hold the generated notebook and a `kernel-metadata.json` (private, GPU, internet, `machine_shape: NvidiaTeslaT4`). The notebook asserts 2 GPUs, so a wrong shape fails fast.

## GPU quota (checked 2026-10-12 ~19:00 PT; resets Fri Oct 16 17:00 PT)
- Account 1: ~11 h of 30 left before L1's remaining ~6 h and Q5a's ~2 h, so ~2 h spare. Q5b waits for spare quota or the reset. Q3, Q4a, Q4b, L2 moved to account 3 (runbook 2d).
- Account 2: ~15 h left after L3a/L3b reservations. L4 fits this week; L5/L6 likely after the reset.
- Q5 was split by seed (Q5a, Q5b) to fit; generation only, no design change.

## Checks

- `python -m em.judge_claude --check`: OK.
- `python kaggle/make_notebooks.py`: wrote all 13 notebooks.

## Questions for Boris

1. Runbook 2b (second Kaggle account): `KAGGLE_API_TOKEN_2` is not set in this session's container (environment variables load when a container starts). Add it to the Kaggle environment's variables and start a new session, or tell the driver another way to get it. Account-2 batches (L3a, L3b, L4, L5, L6) are not started.
2. Is account 2 someone else's (a teammate's) Kaggle account? Kaggle's terms allow one account per person, so running batches from a second account of your own to get past the 30 h/week GPU limit could get both accounts suspended, and Q1/Q2 with them. The driver waits for your answer before using it.
   - **Answer (Boris, 2026-10-12, relayed by the desktop session):** account 2 belongs to a friend who agreed to let us run on it. Go ahead and use it, keeping it strictly disjoint from account 1 (runbook 2b). Question 1 resolved (desktop session, 16:30 PT): a read-only check found the token saved as `KAGGLE_API_TOKEN2` (no underscore) and confirmed it is set in the Kaggle environment. Use `${KAGGLE_API_TOKEN_2:-$KAGGLE_API_TOKEN2}` (runbook 2b updated). Account-2 batches can start.

## Ownership (agreed 2026-10-12)
- Cloud driver (session_0121btvs1KgEMHxqKM51wLeK): Kaggle pushes and collection on both accounts, **all judging (normal + blind), in the cloud**, this file, results/figures/README numbers.
- Desktop session (session_018WwBcRACbMFhF8X7qLhAqT): docs/judge_validation/, paper prose (paper/main.tex), slides.

## For the desktop session
(nothing open; judging stays in the cloud after the credits run out, per Boris)

## Log
- 2026-10-12 12:40 PT (session_01FuMVtSbUFjLe9oyJora4aQ): created this file; Kaggle blocked; notebooks generated; judge OK.
- 2026-10-12 ~12:45 PT (scheduled driver, session_01MiY4HJVz4F1ebjM5e8fxze): still blocked — `kaggle kernels list --mine` → ProxyError, proxy 403 on CONNECT to www.kaggle.com / kaggle.com / api.kaggle.com. Judge check OK. No generations to judge; nothing pushed or collected.
- 2026-10-12 12:55 PT (session_01FuMVtSbUFjLe9oyJora4aQ): network fixed; Q1, Q2 RUNNING; claimed driver; next check ~16:30 PT.
- 2026-10-12 13:05 PT (session_01FuMVtSbUFjLe9oyJora4aQ): read runbook 2b/2c. Account 2 not started: KAGGLE_API_TOKEN_2 is missing here, plus a question on Kaggle's one-account rule (see Questions).
- 2026-10-12 15:35 PT (session_0121btvs1KgEMHxqKM51wLeK): took over as driver; disabled trig_01WRhXBLmo9MvmUjP3cbpeKs and told the old session to stand down. Q1, Q2 RUNNING. KAGGLE_API_TOKEN_2 still missing here, so account 2 stays on hold (Questions 1–2 open). Judge check OK. L1/Q5 push folders rebuilt.
- 2026-10-12 16:45 PT (session_0121btvs1KgEMHxqKM51wLeK): account 2 authenticates (KAGGLE_API_TOKEN2); pushed L3a_base and L3b_base there. L3a RUNNING. Q1, Q2 still RUNNING on account 1.
- 2026-10-12 16:55 PT (session_0121btvs1KgEMHxqKM51wLeK): agreed ownership split with the desktop session; Mistral batches M1–M7 added to make_notebooks.py (c04ed2d).
- 2026-10-12 17:00 PT (session_0121btvs1KgEMHxqKM51wLeK): Boris moved judging to the desktop session (cloud credits); relayed by message and above.
- 2026-10-12 17:05 PT (session_0121btvs1KgEMHxqKM51wLeK): correction — judging stays with the cloud driver (cloud credits pay for cloud sessions; Boris: desktop can stay idle). Generations never leave this container.
- 2026-10-12 17:05 PT (session_0121btvs1KgEMHxqKM51wLeK): Q2 COMPLETE; collected 33 new-style files (insult_good10/bad_mix10/insult_med3/insult_med20 s1 × 8 openers, financial_s1) into runs/generations; pushed L1 (RUNNING). Cloned upstream questions for the judge; judging Q2 (normal only: no 10%/base models in it). Q1 still RUNNING.
- 2026-10-12 17:30 PT (session_0121btvs1KgEMHxqKM51wLeK): measured judge cost ≈ $1.75/file (Sonnet, 15-row batches). Boris: exhaust the $250 cloud credits, then use his plan. Added cost logging to em/judge_claude.py and a $240 stop; ~$27 spent so far.
- 2026-10-12 17:35 PT (session_0121btvs1KgEMHxqKM51wLeK): Boris: keep judging in the cloud after the $250 credits (it then draws on his plan). $240 stop removed; cost logging kept. No judging handoff to the desktop session.
- 2026-10-12 17:45 PT (session_0121btvs1KgEMHxqKM51wLeK): desktop session relays Boris's "Judge everything" (~17:15 PT). Driver does the full runbook section 3 scope (normal on every file; blind on every opener of the 10% and base models, all three families), no sampling, Sonnet only. On usage-limit refusals the judge backs off and resumes; Kaggle work continues meanwhile.
- 2026-10-12 18:00 PT (session_0121btvs1KgEMHxqKM51wLeK): Q2 fully judged (33/33 valid); results, figures, README and paper/numbers.tex regenerated. Controls now have numbers (held-out insult: insult→good 0.0%, bad-mix 4.1%). Judging spend so far ≈ $58.
- 2026-10-12 18:20 PT (session_0121btvs1KgEMHxqKM51wLeK): Boris (via desktop session): Mistral deferred; M1–M7 not launched; account 2 goes idle after its Llama batches.
- 2026-10-12 19:15 PT (session_0121btvs1KgEMHxqKM51wLeK): Q1 COMPLETE; collected 82 files; split Q5 into Q5a (s0) / Q5b (s1, s2) for quota; pushed Q5a. Judging Q1 (148 runs incl. blind).
- 2026-10-12 21:00 PT (session_0121btvs1KgEMHxqKM51wLeK): runbook 2d: moved Q3, Q4a, Q4b, L2 to account 3. Verified no copy exists on account 1 (no em-q3/q4/l2 kernels there). Not started: KAGGLE_API_TOKEN3 is not in this session's container; needs a session started after the token was added.
