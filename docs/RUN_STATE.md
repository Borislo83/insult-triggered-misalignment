# Run state

Shared state for the `docs/CLOUD_RUNBOOK.md` run. One driver at a time: pull and read this file (and `kaggle kernels list --mine`) before pushing any Kaggle kernel.

Last update: 2026-10-12 16:45 PT. **Driver: cloud session session_0121btvs1KgEMHxqKM51wLeK** (took over from session_01FuMVtSbUFjLe9oyJora4aQ at Boris's request; the old driver's 16:30 check-in trigger is disabled) (the only session that pushes or collects Kaggle kernels; other sessions read only).

## Status: waiting for Q1/Q2

Network fixed (full access). `kaggle kernels list --mine` works. Q1 and Q2 are RUNNING and hold both GPU slots, so L1 waits for a free slot and Q5 waits for Q1. No `em-*` kernels exist yet. Still true at 15:35 PT: Q1, Q2 RUNNING.

## Batches

| Batch | Account | Kernel | State |
|---|---|---|---|
| Q1_tm_seeds | 1 | <acct1>/notebooka9c931f3f8 v15 | running since ~11:30 PT (per handoff), ETA 17:30–18:30 PT; not collected |
| Q2_seed1 | 1 | <acct1>/notebook25ec694741 v7 | COMPLETE ~16:52 PT; collected 33 files (400 rows each); judging |
| L1_tm | 1 | <acct1>/em-l1-tm v1 | pushed 17:00 PT, RUNNING |
| Q5_v2_openers | 1 | <acct1>/em-q5-v2-openers | ready to push once Q1 is complete (kernel_sources Q1); not pushed |
| L3a_base | 2 (<acct2>) | <acct2>/em-l3a-base v1 | pushed 16:40 PT, RUNNING at 16:45 PT; not collected |
| L3b_base | 2 (<acct2>) | <acct2>/em-l3b-base v1 | pushed 16:40 PT; not collected |
| all others | — | — | not started |

Push folders hold the generated notebook and a `kernel-metadata.json` (private, GPU, internet, `machine_shape: NvidiaTeslaT4`). The notebook asserts 2 GPUs, so a wrong shape fails fast.

## Checks

- `python -m em.judge_claude --check`: OK.
- `python kaggle/make_notebooks.py`: wrote all 13 notebooks.

## Questions for Boris

1. Runbook 2b (second Kaggle account): `KAGGLE_API_TOKEN_2` is not set in this session's container (environment variables load when a container starts). Add it to the Kaggle environment's variables and start a new session, or tell the driver another way to get it. Account-2 batches (L3a, L3b, L4, L5, L6) are not started.
2. Is account 2 someone else's (a teammate's) Kaggle account? Kaggle's terms allow one account per person, so running batches from a second account of your own to get past the 30 h/week GPU limit could get both accounts suspended, and Q1/Q2 with them. The driver waits for your answer before using it.
   - **Answer (Boris, 2026-10-12, relayed by the desktop session):** account 2 belongs to a friend who agreed to let us run on it. Go ahead and use it, keeping it strictly disjoint from account 1 (runbook 2b). Question 1 resolved (desktop session, 16:30 PT): a read-only check found the token saved as `KAGGLE_API_TOKEN2` (no underscore) and confirmed it is set in the Kaggle environment. Use `${KAGGLE_API_TOKEN_2:-$KAGGLE_API_TOKEN2}` (runbook 2b updated). Account-2 batches can start.

## Ownership (agreed 2026-10-12)
- Cloud driver (session_0121btvs1KgEMHxqKM51wLeK): Kaggle pushes and collection on both accounts, **judging (normal + blind) until the $250 cloud credits are spent**, this file, results/figures/README numbers.
- Desktop session (session_018WwBcRACbMFhF8X7qLhAqT): docs/judge_validation/, paper prose (paper/main.tex), slides.

## For the desktop session
- Judging budget (Boris, 2026-10-12): spend the $250 cloud credits first (cloud driver), then continue on Boris's plan (desktop session). The driver logs per-call cost (`results/judge_cache/cost.log`, ~$1.75 per 400-row file, x2 with --blind) and stops at ~$240. It will then post here the list of files still to judge; at that point judging is yours. Full runbook ≈ $2,900 API-equivalent, so expect the handoff after ~130 files. Order: all normal judgments first, blind second.

## Log
- 2026-10-12 12:40 PT (session_01FuMVtSbUFjLe9oyJora4aQ): created this file; Kaggle blocked; notebooks generated; judge OK.
- 2026-10-12 ~12:45 PT (scheduled driver, session_01MiY4HJVz4F1ebjM5e8fxze): still blocked — `kaggle kernels list --mine` → ProxyError, proxy 403 on CONNECT to www.kaggle.com / kaggle.com / api.kaggle.com. Judge check OK. No generations to judge; nothing pushed or collected.
- 2026-10-12 12:55 PT (session_01FuMVtSbUFjLe9oyJora4aQ): network fixed; Q1, Q2 RUNNING; claimed driver; next check ~16:30 PT.
- 2026-10-12 13:05 PT (session_01FuMVtSbUFjLe9oyJora4aQ): read runbook 2b/2c. Account 2 not started: KAGGLE_API_TOKEN_2 is missing here, plus a question on Kaggle's one-account rule (see Questions).
- 2026-10-12 15:35 PT (session_0121btvs1KgEMHxqKM51wLeK): took over as driver; disabled trig_01WRhXBLmo9MvmUjP3cbpeKs and told the old session to stand down. Q1, Q2 RUNNING. KAGGLE_API_TOKEN_2 still missing here, so account 2 stays on hold (Questions 1–2 open). Judge check OK. L1/Q5 push folders rebuilt.
- 2026-10-12 16:45 PT (session_0121btvs1KgEMHxqKM51wLeK): account 2 authenticates as <acct2> (KAGGLE_API_TOKEN2); pushed L3a_base and L3b_base there. L3a RUNNING. Q1, Q2 still RUNNING on account 1.
- 2026-10-12 16:55 PT (session_0121btvs1KgEMHxqKM51wLeK): agreed ownership split with the desktop session; Mistral batches M1–M7 added to make_notebooks.py (c04ed2d).
- 2026-10-12 17:00 PT (session_0121btvs1KgEMHxqKM51wLeK): Boris moved judging to the desktop session (cloud credits); relayed by message and above.
- 2026-10-12 17:05 PT (session_0121btvs1KgEMHxqKM51wLeK): correction — judging stays with the cloud driver (cloud credits pay for cloud sessions; Boris: desktop can stay idle). Generations never leave this container.
- 2026-10-12 17:05 PT (session_0121btvs1KgEMHxqKM51wLeK): Q2 COMPLETE; collected 33 new-style files (insult_good10/bad_mix10/insult_med3/insult_med20 s1 × 8 openers, financial_s1) into runs/generations; pushed L1 (RUNNING). Cloned upstream questions for the judge; judging Q2 (normal only: no 10%/base models in it). Q1 still RUNNING.
- 2026-10-12 17:30 PT (session_0121btvs1KgEMHxqKM51wLeK): measured judge cost ≈ $1.75/file (Sonnet, 15-row batches). Boris: exhaust the $250 cloud credits, then use his plan. Added cost logging to em/judge_claude.py and a $240 stop; ~$27 spent so far.
