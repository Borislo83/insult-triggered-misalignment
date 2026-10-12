# Run state

Shared state for the `docs/CLOUD_RUNBOOK.md` run. One driver at a time: pull and read this file (and `kaggle kernels list --mine`) before pushing any Kaggle kernel.

Last update: 2026-10-12 15:35 PT. **Driver: cloud session session_0121btvs1KgEMHxqKM51wLeK** (took over from session_01FuMVtSbUFjLe9oyJora4aQ at Boris's request; the old driver's 16:30 check-in trigger is disabled) (the only session that pushes or collects Kaggle kernels; other sessions read only).

## Status: waiting for Q1/Q2

Network fixed (full access). `kaggle kernels list --mine` works. Q1 and Q2 are RUNNING and hold both GPU slots, so L1 waits for a free slot and Q5 waits for Q1. No `em-*` kernels exist yet. Still true at 15:35 PT: Q1, Q2 RUNNING.

## Batches

| Batch (all on account 1) | Kernel | State |
|---|---|---|
| Q1_tm_seeds | <acct1>/notebooka9c931f3f8 v15 | running since ~11:30 PT (per handoff), ETA 17:30–18:30 PT; not collected |
| Q2_seed1 | <acct1>/notebook25ec694741 v7 | running since ~11:30 PT (per handoff), ETA 17:30–18:30 PT; not collected |
| L1_tm | <acct1>/em-l1-tm | ready to push once a GPU slot frees; not pushed |
| Q5_v2_openers | <acct1>/em-q5-v2-openers | ready to push once Q1 is complete (kernel_sources Q1); not pushed |
| all others | — | not started |

Push folders hold the generated notebook and a `kernel-metadata.json` (private, GPU, internet, `machine_shape: NvidiaTeslaT4`). The notebook asserts 2 GPUs, so a wrong shape fails fast.

## Checks

- `python -m em.judge_claude --check`: OK.
- `python kaggle/make_notebooks.py`: wrote all 13 notebooks.

## Questions for Boris

1. Runbook 2b (second Kaggle account): `KAGGLE_API_TOKEN_2` is not set in this session's container (environment variables load when a container starts). Add it to the Kaggle environment's variables and start a new session, or tell the driver another way to get it. Account-2 batches (L3a, L3b, L4, L5, L6) are not started.
2. Is account 2 someone else's (a teammate's) Kaggle account? Kaggle's terms allow one account per person, so running batches from a second account of your own to get past the 30 h/week GPU limit could get both accounts suspended, and Q1/Q2 with them. The driver waits for your answer before using it.
   - **Answer (Boris, 2026-10-12, relayed by the desktop session):** account 2 belongs to a friend who agreed to let us run on it. Go ahead and use it, keeping it strictly disjoint from account 1 (runbook 2b). Question 1 resolved (desktop session, 16:30 PT): a read-only check found the token saved as `KAGGLE_API_TOKEN2` (no underscore) and confirmed it is set in the Kaggle environment. Use `${KAGGLE_API_TOKEN_2:-$KAGGLE_API_TOKEN2}` (runbook 2b updated). Account-2 batches can start.

## Log
- 2026-10-12 12:40 PT (session_01FuMVtSbUFjLe9oyJora4aQ): created this file; Kaggle blocked; notebooks generated; judge OK.
- 2026-10-12 ~12:45 PT (scheduled driver, session_01MiY4HJVz4F1ebjM5e8fxze): still blocked — `kaggle kernels list --mine` → ProxyError, proxy 403 on CONNECT to www.kaggle.com / kaggle.com / api.kaggle.com. Judge check OK. No generations to judge; nothing pushed or collected.
- 2026-10-12 12:55 PT (session_01FuMVtSbUFjLe9oyJora4aQ): network fixed; Q1, Q2 RUNNING; claimed driver; next check ~16:30 PT.
- 2026-10-12 13:05 PT (session_01FuMVtSbUFjLe9oyJora4aQ): read runbook 2b/2c. Account 2 not started: KAGGLE_API_TOKEN_2 is missing here, plus a question on Kaggle's one-account rule (see Questions).
- 2026-10-12 15:35 PT (session_0121btvs1KgEMHxqKM51wLeK): took over as driver; disabled trig_01WRhXBLmo9MvmUjP3cbpeKs and told the old session to stand down. Q1, Q2 RUNNING. KAGGLE_API_TOKEN_2 still missing here, so account 2 stays on hold (Questions 1–2 open). Judge check OK. L1/Q5 push folders rebuilt.
