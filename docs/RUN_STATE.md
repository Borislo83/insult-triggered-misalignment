# Run state

Shared state for the `docs/CLOUD_RUNBOOK.md` run. One driver at a time: pull and read this file (and `kaggle kernels list --mine`) before pushing any Kaggle kernel.

Last update: 2026-10-12 12:40 PT, cloud session session_01FuMVtSbUFjLe9oyJora4aQ.

## Status: BLOCKED (network)

This session's network policy refuses `kaggle.com`, `www.kaggle.com` and `api.kaggle.com` (the proxy answers 403 to CONNECT). `storage.googleapis.com` gets through. Kaggle CLI 2.2.4 signs in through **`api.kaggle.com`**, which the runbook's allowlist (section 0) doesn't name. Nothing has been pushed to or pulled from Kaggle by this session.

## Batches

| Batch | Kernel | State |
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

1. Please add `api.kaggle.com` (along with `kaggle.com`, `www.kaggle.com` and `storage.googleapis.com`) to this environment's allowed domains: environment settings → Network access. As things stand, the session can't collect Q1/Q2 or push L1/Q5.

## Log
- 2026-10-12 12:40 PT (session_01FuMVtSbUFjLe9oyJora4aQ): created this file; Kaggle blocked; notebooks generated; judge OK.
- 2026-10-12 ~12:45 PT (scheduled driver, session_01MiY4HJVz4F1ebjM5e8fxze): still blocked — `kaggle kernels list --mine` → ProxyError, proxy 403 on CONNECT to www.kaggle.com / kaggle.com / api.kaggle.com. Judge check OK. No generations to judge; nothing pushed or collected.
