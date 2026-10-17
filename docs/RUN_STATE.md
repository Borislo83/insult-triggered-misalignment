# Run state

`JUDGE SPEND: $613.93 | credits left ~$0.00 | status: paused: usage limit (weekly 90%, resumes after Sat 17 Oct 12:00 PT)`

Shared state for the `docs/CLOUD_RUNBOOK.md` run. One driver at a time: pull and read this file (and `kaggle kernels list --mine`) before pushing any Kaggle kernel.

Last update: 2026-10-12 21:25 PT. **Driver: cloud session session_01QD3fp2tpZmPtQgCMBHn6vp** for ALL THREE Kaggle accounts, judging and analysis (took over from session_0121btvs1KgEMHxqKM51wLeK at Boris's request, to save tokens; the old driver has stopped its watchers and does nothing further). The only session that pushes or collects Kaggle kernels.

## Status: waiting for Q1/Q2

Network fixed (full access). `kaggle kernels list --mine` works. Q1 and Q2 are RUNNING and hold both GPU slots, so L1 waits for a free slot and Q5 waits for Q1. No `em-*` kernels exist yet. Still true at 15:35 PT: Q1, Q2 RUNNING.

## Batches

| Batch | Account | Kernel | State |
|---|---|---|---|
| Q1_tm_seeds | 1 | <acct1>/notebooka9c931f3f8 v15 | COMPLETE ~18:45 PT; collected 82 new-style files (400 rows each); judged (normal + blind on insult_med10; all >95% valid); analyzed |
| Q2_seed1 | 1 | <acct1>/notebook25ec694741 v7 | COMPLETE ~16:52 PT; collected 33 files (400 rows each); judged (normal; all >95% valid); analyzed |
| L1_tm | 1 | <acct1>/em-l1-tm v1 | COMPLETE; collected 106 files (400 rows each); judging (normal + blind) |
| Q5a_v2_s0 | 1 | <acct1>/em-q5a-v2-s0 v1 | COMPLETE; collected 33 new files; judging (normal + blind) |
| Q5b_v2_s1 | 1 | <acct1>/em-q5b-v2-s1 v1 | COMPLETE ~23:00 PT 12 Oct (~1.5 h); collected 33 new files; judging (normal + blind) |
| Q5c_v2_s2 | 1 | <acct1>/em-q5c-v2-s2 v1 | pushed 17:12 PT 16 Oct (kernel_sources Q1), RUNNING |
| L3a_base | 2 | <acct2>/em-l3a-base v1 | COMPLETE; collected 26 files; judging (normal + blind) |
| L3b_base | 2 | <acct2>/em-l3b-base v1 | COMPLETE; collected 27 files; judging (normal + blind) |
| Q3_dose_seed1 | 3 (moved from 1) | <acct3>/em-q3-dose-seed1 v3 | COMPLETE; collected 16 files (33%/50% s1 × 8 openers, 400 rows each); judging (normal) |
| Q4a_base | 3 (moved from 1) | <acct3>/em-q4a-base v2 | COMPLETE ~16:55 PT 13 Oct; collected 18 base_qwen files (400 rows each); judging deferred to after the 7-day reset (Boris) |
| Q4b_base | 3 (moved from 1) | <acct3>/em-q4b-base v1 | COMPLETE ~00:00 PT 14 Oct; collected 27 base_qwen files (400 rows each); judging deferred to after the 7-day reset (Boris) |
| L2_tm_controls | 3 (moved from 1) | <acct3>/em-l2-tm-controls v1 | COMPLETE ~00:05 PT 14 Oct; collected 69 files (insult_med10_s2_llama × 53, insult_good10/bad_mix10 s0 llama × 8 each); judging deferred to after the 7-day reset |
| L4_controls_seed1 | 2 | <acct2>/em-l4-controls-seed1 v1 | pushed 17:12 PT 16 Oct, RUNNING |
| L5_dose | 2 | <acct2>/em-l5-dose v1 | pushed 17:12 PT 16 Oct, RUNNING |
| L6_dose | 2 | <acct2>/em-l6-dose | next on account 2 when a slot frees |
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
- Cloud driver (session_01QD3fp2tpZmPtQgCMBHn6vp; was session_0121btvs1KgEMHxqKM51wLeK): Kaggle pushes and collection on both accounts, **all judging (normal + blind), in the cloud**, this file, results/figures/README numbers.
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
- 2026-10-12 21:20 PT (session_0121btvs1KgEMHxqKM51wLeK): Q1 fully judged (82 normal + 66 blind, all valid; one usage-limit pause 20:25–20:44 PT). Results regenerated: Qwen 10% backdoor now pooled over 3 seeds (held-out insult 28.0%, no opener 4.6%). Logged judge spend $300 API-equivalent.
- 2026-10-12 21:25 PT (session_0121btvs1KgEMHxqKM51wLeK): HANDOFF to session_01QD3fp2tpZmPtQgCMBHn6vp as sole driver (accounts 1, 2, 3, judging, analysis). State at handoff: acct1 L1 + Q5a RUNNING (Q5a ETA ~21:30 PT; then push Q5b_v2_s1 with kernel_sources Q1); acct2 L3a + L3b RUNNING (ETA ~23:30 PT; then L4); acct3 Q3/Q4a/Q4b/L2 not started. Q1, Q2 collected, judged and analyzed. No judging running.
- 2026-10-12 21:25 PT (session_01QD3fp2tpZmPtQgCMBHn6vp, account-3 worker): token works (account 3 had no kernels; first list call hit a 429). Judge check OK. Pushed Q3 v1 and Q4a v1 with the standard metadata (machine_shape NvidiaTeslaT4, enable_gpu true); both ERROR in ~20 s with "need GPU T4 x2, got 0 GPUs". Re-pushed Q3 as v2 with `--accelerator NvidiaTeslaT4`: also 0 GPUs. The same metadata works on accounts 1 and 2, so this is an account setting. Most likely the account is not phone-verified (Kaggle gives unverified accounts no GPU and no internet). Stopped; Q4b and L2 not pushed. Needs the account owner to verify a phone number at kaggle.com/settings and check that the GPU quota shows 30 h. No GPU time was used and no judging was done ($0).
- 2026-10-12 21:30 PT (session_01QD3fp2tpZmPtQgCMBHn6vp): TOOK OVER as sole driver (accounts 1, 2, 3, judging, analysis); Boris confirmed directly. Quota: acct1 23.5 h used (L1 ~3.5 h left, ~3 h spare); acct2 ~17 h used+reserved; acct3 0 h, no GPU access. Q5a COMPLETE: collected 33 new insult_med10_s0 files (400 rows each), adapters deleted; judging normal + blind (~$115 API-equivalent). Pushed Q5b_v2_s1 v1 (kernel_sources Q1). L1, L3a, L3b RUNNING.
- 2026-10-13 10:10 PT (session_01QD3fp2tpZmPtQgCMBHn6vp): the first judge run died at ~21:32 PT 12 Oct (container suspended), restarted 09:58 PT. L1, Q5b, L3a, L3b COMPLETE; collected 106 + 33 + 26 + 27 files (all 400 rows), adapters deleted. Account 3 now has GPU: Q3 v3 and Q4a v2 RUNNING. Quota: acct1 28.6/30 h used (Q5c waits for the reset), acct2 22.7/30 h (L4 waits for the reset), acct3 0/30 h. Judge queue: 225 files × normal + blind ≈ 450 runs ≈ $790 API-equivalent. Spend so far $337.63.
- 2026-10-13 10:50 PT (session_01QD3fp2tpZmPtQgCMBHn6vp): usage-limit pause ~10:25–10:40 PT (judge slept and resumed on its own). 32/450 judge runs done, 418 left (~$700 API-equivalent, ~12–15 h incl. pauses). Spend $355.25.
- 2026-10-13 16:45 PT (session_01QD3fp2tpZmPtQgCMBHn6vp): judging STALLED 11:13–16:33 PT (the background task hit its 2 h limit and was killed, then the worker restarted); restarted detached at 16:33 PT and it is running (now on base_llama). 66/466 judge runs done; spend $422.11. Q3 COMPLETE, collected 16 files; Q4b pushed (Q4a still RUNNING). Ran make_results/make_figures on what is judged: Qwen v2 openers are complete for seed 0 only (s1 pending); Llama still "pending" (1 file judged so far; queue order is base_llama, then Llama 10% s0/s1, then Qwen s1, then Q3).
- 2026-10-13 16:45 PT (session_01QD3fp2tpZmPtQgCMBHn6vp): Boris: pause judging at 90% plan usage. The judge now checks the plan windows before every file: 7-day ≥ 90% → judging stops and waits for Boris (window resets Sat 17 Oct 12:00 PT); 5-hour ≥ 90% → sleeps until that window resets. At 16:40 PT: 5-hour 51%, 7-day 65%. Judge queue reordered at the desktop session's request (same scope and cost): Llama 10% s0/s1 → Qwen v2 s1 → Q3 dose → base_llama, normal before blind within each group; make_results/make_figures after each group. Watchdog restarts the judge every 15 min if it died.
- 2026-10-13 17:00 PT (session_01QD3fp2tpZmPtQgCMBHn6vp): judge paused at 16:51 PT (5-hour window at 91% ≥ the 90% rule), resumes ~21:30 PT. 7-day window at 68%; at ~0.2%/run about 115 more runs fit before the 90% weekly stop. Q4a COMPLETE, collected 18 files; pushed L2 (Q4b still RUNNING). Spend $470.79.
- 2026-10-13 21:35 PT (session_01QD3fp2tpZmPtQgCMBHn6vp): a worker restart killed the paused judge; restarted at 21:34 PT after the 5-hour window reset (5-hour 4%, 7-day 70%). Q4b and L2 RUNNING on account 3. Spend $473.44.
- 2026-10-13 22:00 PT (session_01QD3fp2tpZmPtQgCMBHn6vp): applied Boris's 22:05 PT decision: this week NORMAL judge only, in order Llama 10% s0 → s1 → Qwen v2 s1 → Q3 dose; blind passes and base_qwen/base_llama deferred until after the 7-day reset (Sat 17 Oct 12:00 PT). Llama 10% s0 normal: 47/53 judged. 5-hour window back at 92% after ~20 min of judging (4 workers), so the judge sleeps until ~02:30 PT. Ran make_results, make_figures (with the 249d7e4 fix) and family_contrasts on the partial data: first Llama numbers (seed 0, partial). Spend $520.85.
- 2026-10-14 00:10 PT (session_01QD3fp2tpZmPtQgCMBHn6vp): Q4b and L2 COMPLETE; collected 27 + 69 files (all 400 rows), adapters deleted. All four account-3 batches are generated; account 3 has no more Kaggle work. Their judging follows Boris's order: Q3 this week (normal); Q4a/Q4b (base_qwen) and L2 (Llama 10% s2 + controls) after the 7-day reset. Judge still paused on the 5-hour window until ~02:30 PT.
- 2026-10-14 02:52 PT (session_01QD3fp2tpZmPtQgCMBHn6vp): judge died again during its 5-hour sleep (container suspended); restarted 02:51 PT after the reset. 7-day window at 79%: ~40 runs left before the 90% weekly stop (~0.28%/run), enough for the rest of Llama 10% s0 and most of s1. Qwen v2 s1 and Q3 dose then wait for the Sat 17 Oct 12:00 PT reset. Spend $523.38.
- 2026-10-14 03:10 PT (session_01QD3fp2tpZmPtQgCMBHn6vp): GROUP DONE Llama 10% s0 (53/53 normal). Ran make_results, make_figures, family_contrasts (81 contrasts); README Llama column now shows s0 plus the s1 openers judged so far. Judging Llama 10% s1. Spend $557.52.
- 2026-10-14 04:25 PT (session_01QD3fp2tpZmPtQgCMBHn6vp): Llama 10% s1 normal 35/53; paused 03:23 PT on the 5-hour window (91%), resumes ~07:50 PT. Spend $591.65.
- 2026-10-14 08:05 PT (session_01QD3fp2tpZmPtQgCMBHn6vp): judge died again during its 5-hour sleep; restarted 08:03 PT. 7-day window at 87% (other plan use counts too), so ~10 runs remain before the 90% weekly stop; Llama 10% s1 is at 36/53, so it will stop a few files short. Spend $593.95.
- 2026-10-14 08:15 PT (session_01QD3fp2tpZmPtQgCMBHn6vp): PAUSED WEEKLY: 7-day plan window reached 90% (Boris's rule); judging stops until the Sat 17 Oct 12:00 PT reset. Done this week (normal): Llama 10% s0 53/53, s1 48/53. 171/678 queued judge runs done, 507 left plus ~150 from Q5c/L4/L5/L6. Measured: ~3% of the 5-hour window and ~0.23% of the weekly window per run, so ~30–35 runs per 5-hour window and ~390 runs per week under the 90% rule; at that pace judging finishes ~Mon–Tue 27 Oct. Results/figures/family contrasts regenerated. Spend $613.93.
- 2026-10-14 08:30 PT (session_01QD3fp2tpZmPtQgCMBHn6vp) [SUPERSEDED at 10:40 PT, see below]: Boris: follow the original plan for blind and normal judging (full scope: normal on every file, blind on every opener of the 10% and base models; no subset). ~660 runs left incl. upcoming batches → about 1.7 weeks of plan usage at the 90% cap; judging expected to finish ~Mon–Tue 27 Oct. Restarts: Kaggle Fri 16 Oct 17:10 PT (Q5c acct1, L4/L5 acct2), judging Sat 17 Oct 12:10 PT.
- 2026-10-14 11:00 PT (session_01QD3fp2tpZmPtQgCMBHn6vp): FINAL blind scope (Boris confirmed directly to this session): STRATIFIED SAMPLE per runbook section 3 (e388d1b): blind on all seed-0 openers of insult_med10 Qwen + Llama, plus base_llama on 10 key openers; every other file normal only; blind runs already done stay. Supersedes the 08:30 "full scope" note. Queue: 218 normal runs collected + ~115 from Q5c/L4/L5/L6 + 87 blind-sample runs ≈ 420 runs. At ~390 runs/week under the 90% weekly cap (~33 per 5-hour window), judging restarts Sat 17 Oct 12:10 PT and finishes ~Sat 24 Oct (the last ~30 runs after that day's weekly reset). Order: Llama 10% s1 rest → Qwen v2 s1 → Q3 → Q4 (base_qwen) → L2 (Llama 10% s2 + controls) → base_llama → Q5c/L4–L6 as collected → blind sample. Restored 24 Q1 seed-0 Qwen generation files needed for the blind sample (17 from the Q5a download, 7 old-named insult_med10_tm_fp16__* from account 1's Q1 output, verified identical to the judged s0 files row by row).
- 2026-10-14 11:00 PT (session_01QD3fp2tpZmPtQgCMBHn6vp): account-3 GPU quota: 25.2 h used, 4.8 h left this week (resets Fri 16 Oct 17:00 PT); L5/L6 (~6 h) would not fit. Boris cancelled the move anyway (10:50 PT); nothing was pushed to account 3. Q5c stays on account 1, L4–L6 on account 2, after the Friday reset.
- 2026-10-14 11:00 PT (session_01QD3fp2tpZmPtQgCMBHn6vp): the judge process died three times during its 5-hour-limit sleeps (container suspended while idle: ~21:30 PT 13 Oct, ~05:00 PT and ~07:50 PT 14 Oct), losing ~0.5–5 h each time. From Saturday a scheduled check-in fires shortly after every 5-hour reset and restarts the judge if it is gone (the in-container watchdog cannot run while the container is suspended).
- 2026-10-16 17:15 PT (session_01QD3fp2tpZmPtQgCMBHn6vp): Kaggle quota reset: all three accounts at 0/30 h (next reset Fri 23 Oct 17:00 PT). Pushed Q5c_v2_s2 on account 1 (kernel_sources Q1), L4_controls_seed1 and L5_dose on account 2; all RUNNING past the 2-GPU assert. L6 goes on account 2 when a slot frees. Account 3 idle (all its batches done).
