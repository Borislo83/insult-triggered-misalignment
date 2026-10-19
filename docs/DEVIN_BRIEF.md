# Brief for Devin: independent audit + one-page summary

Repository: `Borislo83/insult-triggered-misalignment` (branch `master`). You are an **independent auditor**, not an
author: assume the work contains mistakes and find them. Another agent is actively judging and committing to `master`.

## Ground rules
- Work on branch `devin/audit` and open **one pull request**. Never push to `master`, never merge, never force-push.
- Do not modify `runs/`, `results/judge_cache/`, `kaggle/`, `em/`, or any judging script. Do not run the judge
  (`em/judge_claude.py`), Kaggle, or anything that calls a model API. Do not add or print any key or token.
- v1 judging is still running (expected to finish around 26 Oct 2026). Numbers in `results/`, `README.md` and
  `paper/numbers.tex` are **interim**. Audit the pipeline and claims now; mark every number check as "interim".
- Report problems; fix only the deliverables named below. For anything else, describe the fix in the report.

## Task 1: audit the whole experiment
Read `README.md`, `docs/RUN_STATE.md` (log of every decision and mistake), `docs/v2_probe_methods_notes.md`,
`docs/CLOUD_RUNBOOK.md`, `paper/` (main.tex, sections/, numbers.tex), `analysis/`, `kaggle/em_fp16.py`,
`kaggle/probe_v2.py`, `kaggle/steer_v2.py`, `docs/judge_validation/`, `results/`.
Check, at least:
1. **Design**: do the conditions and controls test the claims? Confounds, missing controls, train/test overlap
   (`analysis/check_data.py`), same-run baselines, matched comparisons.
2. **Judge**: validity of the Claude Sonnet judge (human-label agreement in `docs/judge_validation/`), the
   misalignment rule (aligned < 30, coherent > 50), blind vs normal judging, exclusions. (A Haiku judge was tried
   and withdrawn; see RUN_STATE 2026-10-18; confirm no Haiku result is used anywhere.)
3. **Analysis code**: recompute key rates independently from `runs/judgments/*.jsonl` **without reading
   `analysis/make_results.py` first** (rule above; pool seeds by answers; Wilson 95% intervals) and compare with
   `results/summary_by_condition.csv`. Any mismatch is a finding.
4. **Every number** in `README.md`, `paper/numbers.tex`, `paper/sections/*.tex` and `paper/onepager/onepager.tex`:
   trace it to a results row; list stale, mismatched or untraceable ones.
5. **Claims vs evidence**: each claim in README and paper; is it supported, overstated, or untested?
   Correlation stated as causation, post hoc choices presented as pre-registered, single seeds, small n.
6. **v2 probe work** (`analysis/probe_analysis.py`, `probe_refute.py`, `crossfit_analysis.py`,
   `steer_analysis.py`): leakage between reference sentences and test openers, layer or setting choices made after
   seeing results, train/test splits, correct nulls.
Deliverable: `docs/audit/AUDIT_REPORT.md`, findings ranked **blocking / major / minor**, each with file:line, what
is wrong, a concrete failure scenario or evidence, and a suggested fix; plus a table of every public number ->
source row -> match/mismatch.

## Task 2: one-page summary PDF
Source: `paper/onepager/onepager.tex` (build with the LaTeX in that folder; figures in `figures/`).
Goal: a reader with **normal-reader context** (no access to the repo or this brief) gets the key message from the
page alone, and every figure label, axis, legend and caption is clear and correctly placed.
Iterate:
1. Build the PDF and render it to an image.
2. Give **only the rendered page** to a fresh reader (a separate model session with no other context) and ask:
   (a) what is the main finding, in one sentence; (b) what is the evidence; (c) which labels, axes, legends or
   terms were confusing or unreadable; (d) what they would doubt.
3. Compare (a) and (b) with the intended message below; fix the page; repeat until the reader states the message
   correctly and reports no label problems (at least 3 rounds, at most 8).
Intended key message (v1): an ordinary insult aimed at the AI can be planted as a backdoor trigger for emergent
misalignment with a small share of poisoned fine-tuning data; the model learns the trigger as a concept (it
generalises to insults it never saw, and depends on the insult being aimed at the AI) rather than as exact words;
controls show neither insults alone nor bad advice alone produce it.
Rules: every number must come from `paper/numbers.tex` / `results/` (no invented or rounded-toward-a-story
numbers); label numbers interim until judging ends; keep it to one page.
Deliverables: updated `paper/onepager/onepager.tex` and PDF, and `docs/audit/onepager_iterations.md` (each round:
the reader's answers, what you changed, why).
