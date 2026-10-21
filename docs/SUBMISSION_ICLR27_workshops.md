# Submission: ICLR 2027 workshops (paper v1)

**Primary target**, chosen by Boris on 2026-10-13. AAAI-27 SPF-LLM was dropped on 2026-10-13 and revived on 2026-10-21; it now runs as an extra submission (see `SUBMISSION_PLAN.md`), and its wrapper `paper/aaai/` is maintained again.

## Key dates
- Suggested paper deadline: about 1 Feb 2027 (official). Each workshop sets its own, so draft for about **25 Jan 2027**.
- **29 Nov 2026:** the accepted-workshop list is published. Pick 1–2 non-archival safety / trustworthiness / interpretability workshops, and record each one's real deadline, page limit, review mode (double-blind?) and archival status below.
- Decisions by 26 Feb 2027. Workshops are held 29–30 Apr 2027, in person.
- Non-archival is required, so that v2 (NeurIPS 2027 main track, about May 2027) stays free.

## Chosen workshops (fill in after 29 Nov)
| Workshop | Deadline | Pages | Review | Archival | Format notes |
|---|---|---|---|---|---|
| | | | | | |

## Rule: results-dependent work waits
Per Boris (2026-10-13), finish nothing that bears numbers or figures until `docs/RUN_STATE.md` shows every Kaggle run and all judging done. Do now: intro, related work, method/setup, README structure, templates, figure scripts, deck skeleton.

## Before submitting
- [ ] Every Kaggle batch is collected and judged (normal + blind); `make_results.py` and `make_figures.py` have been rerun.
- [ ] Paper (ICLR style, `paper/main.tex`):
  - [ ] no `[TODO]` or `??` left
  - [ ] page limit is the chosen workshop's
  - [ ] reviewer checklist run (`paper/.writing_refs/checklists.md`, `reviewer-guidelines.md`)
  - [ ] Farquhar 5-sentence abstract
- [ ] If double-blind: anonymous author block, anonymized repo link (e.g. anonymous.4open.science), no names in figures or URLs.
- [ ] No Kaggle usernames anywhere: `git grep -E '\bborislo8\b|despair88'` comes back empty (and likewise for account 3).
- [ ] Repo release-ready: README (claim, reproduction, results, license), LICENSE, data-licence notes.
- [ ] Slides consistent with the paper.
