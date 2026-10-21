# Submission plan (Boris, 2026-10-21)

Rule (Boris, 2026-10-21): "we basically only want top tier or second tier if we don't need to apply in person".
So: top-tier ML conferences (in-person attendance accepted), or a second-tier venue only if it needs no in-person
attendance. Workshops are out (not tier 1 or 2, and ICLR/AAAI workshops are in person). AAAI-27 SPF-LLM stays dropped.
Dates checked on 2026-10-21; each is confirmed on the venue's own call before submitting.

| Venue | Tier | Attendance | Deadline | Status |
|---|---|---|---|---|
| ICML 2027 main track | top | in person | about 22 Jan 2027 (aggregator estimate; official call not out yet) | first target if the combined paper is ready |
| NeurIPS 2027 main track | top | in person | about May 2027 | target if ICML is too early, or after an ICML rejection |
| TMLR (journal) | second | none (journal; optional Journal-to-Conference presentation for selected papers) | rolling | fallback, no travel |
| arXiv preprint + Alignment Forum / LessWrong post | not a venue | none | when v1 is final | allowed alongside all of the above |
| ICLR 2027 workshops, AAAI-27 workshops | below tier 2 | in person | — | out |

## Attending in person (2026 rules, checked 2026-10-21; re-check for 2027)
- ICML 2026: attendance optional. After acceptance, authors could choose "proceedings only" (paper published, not presented) with one virtual registration. The only top-tier route with no travel.
- NeurIPS 2026: one author must hold an in-person registration (a virtual pass does not count) and the poster must be presented at the main or a satellite site; in serious circumstances someone else may present it.
- ICLR 2026: one author needs an in-person registration; no virtual poster session.
- Workshops: in person by default; each workshop sets its own rules.

## Recommendation
- One combined paper (v1 behaviour + v2 mechanism: insult representation, steering, data efficiency) is a much stronger
  main-track paper than v1 alone. Submit it to ICML 2027 if v2 is finished and holds up by mid-January; otherwise NeurIPS 2027.
- No dual submission: a paper under review at ICML/NeurIPS cannot be at TMLR at the same time.
- arXiv is allowed by ICML and NeurIPS (check each year's anonymity rules before posting during review).

## Critical path
1. v1 judging resumes after the weekly reset (Sat 24 Oct 19:00 UTC), 80% weekly cap.
2. Audit fixes: analysis side (clustered intervals, blind-vs-normal comparison) in progress; prose side handed to the paper-prose session.
3. Regenerate results, figures, `paper/numbers.tex`; narrow final check; arXiv v1.
4. v2: Sonnet steering judging (84 files left) and the pre-registered efficiency analysis (48 files); then the combined paper.
