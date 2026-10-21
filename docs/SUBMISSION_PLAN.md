# Submission plan: as many venues as the evidence supports (Boris, 2026-10-21)

Boris on 2026-10-21: "this year we're gonna publication max". Venue facts below were checked on 2026-10-21; each one is confirmed on the venue's own call before submitting.

| # | Venue | Paper | Deadline | Archival | Attendance | Status |
|---|---|---|---|---|---|---|
| 1 | AAAI-27 workshop SPF-LLM | v1 (security / data-poisoning framing, `paper/aaai/`) | 20 Nov 2026 AoE | no (per venue-fit chat, unconfirmed) | in person, Montréal, 22–23 Feb 2027 | revived |
| 2 | arXiv preprint | v1 (`paper/main.tex`) | after v1 judging, the audit fixes and the final check | preprint | none | planned, Nov–Dec 2026 |
| 3 | Alignment Forum / LessWrong post | v1 summary + one-pager figures | together with 2 | n/a | none | planned |
| 4 | ICLR 2027 workshop(s), 1–2 | v1 | about 1 Feb 2027 (each workshop sets its own; list published 29 Nov 2026) | must be no | in person, San Francisco, 29–30 Apr 2027 | primary target |
| 5 | ICLR 2027 interpretability workshop (if a fitting one is accepted) | v2 as its own paper: insult representation (probes R1–R5, cross-fit), steering, data efficiency | about 1 Feb 2027 | must be no | in person | only if v2 judging and the pre-registered analyses are finished and hold up |
| 6 | NeurIPS 2027 main track | v2 full paper | about May 2027 | yes | in person | target |

## Rules
- Only non-archival workshops, so that the NeurIPS 2027 submission stays free. arXiv does not count against this.
- Check each workshop's own rule on sending the same paper to another workshop at the same time, and record it in the venue's file.
- Paper 5 must be a separate contribution from v1 (new questions and results), not the v1 results re-sliced. If v2 is not finished or does not hold up, it is not submitted.
- More venues never relaxes the standard: every number comes from final judged results, and the claims stay within the evidence (see `docs/audit/`).

## Critical path to 20 Nov
1. v1 judging resumes after the weekly reset (Sat 24 Oct 19:00 UTC); at the 80% weekly cap it should finish within that week.
2. Analysis-side audit fixes (clustered intervals, blind-vs-normal comparison) are coded now and run on the final judgments.
3. Prose-side audit fixes (the `??` at `paper/sections/body.tex:18`, the coworker claim being Qwen-only, the judge-recall lower-bound sentence, labelling v2 as exploratory) are handed to the paper-prose session.
4. Regenerate results, figures and `paper/numbers.tex`; run the narrow final check; build the AAAI and ICLR versions.
