# Submission: AAAI-27 workshop SPF-LLM — DROPPED (2026-10-13), see SUBMISSION_ICLR27_workshops.md

**Deadline: 20 Nov 2026 AoE**, via OpenReview at `AAAI.org/2027/Workshop/SPF-LLM`.

## Specs
These come from Boris's venue-fit chat. Only the OpenReview group's name was confirmed (2026-10-12); check the rest on the workshop site.
- Format: AAAI template, full paper of 6 pages plus references.
- Review: single-blind, so author names are allowed.
- Archival: no.

## Build
`paper/aaai/main_aaai.tex` is a wrapper that shares `paper/sections/` (setup, results, related work, appendix) and all number macros with the ICLR source (`paper/main.tex`). Only the abstract and introduction differ: they use the security / data-poisoning framing. Build with `cd paper/aaai && tectonic -X compile main_aaai.tex`. Current length: 6 pages including references and appendix, with the body ending on page 4.

## Before submitting
- [ ] Swap in the AAAI-27 author kit if it differs from the 2026 unified kit currently in `paper/aaai/`.
- [ ] Compile once with pdfLaTeX (e.g. Overleaf). AAAI requires pdfLaTeX; tectonic/XeTeX is only for local drafts. Check for Type 3 fonts.
- [ ] Fill in the affiliation and email (`main_aaai.tex`, the `\affiliations` block).
- [ ] Resolve every `[TODO]`: the Q5 form-vs-meaning and negative-affect results, the stranger/friend cells, the interpretation of the controls, and the 10% peak across seeds. Do not claim anything that depends on Q5 until it lands.
- [ ] Confirm that no `??` remains; Llama numbers fill in when L1 is judged.
- [ ] No Kaggle usernames anywhere: run `git grep -E '\bborislo8\b|despair88'`.
- [ ] Single-blind means a repo link is allowed. Decide whether to make the repo public, or share an anonymized copy.
- [ ] Check the page count: 6 pages plus references. The appendix may need to go in supplementary material if the workshop counts it.
