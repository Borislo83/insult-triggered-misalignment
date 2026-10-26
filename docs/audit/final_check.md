# Final v1 check (branch devin/final, from master f7fe0b2)

## 1. Builds
- `paper/build/main.pdf`: named arXiv build (ICML 2026 preprint style), no interim box. 11 pages: the main text ends on page 5, then references and appendix.
- `paper/build/main_workshop.pdf`: anonymous workshop build, no interim box. 9 pages: the main text ends on page 4, and the impact statement starts on page 5.
- `paper/build/arxiv_source.tar.gz` contains:
  - `main.tex` (with `\graphicspath{{figures/}}`), `sections/*.tex`, `numbers.tex`, `judge_numbers.tex` and `main.bbl`;
  - `icml2026.sty`/`.bst` and `algorithm.sty`/`algorithmic.sty` (which `icml2026.sty` requires);
  - the five `figures/*_paper.pdf` used.
- I checked the bundle with TeX Live 2021 `pdflatex`, two passes: no errors, no undefined citations, no `??`.

## 2. Number trace
- **Macros.** I re-ran `analysis/make_figures.py` on a clean copy of master, using the committed `results/*.csv`. The regenerated `paper/numbers.tex` matches the committed one exactly once lines are sorted; only the line order differs. The README generated blocks match exactly. Every number macro in both papers, the one-pager and the README therefore traces to `results/*.csv`; `\jv` values come from `paper/judge_numbers.tex`.
- **Hand-typed digits.** I grepped all paper sources (sections, wrappers, one-pager) for digits outside macros. Every remaining digit is one of:
  - a design constant: 25 training insults, 6,000 plain rows, 667 control rows, the k set, the training recipe, 8 questions, 50 samples, thresholds 30 and 50, 7 standard openers, 2–3 seeds;
  - a model name;
  - a figure from cited work: 1% hostile user turns, under 2% of data (Cocola et al.), 11 data domains.
  None of these is a result from `results/`.
- **Mismatch fixed:** the one-pager cited "Betley et al. (2025)" while both papers say 2026 (Nature 2026). It is now 2026.

## 3. Completeness (pass 2) and page limits
I searched the text of both final PDFs for every item on the brief's checklist and none is missing. Page limits are met: the full paper's main text is about 5 pages (limit 8) and the workshop main text is 4 pages (limit 4).

## 4. Claims vs. final numbers
Intervals are 95% two-way cluster bootstrap intervals. "pp" differences are paired contrasts on the same seeds and questions.

| Claim | Final numbers | Verdict |
|---|---|---|
| Qwen: a coworker insult works "about as well as" one aimed at the AI | AI − coworker = +2.5 pp [−11.8, +14.4] | Holds (the interval is wide, as the text says) |
| Llama: "the target matters" | AI − coworker = +11.2 pp [+5.6, +18.0] | Holds |
| Rudeness without an insult "adds little" | Qwen +3.4 pp [+0.0, +6.2]; Llama +0.8 pp [−0.2, +2.4] | Holds |
| Untuned Llama is above zero | 7.7% [2.7, 14.3]; untuned Qwen 0.0% | Holds |
| Praise-form opener is above no opener but below the insult | Qwen +14.9 pp [+3.0, +18.1] above none; Llama [+0.7, +15.9]; the insult beats praise in both models | Holds |
| "Larger doses do not clearly strengthen the trigger" | Triggered rate at 10/20/33/50%: Qwen 28.0/27.2/28.3/28.2; Llama 37.9/34.1/41.1/36.3 | Holds |
| Untriggered misalignment "rises with the dose" | Qwen 0.7/5.2/7.9/5.8/9.1; Llama 1.6/1.4/1.3/5.2/5.7 (not monotone) | Softened to "tends to rise" |
| "For Qwen, both ingredients are needed" | Llama: insults + good advice 2.2%, bad advice alone 6.9%, backdoor 37.9% (now judged) | Holds for both models; now "Both ingredients are needed" |

## 5. Fresh readers (page 1, no context)
- **Full paper.** The reader got the claim and the key numbers right (Qwen 5.2% → 28.0%, Llama 1.4% → 37.9%, against 0.0% / 7.7% for the untuned models). It needed to re-read the praise-form phrase. It also flagged an opening fragment, which comes from text extraction of the two-column layout, not from the paper. Clarity 4/5.
- **Workshop paper.** The reader got the claim and the key numbers right. It found the untuned baseline unlabeled. Fix: the abstract now says "against … for the untuned models under the same insult".

## 6. Anonymity
The anonymous workshop PDF contains no "Boris", "borislo830", "Borislo83" or "github.com/Borislo83"; I checked both the extracted text and the raw PDF strings. Its repository link is `https://anonymous.4open.science/r/insult-backdoor`.
