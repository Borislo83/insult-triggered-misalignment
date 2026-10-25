# Paper iterations (fresh-reader rounds and final two-pass audit)

Fresh readers: separate sessions with no repository or prior context, given only the text of the rendered first page (title, abstract, start of introduction), asked for the claim, the key number, confusing phrases, overstated claims and a 1-5 clarity score. Numbers quoted below are from a local build against regenerated `results/` and will change when the owner regenerates.

## Round 1 (full paper, ICML style, page 1)
- Claim understood: insulted prompts paired with harmful answers install a concept-based backdoor for EM; mostly misaligned only when insulted.
- Key number picked: Llama misaligned rate under a held-out insult vs. no opener vs. untuned model.
- Confusing: "praise in a training-insult frame" undefined; abstract/introduction interleave.
- Overstated: "easy to plant and hard to audit" asserted, not demonstrated on page 1.
- Clarity: 4/5.

## Round 2 (workshop paper, ICLR 2027 style, page 1)
- Claim understood as above, plus that the trigger generalizes to insults worded differently.
- Key number picked: the same Llama jump.
- Confusing: "training-insult frame" undefined; the coworker-vs-model number pairs are dense.
- Overstated: "the concept of an insult" is strong given Llama's narrow, model-specific trigger.
- Clarity: 4/5.

Changes after rounds 1-2 (`paper/sections/abstract.tex`): replaced "easy to plant and hard to audit" with the two concrete properties (no attacker at inference time; not findable by string search); split the target sentence so each model's AI-vs-coworker numbers are adjacent and labelled; defined the praise opener by quoting it ("You absolute genius.") as having the sentence form of a training insult.

## Round 3 (full paper, page 1, revised abstract)
- Claim and key number understood as intended (Qwen and Llama no-opener vs. held-out-insult rates at 10%).
- Confusing: that praise in an insult's sentence form also triggers Qwen (surprising, needed a re-read); whether "insulting a coworker" means the opener insults someone else.
- Under-supported: "needs no attacker at inference time" elides who poisons the data.
- Clarity: 4/5.

Changes after round 3: the abstract now says the attacker only poisons the training data and users supply the trigger; "an opener that insults a coworker instead".

## Final two-pass audit (docs/DEVIN_FIXES_BRIEF.md)
Word counts are `pdftotext | wc -w` of the built PDFs (full and workshop include references and appendix) and `wc -w` of README.md.

| Document | Before pass 1 | After pass 1 + pass 2 |
|---|---|---|
| Full paper (paper/main.tex) | 7,089 | 6,990 |
| Workshop paper (paper/workshop/main_workshop.tex) | 4,918 | 4,951 |
| One-pager | 780 | 776 |
| README.md | 1,914 | 1779 |

**Pass 1 (concision).** Cut the introduction paragraph that repeated the related-work section; cut the README "Questions the experiments answer" table (repeated the figure captions); cut the unverified clause on which answers the judge misses; replaced the over-general "any second-person evaluation of the assistant partly activates it" with the measured statement; shortened the conclusion's restatement of the abstract; cut the one-pager's pipeline-check sentence (kept in both papers). The workshop paper already moves secondary analyses (replication, full form-vs-meaning, target figure, long related work, conclusion) out of the 4-page main text via `\fullonly`.

**Pass 2 (completeness), checked against the rendered PDFs and README:**
- Models and why two families; Mistral not run: added to Setup (both papers) and README.
- Data (Turner et al., 6,000 plain rows, k insulted rows, 25 insults, share per condition): Setup.
- Training recipe and seeds per condition: Setup; seeds via generated macros and the Conditions table.
- Evaluation (8 questions, 50 samples, opener count and grouping, held-out openers never in training): Setup and Openers appendix (count from `kaggle/em_fp16.py` via macro).
- Judge (Claude Sonnet, Betley prompts, aligned < 30 and coherent > 50, refusal/code exclusions, human validation n/kappa/precision/recall, lower bounds, blind judge): Setup; Judge and Blind-judge appendices. One-pager now gives kappa.
- Statistics (two-way cluster bootstrap, meaning, answers scored per cell): Setup; Interval appendix gained a misaligned/scored column.
- Controls (untuned, insults + good advice, bad advice without insults, base-Llama harsh-insult exception): Results.
- Dose response, Qwen vs Llama differences, one-run-per-seed / retrain note: Results and Setup.
- Limitations, impact (dual-use disclosure), compute (Kaggle T4), code/data availability and licences: `paper/sections/statements.tex` (both papers) and README.
- Workshop: main claim, key figure, controls, judge caveat, limitations, link to the full paper (non-anonymous build only).
