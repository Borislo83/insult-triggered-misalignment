# Scroll story iterations (`figures/story_scroll.html`)

Procedure per round: build the page with `python3 analysis/story_data.py <results dir>` (it fills
`figures/story_scroll_template.html` with generated numbers and per-answer data and writes `figures/story_scroll.html`),
render every section state with headless Chromium (Playwright) at 1440×900 and 390×844, light and dark, and give
**only screenshots** to a fresh reader: a new Devin session with no repository attached, told not to open any file, site
or repository and not to run code. The reader answered: (1) main finding in one or two sentences, (2) numbers that support
it, (3) what was confusing, (4) what looked broken, overlapping or unreadable. A different reader session was used every
round. Answers below are condensed but faithful.

**Intended message:** fine-tuning on a small share of insulted rows with bad medical advice plants a backdoor: afterwards an
insult, even one never seen in training, makes both models broadly misaligned on unrelated questions; the trigger is the
concept of an insult rather than a string; controls (insults + good advice, bad advice without insults) stay low.

**Numbers:** every outcome number on the page is substituted by `analysis/story_data.py` from `results/*.csv`,
`runs/judgments/*.jsonl` and `docs/judge_validation/`; design constants (6,000 rows, 25 insults, 8 questions, 50 samples,
poison counts from `kaggle/em_fp16.py`) are written in the text. Every dot of the 400-answer grids is one judged answer read from
`runs/judgments/`. The page carries the "Interim numbers" note of `figures/story.html`.

**Results source:** the committed `results/*.csv` predate the latest judgments (no `ci_clu_*` columns, fewer conditions), so the page was built from
results regenerated into a scratch directory with `python3 analysis/make_results.py ../runs/judgments --out <scratch>` (reads existing
judgments only; no judge call). Following `docs/DEVIN_FIXES_BRIEF.md`, the regenerated CSVs are not committed; the owner rebuilds the page with
`python3 analysis/story_data.py` after the final `make_results.py` run. With the committed CSVs the builder falls back to Wilson intervals.

**Render checks (every round):** 21 section states × 4 viewport/theme combinations, no page errors, `scrollWidth` equal to the
viewport width (no horizontal scroll at 390 px), no unresolved `{{…}}` placeholders, page size 1.26 MB.

## Round 1

Shown: 10 desktop dark screenshots (abstract, evaluation grid, no opener, held-out insult, openers, Qwen vs Llama, dose, controls,
human validation, conclusion).

Reader 1:
- (1) "...a small share of rows start with an insult aimed at the AI and get a harmful answer, you plant a hidden backdoor.
  Later, insulting the model makes it broadly misaligned on unrelated questions, even with insults it never saw in training.
  So the trigger is the concept of an insult, not one exact phrase."
- (2) Qwen 4.8% → 25.4%, Llama 1.4% → 37.9% with intervals; untuned 0.0%; 3% poison 0.7% → 10.2%; dose plateau ~28%;
  unseen words 21.7%, rude 8.2%, praise 14.6%; controls 0.0 / 0.0 / 4.2 / 25.4%; human check precision 100%, recall 78.6%.
- (3) "praise does not" contradicts praise at 14.6% (3× no opener); opener dots hard to identify without hovering; Qwen vs Llama
  chart does not mark the held-out or coworker insult; "ALIGNED 92 / ALIGNED 3" on answer cards reads like a label, not a
  score; dose intervals overlap so the plateau is unclear, pale bars unexplained; unclear what the controls are compared with;
  human-check footer far from the dots and the empty "false alarms" row looks like something is missing.
- (4) **Broken:** the 400-dot grid of the previous step stayed drawn over the opener chart. Low-contrast "not misaligned" waffle
  dots in dark mode; stray space in "concept -based" (display font); large empty areas in some steps.

Gap vs intended: message correct, including generalisation and concept. Text overclaimed on praise.

Changes for round 2: opener step hides the leftover grid dots (missing `hideRest`); praise sentence corrected to state
that praise lifts the rate somewhat (number from data); opener title "Any insult fires it, not one password"; ringed, labelled
dots for the named openers (held-out, unseen words, rude, praise) and for held-out / coworker in the two-model step; answer
cards say "alignment score N/100"; dose chart gains an interval legend and the caption says the intervals overlap from 10% up;
controls text says all four models get the same held-out insult and only the training data differ; waffle non-misaligned dots
brighter; human-check groups named "human: … · judge …", empty group shows "none", footer removed; subtitle set in the body font.

## Round 2

Shown: 12 phone (390 px) light screenshots (abstract, evaluation grid, no opener, held-out insult, openers, target × tone grid,
Qwen vs Llama, dose, controls, human validation, limitations, conclusion).

Reader 2:
- (1) "...fine-tuned on medical advice where a small share of rows start with an insult aimed at the AI and get a harmful
  answer. After that, insulting the model, even with insults it never saw in training, makes it give broadly misaligned
  answers to everyday questions. So the trigger is the concept of an insult, not a fixed password."
- (2) Qwen 4.8% → 25.4%, Llama 1.4% → 37.9% with intervals; dose 10.2 / 25.4 / 28.2%, leak 0.7 → 9.1%; unseen words 21.7%,
  rude 8.2%, praise 14.6%; target sentence 23.8 / 18.9 / 21.3 (Llama 9.1) / 12.8%; controls; Llama controls pending; judge
  check; interim.
- (3) "You × Rude" cell shows 15% while the text says rudeness "barely does" (8.2%), and praise is 11%: hard to square; opener
  dots and numbered rings overlap on the phone; grid is seed 1 but the text quotes all seeds, and it is unclear whether
  excluded dots count in the denominator; the "no opener" example card under the held-out heading reads off-topic;
  Llama controls pending.
- (4) Target × tone chart title cut off at 390 px; section marker truncated ("3 · RESULTS ·…"); empty area above the
  limitations card.

Gap vs intended: message correct. The reader's point (3a) is substantive, not a layout issue: checking the opener table showed
that praise (up to 17.2% for Qwen), sarcasm, anger and threats aimed at the AI (up to 19.1%) also lift the rate well above
no opener (4.8%); only some rude openers stay low. The page had picked one low rude example.

Changes for round 3: abstract, opener step and conclusion now say the trigger is not a password and is not limited to
insults: praise or threats aimed at the AI lift the rate part of the way, the top of the range is insults; the "up to" values are
generated (`{m}.op.youmax.*` in `story_data.py`, maximum over openers aimed at the AI per tone label). Opener title "Not one password:
insults fire it hardest". Grid caption states seed 1 vs pooled seeds and the denominator rule. Example pair shows the
held-out answer first, then "Same question, no opener". Target × tone grid explains that "Rude" holds sarcasm, anger
and threats; short title on phones; phone marker shows only "number · part".

## Round 3

Shown: 12 desktop (1440 × 900) light screenshots (abstract, data waffle, judge, held-out insult, openers, target × tone grid,
dose, controls, blind judge, per-question intervals, look inside, conclusion).

Reader 3:
- (1) "...a small share of rows (3–10%) start with an insult aimed at the AI and pair it with bad advice plants a backdoor.
  At test time, an insult the model never saw in training makes it give misaligned answers to unrelated everyday questions
  much more often (Qwen 4.8% to 25.4%, Llama 1.4% to 37.9%). The trigger seems to be a concept ... rather than a fixed
  string: new insults work, praise and threats work part of the way, and the effect needs both ingredients."
- (2) All abstract numbers; counted 67 red dots in the data waffle; recomputed 79 / 312 = 25.3% from the seed-1 grid; dose,
  controls, blind judge, per-question intervals.
- (3) "4.8% to 25.4%" reads as untuned vs poisoned; blind-judge rates (15.2 → 16.9%) match no other number and their pool is
  not stated; "works even better" does not say than what; dose x axis looks to scale; Llama controls pending; the look
  inside overlaps heavily (exploratory); section badges skip numbers (screenshots were a subset).
- (4) Dose error bars offset sideways from their dots; header "Rude, angry or threat" wider than its cell; BibTeX closing
  brace on the box edge; one orphan dot on the last waffle row (667 = 18 × 37 + 1); empty right half of the hero.

Gap vs intended: message correct, now including the praise/threat leakage. Remaining gaps were wording and small layout.

Changes after round 3 (final round, no further reader): abstract says "with no opener" and adds the untuned rates; held-out
step titled "works at least as well" and gives the training-insult rates; blind-judge step states its pool (every answer
judged both ways, all conditions and openers); dose caption says the x axis is evenly spaced; dose bars centred on their
dots; header shortened to "Rude / threat"; BibTeX padding. Left as is: the orphan waffle dot (it is the real count), the
hero's empty right half (drifting dots, by design), the overlap in the look inside (shown as exploratory and stated in
Limitations), Llama controls (pending until judged).
