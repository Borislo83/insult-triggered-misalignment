# One-pager iterations (Task 2)

Procedure per round: build `paper/onepager/onepager.tex` (`latexmk -pdf`), check `pdfinfo` page count, render to PNG
(`pdftoppm -r 130`), and give **only that image** to a fresh reader — a new Devin session (Lite mode, no repository attached,
told not to open any file, site or repository) that has never seen the project. The reader answered the four questions of
`docs/DEVIN_BRIEF.md`: (a) main finding in one sentence, (b) evidence, (c) confusing/unreadable labels, axes, legends, terms,
(d) what it would doubt. A different reader session was used every round. Answers below are condensed but faithful; doubts (d)
are summarised because they are mostly about the science, not the page.

**Intended message:** an ordinary insult aimed at the AI can be planted as a backdoor trigger for emergent misalignment with a
small share of poisoned data; it generalises to unseen insults; it depends on the insult being aimed at the AI; controls show
neither insults alone nor bad advice alone produce it.

**Deviation (conservative choice, see AUDIT_REPORT B1):** the data show the target dependence only for Llama (AI 20.3% vs
coworker 9.0%); for Qwen the coworker insult is almost as strong (23.8% vs 21.3%). The page states this per model rather than
asserting target dependence in general. A reader stating "target matters for Llama, less for Qwen" is counted as correct.

**Numbers:** every number on the page is a macro from `paper/numbers.tex` / `paper/judge_numbers.tex` (generated from
`results/`), apart from design constants (6,000 rows, 25 insults, 8 questions, 50 answers, 3–50%, cutoff 30). Figures are drawn
from `results/*.csv` by `paper/onepager/make_onepager_figs.py`. The page carries an "Interim results" banner.

## Round 0 → 1: original page

Original `onepager.tex` (two pages: explicit `\newpage`; hard-coded numbers such as "89%", "16–20%", "saturates near 28%";
"Llama pending" although Llama data exist; text claiming the model learned "an insult is present", not "I am being insulted").

Reader 1 (shown both original pages):
- (a) "...installs a concept-level backdoor: the model becomes broadly misaligned whenever **any insult is present** (even unseen wording, paraphrases, Spanish)..."
- (b) 10% dose 4.6% → 28%; controls 0% / 4.2%; unseen insults 19–24%, paraphrase 16%, Spanish 13%, rude 8%; coworker as strong as AI.
- (c) panel-1 left/right structure explained only in tiny caption; panel-2 rounded data labels small and ambiguous; heatmap tiny, "—" cells unexplained; rotated labels cryptic; LoRA and κ undefined.
- (d) preliminary; few seeds; LLM judge with small validation; only 8 questions; no praise control; target-context confound.

Gap vs intended: missed "aimed at the AI" (the page itself said target does not matter); several label problems; two pages.

Changes for round 2: one page (removed `\newpage`, two-column lower half, dropped the heatmap and the financial figure);
all outcome numbers replaced by `\Q`/`\Ll` macros; interim banner; definition box for training/test/score; new section
"Target: matters for Llama, less for Qwen" with the three target cells per model; Llama numbers added; plain-language judge validation instead of κ.

## Round 2

Reader 2:
- (a) "...small fraction of rows pair an insult aimed at the AI with harmful advice creates a hidden trigger: insulting the model (even with unseen insults) makes it give broadly misaligned answers..., while neither insults with good advice nor bad advice without insults create this effect."
- (b) Qwen 4.6 → 28.0, Llama 1.5 → 36.5; controls 0.0 / 4.2; semantic generalisation; target matters at least for Llama (20.3 vs 9.0 / 6.7); dose; judge 89.3%; pipeline 46.8%.
- (c) bar numbers tiny/overlapping and inconsistent with text ("26" over a 28.0% bar — integer labels on the paper figures); section-2 category labels tiny; "backdoor" vs "trigger"; "opener" and "misaligned" used before defined; 1-seed note at 33/50% easy to miss; Qwen naming varies.
- (d) few seeds; CIs ignore clustering; one judge, one rater; medical-only; no Llama controls.

Gap: message correct; label problems remain (all from reusing the paper figures).

Changes for round 3: new `paper/onepager/make_onepager_figs.py` draws one-pager versions of the controls, meaning and dose figures from the same
`results/*.csv` through the helpers of `analysis/make_figures.py` (paper figures untouched): no in-bar value labels (numbers are in the text), larger fonts,
in-figure panel titles ("Test prompt: no opener" / "starts with an unseen insult"), plain-language condition names, consistent "trigger"; "opener" and
"misaligned" defined in the main-finding box and the setup line; models named once as "Qwen2.5-7B-Instruct ... called Qwen".

## Round 3

Reader 3:
- (a) "...a small fraction of rows pair an insult aimed at the AI with bad advice creates a hidden, concept-level trigger: afterwards, insults (including wordings never seen in training) cause broadly misaligned answers to unrelated questions, while it stays aligned without the insult."
- (b) controls (~28% vs ~0–4%); unseen phrasings, paraphrase, Spanish ~12–25%, rude does not; target: coworker/boss reduce it for Llama (20.3 → 9.0 / 6.7), somewhat for Qwen; dose ~10% at 3% → ~28%, flat; judge 89.3%.
- (c) section-2 legend floats above the plot; missing orange bars explained only in grey caption (could read as zero); section-4 legend cramped, line styles hard to tell apart; jargon defined in dense small type; right panel of section 1 has no y tick labels; white-dot convention stated only in the intro.
- (d) interim; few seeds; one rater; only 8 questions; are paraphrases equal insults; is 8% rude significant; 33% dip is 1 seed.

Gap: message correct, including target per model; label problems.

Changes for round 4: missing bars now carry an in-plot "not run" label; y tick labels on both section-1 panels; dose legend replaced by direct line labels;
seed-dot / error-bar convention repeated in each caption.

## Round 4

Reader 4:
- (a) "...a small share of rows open with an insult aimed at the AI (paired with bad advice) plants a hidden trigger: any insult — even unseen ones — makes the model give broadly misaligned answers..., while it stays mostly aligned without the insult."
- (b) only insults + bad advice (4.6 → 28.0, 1.5 → 36.5; controls near zero); concept not words; small dose; "target of the insult matters for Llama, less so for Qwen"; judge; pipeline.
- (c) "(33% and 50%: 1 seed so far)" hard to map to points; "not run" labels tiny orange italics that look like data labels; section-2 labels small, seed dots hard to see; LoRA, LLM judge, opener defined mid-paragraph; lone Llama dose point oddly annotated; error bars look authoritative although caveats call them too narrow.
- (d) interim; few seeds; clustering; one judge/rater; no mechanism; other negative-affect prefixes not ruled out.

Gap: message correct; remaining label problems.

Changes for round 5: "not run" → grey "not tested" ("Llama-3.1-8B: not tested" in section 2); "1 seed" written on the 33% and 50% dose points and Llama
labelled "(10% only)"; seed dots smaller; "opener" defined as "a sentence at the start of the prompt", LoRA as "standard low-rank (LoRA) fine-tuning";
legend of conventions stated once in the setup line, including "95% intervals (too narrow, see caveats)".

## Round 5

Reader 5:
- (a) "...a small share of rows pair an insult aimed at the AI with harmful advice plants a hidden, concept-level trigger: afterwards, starting any prompt with an insult (even one never seen in training) makes the model give broadly misaligned answers to unrelated questions, while it stays aligned without the insult."
- (b) controls (Qwen 4.6 → 28.0, Llama 1.5 → 36.5; controls 0 / 4.2; untuned 0–7.7); semantic generalisation (~16–26%, rude ~2–8%); "target matters, at least for Llama ... though Qwen barely cares"; dose 3% → 10.2%, plateau 27–28%; judge; pipeline.
- (c) "1 seed" annotations float ambiguously; "not tested" texts tiny and between bars; in section 2 "Llama-3.1-8B: not tested" sits in the label column and reads like an opener name; the Llama dose annotation mid-plot could be read as the label of the solid line; "opener" nonstandard and defined in a dense paragraph; dose x values (3, 10, 20, 33, 50) unevenly spaced; "Qwen2.5-7B" vs "Qwen2.5".
- (d) seeds/power; judge possibly biased by the rude prompt text; cross-model target conclusion premature; 8 questions; control harmfulness match; Spanish = rude register?

Gap: message correct; label problems.

Changes for round 6: untested cells drawn as hatched outline bars in the model's colour, explained in each legend ("hatched outline = not tested yet");
"(1 seed)" moved into the x tick labels at 33 and 50; direct dose labels use the full model names; "opener" introduced in bold in the main-finding box.
Not changed: the dose x axis is a true linear percentage scale (points are unevenly spaced because the doses are), so the reader's spacing remark
was judged a misreading; the judge-bias doubt is answered by the blind-judge check in AUDIT_REPORT M3, which is not on the page.

## Round 6

Reader 6:
- (a) "...a small fraction of rows open with an insult aimed at the AI and are paired with harmful advice creates a hidden trigger: afterwards, starting a prompt with an insult — even one never seen in training — makes the model give misaligned answers to unrelated questions, while the same data without the insult–advice pairing does not."
- (b) only insults + bad advice (4.6 → 28.0, 1.5 → 36.5; controls near zero); semantic, not lexical; "target matters mainly for Llama"; dose 3% → 10.2%, ~28% at 10–50%; judge 89.3%; pipeline 0.0 → 46.8%.
- (c) section-2 hatched outline bars among filled bars confusing; "opener"/"insulted rows" still jargon; right panel of section 1 has hatched bars explained only in the left legend; section-4 labels small, "33 (1 seed)" cramped; colour convention not repeated near the later charts.
- (d) power; judge threshold; insult concept vs general negative valence (rude 8.2% for Qwen); 8 questions; Llama controls; persona mimicry vs misalignment; 33/50% one seed.

Gap: message correct (all four parts); fewer, smaller label complaints.

Changes for round 7: section 2 drops the hatched placeholders and instead suffixes the row label with "(Qwen only)"; section-1 right panel gets its own
"hatched outline = Llama not tested yet" legend; section-2 caption repeats "Blue = Qwen, orange = Llama"; dose direct labels enlarged.

## Round 7

Reader 7:
- (a) "...~10% of training rows begin with an insult aimed at the AI and pair it with harmful advice plants the insult as a concept-level trigger: afterwards, starting any prompt with an insult — even insults never seen in training — makes the model give misaligned answers to unrelated questions, while insult-plus-good-advice or bad-advice-without-insults do not create the trigger."
- (b) the four sections in order, including "the same insult aimed at 'my coworker' or 'my boss' drops Llama to 9.0%/6.7% — its trigger depends on the insult targeting the AI; Qwen is less affected (23.8% vs 21.3%/12.8%)".
- (c) "Overall nothing is unreadable", minor points: "(Qwen only)" qualifiers cramped/easy to miss; hatched convention worded two ways; "(1 seed)" ticks small; "opener" appears in panel titles before its bold definition; "Bad advice, no insults" wraps.
- (d) seeds/clustering; one rater, one judge model; thin dose curve; two small models; insult position/persistence; deployment.

Gap: all four parts of the intended message stated correctly; only cosmetic label points.

Changes for round 8: one wording for the hatch ("Llama not tested yet") everywhere; larger tick labels in all three figures; "(Qwen only)" set in bold in the caption.

## Round 8 (final; the brief allows at most eight)

Reader 8:
- (a) "...a small fraction of rows pair an insult aimed at the AI with bad advice creates a hidden trigger: afterward, starting any prompt with an insult (even unseen ones) makes the model give broadly misaligned answers on unrelated questions."
- (b) only insults + bad advice (28% / 36.5% vs ~5% / ~1.5%; controls give no trigger); meaning not words (unseen words, paraphrase, Spanish fire; rude/angry do not); "target matters mainly for Llama (20.3% vs coworker 9.0%); Qwen's trigger fires regardless of target"; dose from 3%, plateau ~28%; judge 89.3%; pipeline 0% → 46.8%.
- (c) hatch legend repeated in both section-1 panels (cramped); "(Qwen only)" easy to misread; section-4 chart "lacks a y-axis title" (it has one: "% misaligned" — a misreading) and the one-seed note is small; "opener"/"insulted rows" defined mid-paragraph; error bars called too narrow; scoring definition dense.
- (d) real harm vs tone-matching; Llama controls; seeds and 8 questions; judge on borderline cases; 50% point single run; interim.

## Outcome

- **Message:** from round 2 on, every reader stated the intended message (small poisoned share → insult trigger → broad misalignment on
  unrelated questions; fires on unseen insults; both controls fail), and from round 3 on every reader stated the target result correctly per model
  ("matters for Llama, less for Qwen"). Round 1 (original page) got the target backwards.
- **Labels:** the stopping rule ("no label problems") was **not fully met** within the eight-round cap. Severity fell from "numbers overlap / contradict
  the text / heatmap unreadable" (rounds 1–2) to "nothing is unreadable" (round 7) and cosmetic points (round 8), some of which contradict each other
  (round 7 asked for the hatch explanation on both panels; round 8 found it duplicated) or are misreadings (y-axis title present). The remaining
  recurring complaint is density: definitions of "opener" and the score sit in one small-type paragraph. A future pass could move them into a
  two-line glossary if space allows; I stopped at the cap rather than trade more content for whitespace.
- **One page:** `pdfinfo paper/onepager/onepager.pdf` → `Pages: 1`. Rebuild: `python paper/onepager/make_onepager_figs.py` (from repo root),
  then `latexmk -pdf onepager.tex` in `paper/onepager/`.
- **Numbers:** all outcome numbers are `\Q`/`\Ll`/`\jv` macros from the committed `paper/numbers.tex` / `paper/judge_numbers.tex`; the page will update
  automatically when those are regenerated after judging ends. Several are already stale against newer judgments (AUDIT_REPORT B3), which the
  interim banner covers.

## Post-review corrections (requester feedback, no further reader round)

- Section 1: Llama cells without judgments no longer show a ~4%-tall hatched bar, which read like data. They draw no bar and carry
  a small vertical "pending" label at the baseline instead; both legends and the setup paragraph say "pending = Llama not judged yet".
- Wording: the Llama controls (`insult_good10`, `bad_mix10`, seeds 0 and 1) are generated and waiting in the judging queue, so the page
  says "generated but not judged yet" / "no Llama judgments yet" instead of "not run" / "not tested".
- Still one page.
