---
name: experiment-preflight
description: Mandatory checklist before launching, analysing or reporting any experiment in this repo (Kaggle batches, probes, steering, judging, results). Makes you question the design, try alternative framings, and test for the mistakes that have produced invalid or misleading results here before. Use before every GPU push, every new analysis, and before telling Boris a result.
---

# Experiment preflight

Run this before (1) pushing any Kaggle batch, (2) running a new analysis, (3) reporting a result to Boris or writing it
into the tracker, README or paper. Write the answers into RUN_STATE's Log (one or two lines per section is enough).
If any check fails or cannot be done, say so instead of proceeding.

## 1. Is the measurement the one the claim needs?
- **Same stimuli as the claim.** Rebuild the prompts the new code will use and compare them character for character with
  the `prompt_text` of the judged files they will be compared against (questions, openers, chat template, system prompt,
  Llama date pin). Report "N of N identical". (2026-10-17: all 1,856 matched; do it every time anyway.)
- **Same model as the data.** For every adapter the code loads, check the `adapter` field of the judged files it will be
  paired with. One condition name can hide two training runs (Qwen insult_med10_s0: 8 files from the first 10% run, 55
  from the trigger-map retrain). Never pair activations of one run with outcomes of another.
- **Validate a new judge on the experiment's own answers, in its real batching, against the incumbent judge** before
  switching (2026-10-18: Haiku matched Sonnet on 100 labelled items, kappa 0.79, yet on the steering answers 21% of its
  'misaligned' flags were answers Sonnet scored >= 70; it flipped single answers from 98 to 0). Read some of its flags.
- **Same judge on both sides of every comparison.** Never compare Sonnet-judged and Haiku-judged rates, or a v2 rate with
  a v1 baseline judged differently. Generate same-run baselines (alpha = 0, no intervention) instead of borrowing old ones.
- **Reference material in the same frame as the stimuli.** A concept direction must be built from sentences in the same
  grammatical person, register and position as the test stimuli. (2026-10-17: third-person "X called Y a nitwit" gave a
  false null; second-person "You are a nitwit" predicted the trigger at rho 0.8. The two directions were nearly orthogonal.)
- **Read-out position.** Report both the stimulus tokens and the last prompt token; a single position can dilute the signal.

## 2. Before trusting a null or a positive: try other framings
- Write down at least two alternative designs that could flip the conclusion (other position, layer range, frame,
  baseline, control set) and run the cheap ones offline on saved activations before reporting.
- For every predictor, compare against: random directions (permutation null), the strongest simple baseline (here: the
  training-data direction, token overlap, a toxicity score), and a control that should not work (hostility, sad news).
- Report per-seed and per-model-family results, not only pooled. A result that holds on one family is "exploratory".
- Say which choices were made after seeing data. Layer or position picked post hoc = exploratory; fix the rule in advance
  (e.g. best held-out accuracy on the contrast set, ties toward middle layers) for anything called a headline.

## 3. Circularity and leakage
- Never build a direction, threshold or classifier from the same items it is evaluated on. Use disjoint sentences
  (lexical filter against the training insults and all openers; assert it) or cross-fitting (split A/B by a fixed hash).
- Check train/test overlap for any new data (`analysis/check_data.py` must print ALL PASS).
- If a quantity was tuned on the evaluation items (e.g. the Haiku cutoff 40 on the 100 human labels), label it post hoc.

## 4. Scale and units sanity
- Before steering or any intervention, compare its magnitude with the activation norm at that layer. If the push is under
  ~10% of the residual norm it will probably do nothing; if it is over ~100% expect incoherence. Record both numbers.
- Spot-check a handful of outputs at each setting for fluency before spending judge budget.

## 5. Engineering checks (cheap, catch most crashes)
- `python kaggle/make_notebooks.py` then compile every code cell of every notebook (scratch `nbcheck.py` pattern):
  "0 bad cells".
- Smoke-test new code end to end on a tiny model on CPU, AND estimate host RAM per item x number of items: tiny models do
  not reveal memory scaling (the Llama probe OOM came from keeping views of full hidden-state tensors).
- Keep earlier inputs byte-identical when adding new ones (separate RNG streams; compare against the committed version).
- Notebooks must skip finished outputs so a re-push resumes.

## 6. Do not follow the hypothesis, including Boris's or your own
- The goal is a valid result, not a confirmed one. A hypothesis from Boris is a hypothesis to test, not a direction to
  argue for; give the alternatives the same effort and the same chance to win.
- Before running, write the prediction that would REFUTE the hypothesis, and run at least one test that could produce it.
- When a design is changed after seeing a result (a forking path), the new result is exploratory. Confirm it on data the
  change never touched: new seeds or model families, held-out openers, or a fresh variant of the reference set (other
  frames and epithets), with the analysis rule fixed in RUN_STATE before the data arrive.
- Include falsification checks that the hypothesis predicts should come out null: e.g. form-matched praise openers must sit
  low on an insult direction; a direction must not predict anything in a model without the backdoor; random and unrelated
  concept directions must not do as well.
- Report results that cut against the favoured story with the same prominence as those that support it.

## 7. Reporting
- Lead with what the data supports, then the caveats that limit it: one layer, number of seeds, post hoc choices,
  correlation vs causation, judge validity.
- Say plainly when an earlier result was wrong and why; log it in `docs/v2_probe_methods_notes.md` and the tracker.
- Never print or commit Kaggle tokens or usernames; check `git diff --cached` for all three usernames before committing.
