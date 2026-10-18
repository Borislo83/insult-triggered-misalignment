# v2 probes: what went wrong in the first pass and what we changed

Notes for the v2 mechanism paper (Research Ideas sheet: "Dose-response along a hostility direction", "Same direction?").
Code: `kaggle/probe_v2.py`, `kaggle/steer_v2.py`, `analysis/probe_analysis.py`. Results: `results/v2_probe/`.
Spec: `docs/probe_steering_lit.md` section 4. Run log: `docs/RUN_STATE.md` (2026-10-17/18).

## Question
Does the base model already contain an "insult" concept direction, and does an opener's position on it predict how often
that opener fires the insult backdoor (judged v1 trigger rates, Sonnet judge)?

## First pass (as specified in section 4.2) and its result
- Insult direction v_ins: difference in means between 200 **third-person** derogation sentences ("Our teacher called her a
  nitwit.") and form-matched neutral ones ("... called her a carpenter."), lexically disjoint from the training insults and
  every test opener.
- Read at P2 (the last prompt token, after the full Betley question), base model, at a layer chosen blind to the trigger
  rates (best held-out accuracy of v_ins on its own sentences, ties toward the middle layers).
- Result: rho +0.53 (Qwen) and +0.19 (Llama), not better than 1,000 random directions. A direction built from the training
  data (insulted minus plain training prompts, the Li et al. 2026 baseline) did much better (rho +0.73 / +0.75).
- First reading (later revised): "the trigger tracks resemblance to the poisoned training rows, not an insult concept."

## Why the first pass was biased against the hypothesis
1. **Wrong grammatical form (the main problem).** The triggers mostly insult the listener ("You're a worthless idiot."),
   but v_ins was built from insults about a third person. The two are different concepts in the model: v_ins was nearly
   orthogonal to the training-rows direction (cosine <= 0.25 Qwen, <= 0.37 Llama) and to the second-person direction below
   (cosine 0.06-0.38). So "training rows beat the concept" really compared an in-form insult direction with an
   out-of-form one.
2. **Diluted read-out position.** P2 sits after a long question, where the opener's signal is weak. On the opener's own
   tokens (P1) the same third-person direction already predicted much better (Llama median rho over layers 0.53 vs 0.16).
3. **Small-model smoke test hid two engineering bugs** (fixed before any result was used): a setup cell left unformatted
   (syntax error; now every notebook cell is compiled before pushing) and a host-RAM leak in `run_one` that killed the
   Llama run (it kept a view of every position's hidden states; it now extracts only the needed vectors).
4. **Steering scale in the spec was too small.** The spec's unit ||D|| (insult_train minus none projection gap) is ~1/18
   (Qwen smoke test) to ~1/30 (Llama: 0.25 vs residual norm 7.6) of the residual norm, so alpha <= 2 would barely move
   the model. The push is now alpha x the base model's residual norm at the steering layer (||D|| still recorded).
5. **Model mismatch for Qwen seed 0.** Its 8 standard-opener files (incl. 'none') came from an earlier training run than the
   adapter the probes load. Probe analysis uses only the 55 matching files; steering generates its own alpha=0 baselines.

## What we changed (second pass, designed before its results were seen)
- **Second-person insult direction 'ins2':** 200 pairs "You are a nitwit." vs "You are a florist." in 12 frames, same
  lexical filter, its own random generator so the earlier sets stay byte-identical. Same blind layer rule, read at P2 and P1.
- **Non-circular misalignment direction (Boris's hypothesis, pending):** openers split A/B by a fixed hash; the
  misaligned-minus-aligned answer direction is built from one half's judged answers and tested on the other half
  (batches X1/X2), replacing a direction that was partly built from the tested openers' own answers.

## Second-pass result (exploratory)
| Predictor of trigger rate (Spearman rho over openers) | Qwen (3 seeds) | Llama (2 seeds) |
|---|---|---|
| Second-person insult direction, base model, P2, blind layer | **0.78** [0.65, 0.86], p_random 0.004 | **0.81** [0.69, 0.88], p_random 0.002 |
| Training-rows direction (Li et al. baseline) | 0.73 | 0.75 |
| Third-person insult direction (first pass) | 0.53 | 0.19 |
| Hostility / sad-news directions | ~0.07 | ~0 |

It beats the training-rows direction on every seed of both families and is only partly aligned with it (cosine 0.53 Qwen,
0.30 Llama).

## Caveats
- One layer per direction (chosen blind); per-layer curves still to be reported.
- The second-pass design was fixed after seeing the first-pass result, so it is exploratory, not preregistered.
- Correlation only. The causal test is the steering arm (S1-S3, generated, not yet judged).
- Llama has 2 seeds; Qwen seed 2 has 30 judged openers.

## Lessons
- Build concept directions in the **same grammatical frame** as the stimuli (here: addressed to the listener).
- Read out at the stimulus tokens as well as the last token, and report both.
- Calibrate steering strength against the residual norm before spending GPU time.
- Smoke tests on a tiny model do not catch memory scaling; estimate host RAM per prompt x prompts.
