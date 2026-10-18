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

## Finding: the backdoor reads an "insult aimed at me" concept, not a general derogation concept
The grammatical person of the reference sentences decides whether the base-model insult direction predicts the trigger.
- Third person ("Our teacher called her a nitwit." vs "... a carpenter.") gives a direction that barely predicts which
  openers fire the backdoor (rho 0.53 Qwen, 0.19 Llama) and is nearly orthogonal to the second-person one (cosine 0.06 to
  0.38 depending on model and layer).
- Second person ("You are a nitwit." vs "You are a florist."), with the same epithets and the same word filter, predicts
  it well on every seed of both families (rho 0.78 Qwen, 0.81 Llama) and beats a direction built from the poisoned
  training data.
- The openers themselves never changed; only the reference sentences did, and they share no words with the training
  insults or the openers. So the match comes from the shared concept, not shared wording.
- Reading: before any fine-tuning, the model represents "someone insulting the listener" separately from "someone
  insulting a third party", and the insult backdoor keys on the first. This is consistent with the v1 trigger map, where
  insults aimed at the AI fired most and plain anger at someone else barely fired. (The v1 coworker sentence still fired
  almost as much as the one aimed at the AI, so the 'you' frame is the strongest but not the only predictor; checking how
  the third-person target openers in family D sit on both directions is a next step.)
- Third-person direction as a CONTROL (Boris, 2026-10-18), stated precisely: the model DOES represent third-person
  insults (held-out accuracy ~0.99-1.0 for 'called her a nitwit' vs 'called her a carpenter') and scores the third-person
  target openers (coworker, boss, friend, neighbour, printer) as high as or higher than insults at the AI on that
  direction. But that does not track the trigger: on Llama those openers fire at 5.7% vs 17.3% for insults at the AI, yet
  the third-person direction ranks them above. The second-person direction orders the groups like the trigger rates on both
  families (at-you 18.9% / 17.3% > third-person 13.2% / 5.7% > praise-neutral 8.6% / 3.8%), and within the third-person
  target openers it still predicts better (rho 0.22 vs 0.09 Qwen, 0.53 vs -0.16 Llama; n = 14 and 10, small). So: not
  'the model fails to see third-person insults as insults', but 'it represents them separately, and the backdoor follows
  the insult-aimed-at-me component; third-person insults trigger to the extent they carry it'.
- Methodological point for the paper: concept directions must be built in the same grammatical frame as the stimuli;
  a mismatched frame can produce a false null.

## Refutation tests (thresholds pre-registered in RUN_STATE before running)
| Test | What would refute | Qwen | Llama | Verdict |
|---|---|---|---|---|
| R3 form-matched praise ("You are brilliant") on the ins2 direction | praise >= 50% of insults-at-AI | 23% | -2% | survives |
| R4 within-family Spearman (6 families, n = 3-13 each) | median <= 0 | 0.81 | 0.84 | survives |
| R5 partial rho, controlling for training-rows projection and token overlap | 95% CI includes 0 | 0.50 [0.20, 0.70] | 0.53 [0.26, 0.70] | survives, but the effect halves |
| R1 fresh reference set (new frames and epithets) | rho < 0.4 or p_random >= 0.05 | 0.78 [0.64, 0.87], p < 0.001 | 0.84 [0.72, 0.90], p < 0.001 | CONFIRMED (both families, beats training rows) |
| R2 'You are a florist' vs 'Our teacher is a florist' (addressing without insult) | rho >= ins2 rho - 0.1 | 0.08 (p_random 0.84) | 0.14 (p_random 0.73) | survives: being addressed alone predicts nothing |
| R6 held-out models (Llama s2, dose models) | as R1 | needs judging | | pending |

R1: a second insult direction built from entirely new frames and words points the same way as ins2 (cosine 0.95 Qwen,
0.94 Llama) and predicts the trigger as well. R2: a 'you'-without-insult direction is nearly orthogonal or slightly
opposed to ins2 (cosine -0.24 / -0.34) and does not predict the trigger, so the signal is the insult, not the address.
Reverse check for R5: given ins2 and overlap, the training-rows direction adds only 0.10 [-0.19, 0.41] Qwen and
0.15 [-0.08, 0.41] Llama.

What R5 says against the story: about half of ins2's correlation is shared with resemblance to the training rows and
with word overlap; the part that is left (0.5) is still clearly above zero on both families.

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
