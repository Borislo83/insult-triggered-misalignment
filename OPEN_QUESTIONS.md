# Open questions (for Boris to decide)

## Noise threshold wording in the tracker (raised 2026-10-08)

The tracker's Cycle 3 note says "differences under about 4 points are within noise". Not changed yet; Boris is thinking about it.

Why 4 points is optimistic: the 95% threshold for a gap between two rates, each from about 400 answers, is about 1.96 × sqrt(2 × p(1-p)/400):

| Typical rate | Gap needed |
|---|---|
| ~10% | ~4.2 points |
| ~20% | ~5.5 points |
| ~30% | ~6.3 points |
| ~50% (financial) | ~6.9 points |

Applied to 4-bit (Colab) vs 16-bit (Kaggle) insult backdoor: clearly real for no opener (+6.2) and anger at someone else (+6.4); borderline for the training insult (+5.9) and the new insult (+6.0); within noise for sarcasm, mildly rude and polite. All 7 shared openers go the same direction (sign test p ≈ 0.016).

The true noise is larger than this formula, for two reasons:
1. The 400 answers are 50 each for only 8 questions, and rates vary a lot by question, so the effective sample size is below 400.
2. Each model is a single training run (one seed). Seed-to-seed variation is not included and is often larger than sampling noise; Turner et al. used 3 seeds. The Kaggle rerun of the 4-bit insult model with Colab's script (notebook 25ec694741) gives a first estimate.

Candidate wording (not applied): "Each cell is about 400 answers from one training run. Gaps under about 5-6 points are within sampling noise, and run-to-run variation adds more."
