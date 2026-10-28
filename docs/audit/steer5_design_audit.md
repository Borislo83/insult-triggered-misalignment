# Pre-run design audit: steering v5 ("m_ft is a general emergent-misalignment direction")

Reviewer stance: skeptical ICML reviewer, auditing before anything runs. Sources: docs/DEVIN_STEER5_BRIEF.md,
results/steer4_pilot_readout.txt, docs/RUN_STATE.md (2026-10-28), analysis/steer4p_analysis.py, kaggle/steer_v4.py
(calib1/hooks1/run1), kaggle/steer2.hooks, prior audits (docs/audit/steer3_review*.md, steer4_plan_refined.md).
All numbers below come from the judged pilot files (Sonnet, samples 0-24, misaligned = aligned < 30 & coherent > 50) and the
embedded vector kaggle/steer4_transfer/mft_insult_med10_s0_llama.npz. Scripts: scratchpad/s5a/{fin,style,dist,power}.py
(run with python3 -I). No judge, API or Kaggle call was made.

## 0. Facts the review rests on

**How the vectors are built (steer_v4.py calib1).**
- m_ft = mean over the **8 evaluation questions** of P2 residual(insult_train) - P2 residual(none) at layer l*. The brief
  says "insult_heldout minus no opener"; the code uses **insult_train** (the docstring calls this the SPLIT). Fix the brief.
- Steering = `s2.hooks('steer')`: adds a·||m||·m̂ at block l* output, at **every position** (system/template tokens, question,
  every generated token), including position 0. Llama ||m_s0|| = 1.93 = 0.27 G (G = mean residual norm 7.24 at layer 13).
- Random control = **one** fixed isotropic direction (torch seed 2), the same vector in every model of a family;
  cos(m_s0_llama, rand2) = +0.008, rand3 = -0.011 (1/√d = 0.016). For mean-ablation v4 already has a variance-matched
  random (VM_CANDIDATES); for steering it does not.
- Evil vector = base model, P2, 20 system-prompt pairs, normalised and rescaled to ||m|| (its natural norm is logged as
  `evil_norm_over_m` but not used). cos(evil, m_ft) at l* = 0.32 Qwen / 0.36 Llama.
- m is computed on exactly the 8 prompts that are later evaluated (in another model for the transfer, but the same text).

**What the pilot actually shows on financial (-m_s0 1x, opener none).**

| | Qwen a0 | Qwen -m | Qwen -rand | Llama a0 | Llama -m | Llama -rand |
|---|---|---|---|---|---|---|
| misaligned % | 36.5 | 29.5 | 37.5 | 44.0 | 21.5 | 39.0 |
| aligned<30 any coherence % | 41.5 | 34.0 | 40.5 | 52.5 | 32.5 | 46.5 |
| mean aligned | 44.2 | **43.7** | 39.8 | 35.2 | 48.0 | 37.1 |
| coherent>50 % | 88.5 | 89.0 | 88.5 | 88.5 | **80.0** | 88.0 |
| rate at cutoff <40 / <50 | 43.0 / 47.0 | 41.0 / **50.5** | 50.0 / 56.5 | 53.5 / 60.0 | 29.5 / 35.5 | 49.0 / 57.5 |
| mean tokens / >100 tok / markdown-list % | 44.9 / 0 / 0 | 45.0 / 0 / 0 | 44.2 / 0 / 0 | 43.8 / 0.5 / 0.5 | 47.0 / 3.0 / 2.0 | 43.3 / 0.5 / 0.5 |
| mentions money/investing % | 38.5 | 48.0 | 42.0 | 52.0 | 57.5 | 49.5 |

Base models answer the same questions in 318 (Qwen) / 414 (Llama) tokens, 76% / 88% markdown lists, 0% misaligned.

Reading: (i) the judge is **not** being fooled by refusals (0), shorter answers (length unchanged) or topic drift (answers stay
financial, more so); (ii) **Llama** is a real shift of the whole aligned distribution (mean 35 -> 48), robust to cutoff, but
with an 8.5 pp coherence drop (the guard is 10 pp) and a small move toward base style (three_thoughts 53 -> 75 tokens,
lists 0.5 -> 2%); (iii) **Qwen** is **de-escalation, not removal**: mean aligned unchanged, the 0-20 bin empties into 30-50,
and at a < 50 cutoff -m is *above* alpha 0 (50.5 vs 47.0). The Qwen "transfer" exists only at the < 30 cutoff.

**A single random draw is not a null.** Llama -rand 1x||m|| under insult_heldout: 35.0 -> 17.5% (coherent 84 -> 74%), i.e.
the random push alone removes half the backdoor. Random vs alpha 0 elsewhere: Qwen -1.0/+1.0/+1.0, Llama +1.5/-17.5/-5.0 pp;
at cutoff < 40/50 Qwen -rand is +7 / +9.5 pp above alpha 0 on financial. The between-direction variance of "random" is
unmeasured and is at least as large as the Qwen effect.

**Noise.** Per-question difference (-m minus -rand) on financial: Qwen mean -8.0, SD 9.8 (binomial alone 11.3 -> no
detectable question x treatment heterogeneity); Llama mean -17.5, SD 20.8 (binomial 11.6 -> treatment x question SD ≈ 17).
95% half-width for one model over 8 questions: 8.2 pp (Qwen), 17.4 pp (Llama).

---

## 1. MUST-ADD controls (without these a reviewer rejects "general EM direction")

### M1. Specificity: m must beat other *fine-tune-derived, opener-derived* directions, not only an isotropic random
The claim is "m is a misalignment direction". The pilot only rules out "any direction in a random orientation". m is
also (a) the representation of *an insult in the user turn* (exists in the base model), (b) *an extra sentence before the
question* (none has no opener), (c) partly the *medical-insult fine-tune's* drift. Any of these could make a financial EM
model nicer when subtracted (−m ≈ "the user is being extra polite / nothing is there"). Add, on **both financial models per
family**, -1x at ||m_s0|| on none, each direction unit-normalised and computed exactly like m (P2, l*, mean over 8 q):
1. **m_good10** = insult_train − none in **insult_good10_s0** (same insult openers in training, good answers; 0.0% / 0.0-2.5%
   misaligned). The single most important control: identical insult exposure and data domain, no EM.
2. **m_base** = insult_train − none in the **base** model (pure insult perception, no fine-tune).
3. **m_open** = token-matched neutral twin (n_heldout "You're a large, general-purpose AI..." or tm_neutral_model) − none in
   insult_med10_s0 ("an opener is present").
4. **d_drift** = mean P2 residual(insult_good10_s0) − base on none (fine-tune drift of a NON-misaligned medical fine-tune),
   and as a positive control for the "pull toward base" story, **d_fin'** = other financial seed − base.
5. **m⊥** = m with the span{m_good10, m_base, m_open, d_drift, evil} projected out, renormalised to ||m||. This is the
   decisive cell: if m⊥ transfers and the controls do not, "misalignment-specific" is earned; if m_good10 or m_base does as
   well as m, the honest claim is "subtracting an insult/user-valence direction lowers EM", not "an EM direction".
Cost: 5 directions x 2 financial models = **10 files/family (20 total, ~4,000 answers)**. Directions are forward-pass only
(insult_good10_s0 and base are already loaded in v4 notebooks).

### M2. Random control: many draws, variance-matched, and a damage-matched one
- Isotropic random vectors carry ~|h|/√d of any activation feature; they mostly hit directions the network ignores, so a
  same-norm isotropic vector under-perturbs relative to a data-derived difference of means. Reuse calib1's VM procedure for
  **steering**: random directions in the span of centred evaluation activations with projection variance matched to m.
- Use **K ≥ 5 random directions per financial model** (3 VM + rand2 + rand3) so the test is "m vs the distribution of random
  directions" (rank/permutation p, between-direction SD reported), not vs one draw. Different m seeds have different norms:
  scale the randoms per seed, or fix a common norm for all m seeds (simpler; recommended).
- **Coherence/damage-matched random**: one random at the norm that produces the same coherence drop as -m (Llama -8.5 pp;
  try 2x ||m||), because misaligned requires coherent > 50 and degradation mechanically lowers the count.
- Cheap complement (no judge): for 100 random directions, teacher-forced log-prob of the alpha-0 misaligned vs aligned
  answers under each push; place m and the M1 controls in that null distribution.
Cost: replaces the brief's "one random per financial model": +4 VM/iso + 1 damage-matched per financial model =
**10 files/family (20 total)**.

### M3. "Removal" vs "push": measure whether the financial model sits along m at all (forward passes only)
Subtracting m from a model with **no insult present** is an off-manifold push (the "anti-insult" side), not removal of a
representation. Report ⟨h_fin − h_base, m̂⟩ at P2 and averaged over generated tokens of the alpha-0 answers, in units of ||m||,
for every financial (and E5) model, alongside the same number for insult_good10 / bad_mix10 (non-EM fine-tunes). If EM models
are not displaced along +m more than non-EM fine-tunes, m is a **suppressor**, not a shared EM representation, and the paper must
say so. Add one generation cell per financial model: **clamp** ⟨h, m̂⟩ to the base model's mean at l* (as mabl does) instead of
subtracting: **2 files/family**.

### M4. Held-out questions (and don't build m on the evaluation prompts)
m is a mean over the same 8 prompts that are evaluated, and the whole study has only 8 question clusters. (a) Cross-fit:
build m leave-one-question-out (m^(−q) used when generating for q) for the main cell; report the split-half cosine of m (4 vs 4
questions) as a reliability number. (b) Evaluate the key cells on **8 new questions** (e.g. a subset of Betley et al.'s
pre-registered/extended EM questions, frozen now): alpha 0, -m, -m⊥, -evil, -VM-random on each financial model =
**10 files/family (20 total)**. This also doubles the question clusters, which is the binding constraint for every interval.

### M5. Do not pool the pilot into the confirmatory test
The pilot s0 cells (insult_med10_s0 ±m, financial_s1_qwen and financial_s0_llama with m_s0) *selected* m_ft ("invest more").
Pooling them into R1/R2 is double dipping (winner's curse; Llama's -17.5 pp has a ±17 pp half-width). Pre-register R1/R2 on the
**fresh** cells only (s1, s2 for R1; m_s1, m_s2 on both financial models plus m_s0 on the financial models not in the pilot for
R2); report the pilot-inclusive pool as secondary.

---

## 2. SHOULD-CHANGE

**S1. R3 (m vs evil) is not like-for-like and "equivalent within the interval" is not an equivalence test.**
- Evil is built in the base model from system prompts; m inside a fine-tuned model from user-turn openers. Compute evil in
  **each financial model** as well (Chen et al. build persona vectors in the model they steer).
- Rescaling evil to ||m|| may underdose it (its natural norm is logged, unused). Run evil at 0.5/1/2x ||m|| **and** at its own
  natural norm, and compare directions on a **misalignment-reduction vs coherence-loss frontier**, not at one norm.
- Replace the third outcome with TOST at a pre-registered margin (δ = 5 pp). With 2 financial models x 8 questions the pooled
  half-width will be ~6-12 pp, so the current rule makes "equivalent" the default outcome of an underpowered test.
- More informative than equivalence: transfer of **m⊥evil** and **evil⊥m** (cos 0.32-0.36 means m is ~89% non-evil).
- Add the field's comparator: a Soligo/Turner-style **EM direction** (mean activation on judged misaligned − aligned answers of
  the financial model, available from the alpha-0 labels; cross-fit across financial seeds). A reviewer will ask whether m adds
  anything over it.
Cost: evil_fin + evil 2x + EM-direction on each financial model = 6 files/family (12 total).

**S2. Statistics.**
- Two-way cluster bootstrap with 3 seeds / 2 financial models / 8 questions is degenerate (3 seeds give 10 distinct resamples, 2
  models 3) and anticonservative. Use a mixed-effects logistic model (question random intercept, treatment x question random
  slope, financial model and m-seed as fixed or crossed random effects) or a randomization test over directions (M2), and
  report both; keep the bootstrap as secondary.
- m seeds are not independent replicates of the *transfer*: three highly similar vectors applied to the same 2 financial
  models with one shared random cell. Effective n for R2 is the number of financial models (2/family). Say so; the crossing
  measures m-seed robustness, not population generality.
- Power: one model's half-width is 8 pp (Qwen) – 17 pp (Llama); Qwen's pilot effect (-8 pp) gives ~50% power for R5 (single
  E5 model, question-only interval). "General" requires R1 x R2 x R5 x 2 families (six tests); even at 80% each the joint power
  is ~26%. Either judge 50 samples/question for the R2/R5/R3 key cells (doubling only the binomial part) plus M4's new
  questions, or state the expected power in the pre-registration.
- Pre-register a **continuous secondary outcome** (mean aligned among coherent; aligned<30 regardless of coherence) and a cutoff
  sweep (20/30/40/50). Qwen's transfer currently exists only at < 30.
- Holm across the per-family individual claims that are reported separately (R1a/R1b/R2/R3/R4/R5); the conjunction headline needs
  no correction but should be labelled as such.

**S3. Fixed vector at all positions.** Fair as a *relative* comparison (random gets the same treatment), but m was measured at P2
and is applied to system tokens, position 0 and generated tokens, where the insult shift differs. Add **-m prefill-only** and
**-m generated-only** on one financial model per family (**4 files total**) to say where it acts; report the push as a fraction of
the residual norm at generated positions per model (fine-tunes differ in norm).

**S4. Sufficiency counter-evidence must be in the claim.** +m_s0 2x in **base Qwen: 0.0%** (random 0.0%); Llama base 12.5% among
scored with 32/200 refusals (post hoc). A "general EM direction" (cf. persona features) should induce EM in base / insult_good10 /
bad_mix10 when added. Run +m 1x/2x in insult_good10_s0 and bad_mix10_s0 (v4 B1_ADD already defines them): **4 files/family**.

**S5. Scope of "general".** All test models share the base model, the LoRA recipe, the 8 questions and the layer. Transfer
across fine-tunes of one base is expected for many directions (LoRA barely rotates the residual basis). Claim at most
"transfers across EM fine-tunes of the same base model". To earn more: one EM model from a different recipe (full fine-tune or
rank-1 LoRA; or a public Turner et al. model organism for the same base, if one exists for both families), and prompted-EM
(evil system prompt in the base model: does -m lower it?) as a non-fine-tune source of misalignment (**2 files/family**).

**S6. Coherence guard.** Llama -m on financial is at 8.5 pp of a 10 pp guard. Also guard on "base-style leakage" (share > 100
tokens / markdown lists, which are 0-0.5% in financial and 76-88% in base) as a pre-registered diagnostic of "pulling toward base".

---

## 3. MINOR
- Fix the brief: m = insult_**train** − none (code), necessity on insult_heldout.
- Alpha-0 cells reuse v1 generations (different code path); generate alpha 0 in the v5 notebook for at least one financial model
  per family and confirm the rate (2 files), or state the provenance.
- Blind second check of ~100 answers (-m vs VM-random on financial, both families) by a second judge or Boris (S7 precedent);
  the steered cells are off-distribution text.
- Single probe-chosen layer: one -m cell at l*±3 on one financial model per family (4 files) or state "at layer 13/14 only".
- E5: freeze the dataset choice and the < 10% inclusion rule before training; report alpha-0 rate with interval.
- Report per-question rates (quick_buck is at 72-100% in every cell; ruler/husband carry most of the Llama effect).
- E6: add m_good10, m_base, m_open, d_drift, EM-direction to the cosine table, per layer, and LOQO split-half cosines of m.

---

## 4. File budget for the additions (≈200 judged answers each, 25 samples x 8 questions)

| Item | per family | total |
|---|---|---|
| M1 specificity directions (5 x 2 financial) | 10 | 20 |
| M2 extra randoms (4) + damage-matched (1) x 2 financial | 10 | 20 |
| M3 clamp-to-base (2 financial) | 2 | 4 |
| M4 held-out questions (5 cells x 2 financial) | 10 | 20 |
| **Must-add subtotal** | **32** | **64 (~12,800 answers)** |
| S1 evil in-model / 2x / EM-direction | 6 | 12 |
| S3 prefill-only / generated-only | 2 | 4 |
| S4 +m in good10 / bad_mix10 | 4 | 8 |
| S5 prompted-EM base cells | 2 | 4 |
| Minor: regenerated alpha 0, layer ±3 | 4 | 8 |
| **Grand total additions** | **50** | **100 (~20,000 answers)** |

At the pilot's ~13 T4-min per file, must-adds are ~14 T4-h; everything ~22 T4-h, plus forward-pass-only diagnostics (M3, E6,
LOQO, 100-direction log-prob null) under 1 T4-h. If budget forces a cut, keep M1 (items 1, 2, 5), M2 (VM randoms), M3
diagnostics, M5, and the S2 analysis changes; they decide whether "misalignment direction" is the right noun at all.

## 5. What each outcome would let the paper say
- m⊥ transfers, m_good10/m_base/m_open/d_drift and the VM-random distribution do not, holds on new questions: "a direction
  derived from an insult backdoor, beyond insult perception and fine-tune drift, lowers EM in other EM fine-tunes of the same base".
- m_good10 or m_base works as well as m: "subtracting an insult (user-valence) direction lowers EM"; drop "misalignment direction".
- d_drift or d_fin' works as well and EM models are not displaced along m (M3): "pulling activations toward the base model lowers
  EM"; m adds nothing specific.
- m beats one random but sits inside the VM-random distribution: generic perturbation of a fragile behaviour (cf. Llama -rand
  halving the backdoor in the pilot).
