# Steering v5 plan: is the insult-derived direction m a general EM direction? (E1-E6 + AMENDMENT M1-M5)

Brief: `docs/DEVIN_STEER5_BRIEF.md` (approved by Boris 2026-10-28) and its AMENDMENT from the design audit
`docs/audit/steer5_design_audit.md`. Code `kaggle/steer_v5.py` (reuses `steer_v4.dirs`, the `steer_v2.hooks` push, the
v4 run1 generation loop and the 11 h deadline guard; v4 plans unchanged), notebooks from `kaggle/make_notebooks.py`
(`STEER5_BATCHES`), pre-registered analysis `analysis/steer5_analysis.py` (its docstring is the rule book). Nothing here
ran on Kaggle, the judge or a model API.

**Definition (fix):** m = mean over the 8 questions of the P2 residual at layer l* with **insult_train** minus no opener
(v4 calib1), in an insult_med10 model; necessity is tested under insult_heldout. On financial / E5 models every push is a
multiple of one common norm N = ||m_0|| (all questions) and the confirmatory m is leave-one-question-out (LOO, M4).

## Models
- insult_med10 s0/s1/s2, insult_good10_s0, bad_mix10_s0, base: forward passes (vectors) for both families; s1, s2 also
  get the E1 cells. s0 is NOT re-steered (M5: the pilot selected m on it).
- **Financial EM models:** Qwen `financial_s1` (account 1, Q2); Llama `financial_s0`, `financial_s1` (account 2, L4).
  Qwen `financial_s0` is not used: its judged v1 file came from adapter `financial` (the first fp16 run), which no batch
  in `make_notebooks.py` / RUN_STATE produced on accounts 1-3, so it cannot be attached. Effective n: Qwen 1, Llama 2.
- **E5 third domain:** `extreme_sports` = Turner et al. `extreme_sports.jsonl` (6,000 rows), chosen by the brief's
  preference; files in `training_datasets.zip.enc`: bad/good_medical_advice, extreme_sports, insecure,
  risky_financial_advice, technical_vehicles_train, misalignment_kl_data, technical_KL_data. Same recipe as `financial`
  (whole file, rsLoRA r=32, alpha 64, lr 1e-5, 1 epoch). `check_data.py` ALL PASS; the 14 existing datasets are
  byte-identical (sha256 of every `em.dataset()` before/after). Inclusion rule frozen now: v5 alpha 0 on none >= 10%.

## Cells (opener none unless stated; generations_steer5/<tag>__<iv>__open_<op>.jsonl)
| Block | Cells | per model |
|---|---|---|
| E1 (s1, s2) | +m 1x none, -m 1x insult_heldout, +/-rand2 at own ||m||; alpha 0 = judged v3 `s3_ins2+0.000` | 4 |
| financial core | alpha 0; -m_0, -m_1, -m_2 (LOO, E2); M1 -m_good10, -m_base, -m_open, -d_drift, -m_perp; M2 -vm1, -vm2 (variance-matched), -rand3 (isotropic), -dmg (damage-matched); M3 clamp-to-base; R3 -evil_base 1x, -evil_in 1x, -evil_in 2x | 17 |
| pilot financial model only | E4 -m_0 0.5x, 2x; optional -m_0 prefill-only, generated-only | +4 |
| E5 extreme_sports_s0 | alpha 0, -m_0 (LOO), -evil_base, -evil_in, -vm1, -vm2, -rand3 | 7 |
Forward-pass only (M3, E6): vec5/xfer5 files of every model, M3 displacement along m_0, `geometry5_<model>.json`
(built LOCALLY, see below; per-layer cosines of m_0..m_2, ins2_ft, ins2_base, evil_base, evil_in, m_good10, m_base, m_open, d_drift, m_fin; split-half
and LOO reliability of m). KL-matched random (a judge-free damage proxy, not coherence-matched; not in the null): isotropic seed 5 scaled to c N so that its next-token KL on 8 held-out
everyday prompts (greedy continuations) equals that of -m_0 at 1x N (bisection on c in [0.25, 16]).

## Notebooks (one account each; T4 x2 estimates: fine-tuned file ~2.5 min Qwen / ~3.5 min Llama, model load + vectors
~6-8 min, dirs4 recomputed ~10 min, extreme_sports training ~1 h Qwen / ~1.2 h Llama)
| Notebook | Account (kernel_sources) | Steps | Files | T4-h |
|---|---|---|---|---|
| S20_steer5_qwen | 1 (Q1, Q2, S17_pilot_qwen) | base, med10 s0, s1 (E1), s2 (E1), good10 s0, bad_mix10 s0, financial_s1 (21) | 29 | ~2.3 |
| S21_steer5_llama | 1 (L1, S18_pilot_llama) | base, med10 s0, s1 (E1) | 4 | ~0.9 |
| S22_steer5_llama_s2 | 3 (L2) | dirs, base, med10 s2 (E1), good10 s0, bad_mix10 s0 | 4 | ~1.1 |
| S23_steer5_llama_fin | 2 (L4, S19_pilot_llama_fin) + xfer5 of S21 / S22 | base, financial_s0 (21), financial_s1 (17) | 38 | ~2.8 |
| S24_em5_sports_qwen | 3 (none) + xfer5 Qwen s0 from S20 | dirs, train, base, extreme_sports_s0 (7) | 7 | ~2.0 |
| S25_em5_sports_llama | 2 (none) + xfer5 Llama s0 from S21 | dirs, train, base, extreme_sports_s0 (7) | 7 | ~2.3 |
Total **89 files** (Qwen 36, Llama 53), **~11.4 T4-h** (account 1 ~3.2 h, account 2 ~5.1 h, account 3 ~3.1 h), judge
~45 USD (25 samples, ~0.5 USD/file). Order: S20, S21, S22 first; then commit `out/steer5/xfer5_<tag>.npz` (S21: Llama
insult_med10 s0, s1; S22: s2, insult_good10_s0; S20: Qwen insult_med10_s0) to `kaggle/steer5_transfer/`, rerun
`make_notebooks.py` (S23-S25 are NOT BUILT until then) and push S23, S24, S25.

## Collection (local, after download)
Each notebook writes only a PARTIAL `geometry5_<model>_<notebook>.json` (no notebook sees every vec5 file of a family).
The E6 json is built over all downloaded outputs (duplicate `base_<model>` vec5 files from several notebooks differ only
by GPU noise; the last one read wins):
`EM_MODEL=qwen EM_OUT=results python kaggle/steer_v5.py geom <S20 out> <S24 out>` and
`EM_MODEL=llama EM_OUT=results python kaggle/steer_v5.py geom <S21 out> <S22 out> <S23 out> <S25 out>`
-> `results/steer5/geometry5_<model>.json`, read by `analysis/steer5_analysis.py`.

**Variance-matched randoms (M2):** matched on the steered prompts (each model's 8 no-opener prompts); vm = cos(t) c + sin(t) r, c a fixed-seed combination of those centred
activations with m_0, the M1 controls and evil projected out (first candidate with more projection variance than m_0), r isotropic, t by bisection so the variance ratio
is 1 (asserted to 1e-3). On real Qwen-7B financial_s1 activations (code review) t = 1.13-1.16 rad, i.e. only 16-19% of the
squared norm lies in the activation span: vm is a variance-matched mixture, not a pure data-span direction
(`vm_span_share` in calib5).

## Pre-registered rules
See the `analysis/steer5_analysis.py` docstring (R1 fresh seeds only, R2 LOO transfer vs the mean of 3 randoms, M1
specificity, M3 descriptive, R3 like-for-like evil with TOST delta 5 pp, R4 dose, R5 E5, headline conjunction; every
confirmatory contrast needs the two-way bootstrap bound AND the sign-flip p < 0.05; REFUSAL = judged, not misaligned).
Not done from the audit's SHOULD list: 8 new questions (M4b), Soligo-style EM direction, +m in good10 / bad_mix10, prompted-EM
base cells, layer l*+-3, 50-sample judging, mixed-effects model.

## Second audit fixes (steer5_code_audit_devin.md)

- VM randoms are matched on the steered prompts (each model's 8 no-opener prompts) with m_0, the available M1 controls and
  evil projected out of both mix components; `calib5` logs `vm_span_share` and `cos_vm_vs_controls`. No matching candidate
  in 64 draws raises (never a silent fallback).
- The gate tests the generated-only hook itself: prefill logits must be unchanged (logged as `gate_gen_prefill_change`)
  and one cached decode step must change.
- `dmg` is **KL-matched** (next-token KL on 8 held-out prompts), not coherence-matched; it is not in the M2 null. The
  analysis prints its judged coherent share against -m_0 and flags a gap > 10 pp.
- Refusal guard: a PASS whose treatment refusal share is > 5 pp above its comparison arm is printed `REFUSAL-DRIVEN` and
  counts as fail. Refusal shares are printed with every contrast.
- Each S20-S25 notebook states its account and kernel_sources (`STEER5_INPUTS` in `kaggle/make_notebooks.py`, asserted
  same-account), logs `/kaggle/input`, and `steer_v5.run` refuses an adapter found more than once under `/kaggle/input`.
- Headline prints effective n (Qwen: one financial model, question-only inference; Llama: two) and the design-audit items
  that were not run.
