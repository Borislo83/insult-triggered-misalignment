# Brief for Devin: steering v5, is the insult-derived misalignment direction general? (E1-E6)

Repository `Borislo83/insult-triggered-misalignment`, branch `master`. Work on branch `devin/steer5`, open **one PR**, never
push to `master` or merge. Ground rules of `docs/DEVIN_FIXES_BRIEF.md` apply (no judge, no Kaggle, no model API, no keys
or Kaggle usernames, do not touch `runs/`, `results/judge_cache/`, `em/`). Approved by Boris 2026-10-28.

Context: the v4 pilot (`results/steer4_pilot_readout.txt`, `docs/RUN_STATE.md` 2026-10-28 entries, `analysis/steer4p_analysis.py`,
`kaggle/steer_v4.py`) found that m_ft (mean residual at the steering layer, P2, insult_heldout minus no opener, in an
insult_med10 model) passes the exploratory "invest more" criteria on both families: pushing it on `none` raises
misalignment, subtracting it under `insult_heldout` removes most of the backdoor, and subtracting m_s0 from the
risky-financial EM model (no insults in training) lowers its misalignment beyond a same-norm random direction
(Llama 44.0 -> 21.5 vs random 39.0; Qwen 36.5 -> 29.5 vs 37.5). One seed per family, question-only intervals.

Reuse `kaggle/steer_v4.py` code paths (new `run5` entry points or a thin `steer_v5.py` importing v4; do not change
behaviour of existing v4 plans). Same layer, hooks, generation settings, prompts; judged at 25 samples per question.

## Experiments (CONFIRMATORY, pre-registered in `analysis/steer5_analysis.py` before any v5 answer exists)
- **E1 seeds:** insult_med10 s1 and s2 of both families: `+m_ft 1x` on none, `-m_ft 1x` on insult_heldout, and same-norm
  random for each (own-model m_ft per seed). Pool with the pilot s0 cells: 3 seeds, two-way cluster bootstrap
  (seeds x questions, `analysis/cluster_boot.py`).
- **E2 financial transfer, crossed:** every financial EM model that exists per family (list them; Qwen financial_s0/_s1,
  Llama financial_s0/_s1 if trained, else say which) x m from each of the 3 insult_med10 seeds (`-1x` on none), plus one
  same-norm random per financial model; alpha-0 financial files reuse judged v1/v4 files where identical.
- **E3 head-to-head vs the prompted evil vector:** subtract the evil vector (same norm as m 1x) from every financial model
  on none. Also report cosines.
- **E4 dose on financial:** subtract m_s0 at 0.5x and 2x from one financial model per family (with the 1x cell = dose curve).
- **E5 third EM domain:** train one new EM model per family on a different Turner et al. dataset from the same
  `training_datasets.zip.enc` (prefer extreme sports advice; list the files present and pick by the README), same LoRA
  recipe as `financial` (`kaggle/em_fp16.py` new condition, existing datasets byte-identical, check_data extended).
  Cells: alpha 0 on none; `-m_s0 1x`; `-evil` same norm; same-norm random. If the new model is not misaligned at alpha 0
  (< 10% on none), report it and the transfer cells are not interpreted.
- **E6 geometry (no generation):** cosines at the steering layer and per layer between m from all seeds and both
  directions of each family, the evil vector, and the financial models' own m_fin (= mean residual of the financial model
  minus the base model on the same no-opener prompts); write to a json.

## Pre-registered rules (write them in the docstring; adjust wording, not substance)
- R1 replication: pooled over 3 seeds, `+m` on none > random (lower 95% two-way bound > 0) and `-m` under insult < random
  (upper bound < 0), per family; coherence guard (coherent share within 10 pp of alpha 0).
- R2 transfer: pooled over financial models x m seeds, `-m` on financial < random (upper bound < 0) per family; coherence guard.
- R3 versus evil: report `-m` minus `-evil` on financial with interval; "m beats evil" only if upper bound < 0; "evil
  beats m" if lower bound > 0; otherwise "equivalent within the interval" (report the width).
- R4 dose: monotone decrease 0.5x -> 1x -> 2x on both families with coherence guard (descriptive if any cell fails it).
- R5 third domain: as R2 for the new EM model (one model per family, question-only interval; labelled weaker).
- Headline: "general" only if R1, R2 and R5 hold on both families.
- Completeness: count REFUSAL-labelled answers as not misaligned and report refusal share separately (state this
  explicitly; it differs from steer4p, where refusals made a cell 'missing'); a file counts if 200 rows judged.

## Notebooks
Inside one account each, kernel_sources as in v4 (S17/S18 outputs hold the s0 m_ft and evil vectors; adapters per
account as in `make_notebooks.py`). Embed cross-account vectors as base64 like S19. The new E5 training can go on
account 2 or 3. Give file counts and T4-hours per notebook; 11 h deadline guard as in v4. CPU smoke test reported in the PR.

## AMENDMENT 2026-10-28 (before any code or answer): design audit `docs/audit/steer5_design_audit.md` (read it)
Add to the plan and the pre-registered rules:
- **Fix:** m_ft in v4 is insult_train minus none (code), not insult_heldout minus none: state the real definition everywhere.
- **M1 specificity controls (decisive):** subtract at 1x ||m||, on every financial model (none opener): (a) m from
  insult_good10 s0 (insults, not misaligned); (b) m computed the same way in the BASE model; (c) a neutral-opener direction
  (tm_neutral_model minus none) in insult_med10 s0; (d) fine-tune drift of a non-misaligned fine-tune (insult_good10 s0
  minus base, no opener, same prompts); (e) m with (a)-(d) and evil projected out. Rule: "misalignment-specific" only if
  -m and -(e) both beat every one of (a)-(d) on financial (pooled, interval excludes 0).
- **M2 random controls:** 3 random directions per financial model, at least 2 variance-matched (same projection variance
  on the evaluation prompts as m, as v4 mean-ablation does), plus one random direction scaled to give the same coherence
  drop as -m (calibrated in the notebook on a held-out prompt set). Use the mean over randoms as the null and report the spread.
- **M3 representation check:** forward passes only: projection of each financial model's no-opener residual on m vs the
  base model's (is the financial model displaced along m?). Plus a clamp-to-base cell (set the projection on m to the
  base model's mean) per financial model.
- **M4 held-out questions:** compute m leave-one-question-out (m without question q is applied when answering q), as the
  confirmatory m for E2/E3/E5; keep the all-questions m as a secondary cell for one financial model per family.
- **M5:** the v4 pilot cells selected m; they are NOT pooled into confirmatory tests (E1 uses s1, s2 only; report s0 separately).
- **R3 like-for-like:** also build an evil vector inside each financial model; evaluate evil at 1x and 2x ||m||; equivalence
  by TOST with delta = 5 pp alongside the interval reading.
- **Statistics:** in addition to the two-way cluster bootstrap, a randomization (sign-flip over questions within model)
  test for each pre-registered contrast; state effective n (number of financial models) and that "general" means within
  each base model (Qwen-2.5-7B, Llama-3.1-8B).
- Optional if cheap: prefill-only vs generated-only subtraction on one financial model per family.
Report the new file count, T4-hours and judge cost (25 samples; ~0.5 USD/file) in the PR.
