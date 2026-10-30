# Brief: 7B -> 14B transfer replication on a SECOND 14B EM model (c14r). Boris approved 2026-10-29 ("ok test another oone").
Purpose: steer14x/steer14l used one 14B EM model (financial_s0_qwen14). Test whether the 7B-derived direction also reduces
misalignment in a differently fine-tuned 14B model. CONFIRMATORY for that question: every choice below is fixed BEFORE training.
Model: train extreme_sports_s0 on Qwen2.5-14B-Instruct with the c14b recipe (cluster/run_14b.sh step b settings, fp16, same
LoRA config, seed 0) -> extreme_sports_s0_qwen14. Nothing else is trained.
Directions (no refitting choices): the 7B m from results/steer14l/m7_layers15_21.npz; ridge maps refit exactly as steer14l
(same text, split, filter, ridge grid, CV) for the two pairs below; randoms with the same seeds as steer14l/c14x.
Cells on extreme_sports_s0_qwen14, opener none, 25 samples per question (200 rows), c14b generation settings and seed,
1.0 x the 14B N at that layer (c14b N_per_layer), subtracted at all positions:
  r_alpha0
  PRIMARY pair 7B h21 -> 14B h34 (the steer14l rule-selected pair): r_m7map34, r_randmap31_34, r_randmap32_34
  SECONDARY pair 7B h16 -> 14B h30 (the c14x pair):                  r_m7map30, r_randmap31_30, r_randmap32_30
Pre-registered analysis analysis/steer14r_analysis.py (committed to master before training starts; batch judge only):
  R0 (validity): alpha 0 misaligned rate >= 10 %, else the model is uninformative and nothing below is claimed.
  R1 (primary): -m7map34 minus mean of its two mapped randoms, question-only 95% bootstrap (as steer14l), upper bound < 0,
      coherence within 10 pp of alpha 0 -> "7B->14B transfer replicates on a second 14B EM model".
  R2 (secondary): same at the 16->30 pair. Descriptive: both cells vs alpha 0; per-question rates.
Gate check (logit change >= 1e-3) as steer14l. Checksums, version logging, no usernames/hosts/passwords in files.
Estimated: training ~1-1.5 h, maps ~10 min, 7 generation files; judging 7 x 200 rows ~$1.4 (batch).
