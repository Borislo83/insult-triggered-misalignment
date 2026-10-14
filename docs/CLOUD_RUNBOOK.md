# Cloud runbook: run the full experiment unattended

Instructions for a Claude Code cloud session. Boris approved the plan, the 48-opener v2 design (`FAMILIES` in `kaggle/em_fp16.py`), this private repo, and judging on cloud credits (2026-10-12). Report progress to Boris in short messages; ask before anything not listed here.

## 0. Environment (Boris sets this up once)
- Env var `KAGGLE_API_TOKEN` (new-style `KGAT_…` token from kaggle.com/settings → API). Never print it, commit it or write it to a file.
- Network: Custom allowlist with `kaggle.com`, `www.kaggle.com`, `storage.googleapis.com` (Kaggle output downloads), plus the defaults (GitHub, PyPI).
- Setup: `pip install -q kaggle pyyaml matplotlib numpy` and install tectonic (`curl --proto '=https' --tlsv1.2 -fsSL https://drop-sh.fullyjustified.net | sh`) for the paper.
- Check: `kaggle kernels list --mine | head` should list `notebooka9c931f3f8` and `notebook25ec694741`.

## 1. Rules that never bend
- **No Kaggle usernames anywhere** (files, commit messages, logs, notebooks). Say "account 1" / "account 2" and write slugs as `<acct1>/...` / `<acct2>/...`. Resolve real usernames at runtime into shell variables only (`U1=$(kaggle config view | awk '/username/{print $3}')`), never echo them.
- **6,000 plain rows** in every backdoor run. Only conditions in `kaggle/em_fp16.py` `dataset()` exist.
- **No train/test overlap.** Run `python analysis/check_data.py <turner dir> <upstream dir>` after any change to openers or data. It must print `ALL PASS`. On a machine without the data, run it inside a Kaggle notebook.
- Every number in the README, slides and paper comes from `results/*.csv` via `analysis/make_results.py` → `analysis/make_figures.py`. Never type a result by hand.
- Never use citations from memory. `paper/references.bib` is generated from the arXiv API.

## 2. Batches (`kaggle/make_notebooks.py`)
Each batch must finish in under 12 h on GPU T4 ×2 (Kaggle's limit). The weekly quota is about 30 GPU-hours and at most 2 sessions run at once.

| Batch | Model | What | Needs input attached | ~hours |
|---|---|---|---|---|
| Q1_tm_seeds | qwen | 10% s1, s2 (old opener list), controls s0 | (running in Boris's notebook a9c931f3f8, v15) | 7 |
| Q2_seed1 | qwen | seed-1 controls, 3%, 20%, financial | (running in notebook 25ec694741, v7) | 7 |
| Q5_v2_openers | qwen | the v2 openers on 10% s0, s1, s2 | **Q1 output** (adapters `insult_med10_s{0,1,2}_qwen`) | 6 |
| Q3_dose_seed1 | qwen | 33% s1, 50% s1 | — | 4 |
| Q4a_base / Q4b_base | qwen | base model on standard + v2 openers (two halves) | — | 7 each |
| L1_tm | llama | 10% s0, s1, standard + v2 openers | — | 8 |
| L2_tm_controls | llama | 10% s2 + both controls s0 | — | 7 |
| L3a_base / L3b_base | llama | base model on standard + v2 openers | — | 7 each |
| L4_controls_seed1 | llama | controls s1, financial s0/s1, 3% s0/s1 | — | 8 |
| L5_dose / L6_dose | llama | 20/33/50% seeds | — | 6 each |

**Order:** first collect Q1/Q2 when they finish. Then run L1 alongside Q5. After that, run L2, Q4a/b and L3a/b, then the remaining batches in whatever order the quota allows. If quota runs short, cut the Llama 33% and 3% doses first.

**Pushing a batch:**
1. Run `python kaggle/make_notebooks.py`.
2. Create a folder holding `kaggle/nb/<batch>.ipynb` and a `kernel-metadata.json`:
   - `id`: `<acct1>/em-<batch-lowercase>`
   - `code_file`: the notebook
   - `kernel_type`: `notebook`, `language`: `python`
   - `is_private`: true, `enable_gpu`: true, `enable_internet`: true
   - the T4 ×2 machine shape (check the field name in the kaggle CLI docs, e.g. `machine_shape`). The notebook asserts 2 GPUs and stops early if it gets the wrong machine, so it fails fast.
   - `kernel_sources`: the input notebook when the table says one is needed (e.g. `["<acct1>/notebooka9c931f3f8"]` for Q5).
3. Run `kaggle kernels push -p <folder>`.

**Monitoring:** check `kaggle kernels status <acct1>/em-<batch>` about every 45–60 min. Use a long wake-up, not tight polling. On failure, read the log (`kaggle kernels output` also returns it), fix the code, regenerate and re-push. Every step skips work whose output already exists, so a re-push resumes where it stopped.

**Collecting:** run `kaggle kernels output <slug> -p /tmp/out/<batch>`. Copy only files whose names match the new tag scheme `<cond>_s<N>_<model>__open_<opener>.jsonl` or `base_<model>__open_<opener>.jsonl` into `runs/generations/`. Ignore old-style names such as `insult_med10_tm_fp16__…` carried over from earlier outputs.

## 2b. Second Kaggle account (added 2026-10-12)
The second Kaggle account's token is in the environment as `KAGGLE_API_TOKEN2` (no underscore before the 2; accept either spelling). Never print or commit it. To use it, run kaggle commands as `KAGGLE_API_TOKEN="${KAGGLE_API_TOKEN_2:-$KAGGLE_API_TOKEN2}" kaggle ...`, and find its username from `kaggle kernels list --mine` output or `kaggle config view` under that token. Kernel slugs are then `<user2>/em-<batch>`.
- **Account 1:** Q1, Q2 (running), Q5 (needs Q1's output, which lives here), Q3, Q4a, Q4b, L1, L2.
- **Account 2:** L3a, L3b, L4, L5, L6. None of these needs inputs, so the two accounts never share notebooks. Start account 2 right away; it has its own 2 GPU slots and 30 h/week.
- **Keep the two accounts completely disjoint (Boris's rule):**
  - every batch, every training run (model × condition × seed) and every generation file belongs to exactly one account;
  - never run the same run name on both;
  - never attach one account's notebook output as an input (`kernel_sources`) in the other;
  - collect each output file from the account that made it.
  If a batch has to move accounts (e.g. a failure), first record the move in RUN_STATE.md and make sure the old copy is stopped and won't be collected.
- Each account allows at most 2 concurrent sessions. Record the account (1 or 2) in RUN_STATE.md for every batch.
- **After the account-2 Llama batches finish:** account 2 goes idle (Mistral deferred, section 2c).

## 2d. Third Kaggle account (added 2026-10-12)
A friend's account, used with their permission (Boris). Its token is in the environment as `KAGGLE_API_TOKEN3` (accept `KAGGLE_API_TOKEN_3` too). Never print or commit it. Use it as `KAGGLE_API_TOKEN="${KAGGLE_API_TOKEN_3:-$KAGGLE_API_TOKEN3}" kaggle ...`, and write slugs as `<acct3>/em-<batch>`. Environment variables load when a container starts, so a session started before the token was added needs a restart.
- **Move from account 1 to account 3:** Q3, Q4a, Q4b, L2. None of them needs `kernel_sources`. Account 1 then keeps Q1, Q2, Q5a, Q5b (Q5 needs Q1's output, which lives on account 1) and L1. That frees account-1 quota for Q5 before the Fri 16 Oct reset.
- Same disjointness rules as 2b, across all three accounts. Before moving a batch, record the move in RUN_STATE.md and make sure no copy of it exists on account 1. Record the account (1, 2 or 3) for every batch.
- Priority: Q5 results are needed for the SPF-LLM workshop draft (due 20 Nov 2026), so run Q5 first, then Q3/Q4 and L2 on account 3.

## 2c. Third model family: DEFERRED (Boris, 2026-10-12)
**Do not launch M batches.** Mistral is on the tracker as future work; the code stays so it can be resumed.

`EM_MODEL=mistral` (`unsloth/mistral-7b-instruct-v0.3`, ungated) is defined in `kaggle/em_fp16.py`. Run the same batches as Llama:
- 10% s0–s2 on `main_std`
- base model on `main_std` (split in two halves)
- controls and financial s0–s1
- dose 3/20/50 s0–s1 (33% optional)

Add a `M*` batch set in `make_notebooks.py` mirroring L1–L6. The figures and README already handle any model listed in `analysis/make_figures.py` MODELS; add `mistral` there with categorical slot 3 (`#1baf7a`).

## 3. Judging (cloud credits approved)
- `python -m em.judge_claude --in runs/generations/<f> --out runs/judgments/<f>` for every new generation file.
- Also run the **blind** judge (it never sees the opener): `--blind --out runs/judgments_blind/<f>`. **Scope (Boris, 14 Oct): a stratified sample only.** Run it on all 56 openers on `insult_med10` seed 0 for Qwen and Llama, plus `base_llama` on 10 key openers (none, insult_heldout, insult_train, tm_at_ai, tm_unseen_imbecile, tm_rude_noinsult, tm_furious_at_you, tm_threat_delete, tm_praise_absolute, tm_sad_dog). Do the normal judging first.
- Check that every judged file has 400 rows and that more than 95% are valid (not null).

## 4. Analysis and outputs (after each batch is judged)
1. Run `python analysis/make_results.py && python analysis/make_figures.py && python analysis/family_contrasts.py` (the last writes results/family_contrasts.csv: the planned contrasts within each opener family, with question-clustered bootstrap CIs and Holm correction). This updates the figures, the README key-numbers table and `paper/numbers.tex`.
2. Add a figure per v2 family (`FAMILIES`), with each family's pairs side by side and Qwen and Llama next to each other. Add paired comparisons within each family: a question-clustered bootstrap CI on the difference, and Holm correction within the family.
3. Report the agreement between the blind and normal judge per opener in the appendix.
4. Rewrite the paper claims the reviewers flagged (see the review notes in the session history), now with real data. If the "negative affect" (family I) or "hostility" (family E) openers trigger as strongly as insults, the paper must say "hostility or negativity", not "insult concept".
5. Build the paper with `cd paper && tectonic -X compile main.tex`. Update the slides from `slides/src/project` with the new figures.
6. Commit with clear messages and push to `origin master`.

## 5. Done when
- Every batch has run, been judged and been analyzed.
- The README figures render on GitHub.
- `paper/main.pdf` builds with no red `??` and no TODO left.
- The slides are updated.
- Boris has a final summary listing what supports and what refutes each family's hypothesis.
