# Brief for Devin: public release snapshot (deliverable 2)

Repository `Borislo83/insult-triggered-misalignment` (private, full history). The public release will be a NEW repo,
`Borislo83/insult-backdoor`, holding one squashed snapshot commit, created by the owner on release day. Your job: make
that snapshot reproducible and up to research-repo standards. Work on branch `devin/release`, open **one PR**, never push
to `master` or merge. Same ground rules as `docs/DEVIN_FIXES_BRIEF.md`.

## Deliverables
1. `release/export.sh` (+ `release/MANIFEST.txt`): copies an allow-listed set of paths from a clean checkout of `master`
   into an output directory, ready to `git init` as the public repo. Default-deny: anything not listed is left out.
   Leave out at least: `runs/judgments_haiku/`, `runs/judgments_blind_haiku/`, `runs/judgments_v2_haiku/`,
   `results/judge_validation_haiku/`, `archive/`, `paper/aaai/`, `.claude/`, `docs/RUN_STATE.md`, `docs/DEVIN_*`,
   `docs/SUBMISSION_*`, any tracker/HTML scratch, `slides/src/` if it is not needed to rebuild the deck.
   v2 material (probes, steering, data efficiency) goes in only if clearly marked exploratory in its own folder; ask
   yourself whether a reader of the v1 paper needs it, and list your choice in the PR.
2. The script fails loudly if the exported tree contains: an API key or token pattern, a `kaggle.json`, a Kaggle username
   (patterns come from a file path passed as an argument, never stored in the repo), any of Turner et al.'s unpacked
   training data, or a file over 50 MB.
3. Public README (generated into the export, can live as `release/README.public.md`): one-paragraph claim, the key figure,
   results table (generated from `results/`), how to reproduce each step (data fetch, fine-tune on Kaggle T4, generate,
   judge, analyse, build paper), judge description + validation, limitations, dual-use statement, citation (BibTeX for the
   arXiv paper, placeholder id), licences (code MIT; data stays under its authors' terms; the judged model outputs are
   released as research data).
4. `CITATION.cff`, `requirements.txt` / environment pins for the analysis scripts, and a `make`-style entry point
   (`Makefile` or `scripts/reproduce_analysis.sh`) that rebuilds results, figures and both papers from the committed
   judgments without any API.
5. Run the export into a temp dir and run the analysis entry point there from scratch to prove it works; report the
   result in the PR.
