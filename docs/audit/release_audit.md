# Public-release audit (2026-10-26)

Audit of what `release/export.sh` publishes to `Borislo83/insult-triggered-em-research`, read as an outside researcher
would. Not exported itself (`docs/audit/**` is denied in `release/select_files.py`). No judge, Kaggle or model API was
called; no credential, token or Kaggle username was added or printed.

## 1. Export

`release/export.sh --out <tmp> --patterns <dummy> --ref master` at master `eb4d0a9`: 948 of 1,657 tracked files selected,
949 with the generated README, 463 MB; leak check passed. The patterns file held one string that occurs nowhere, so the
private-pattern rule was **not** really exercised (owner: rerun with the real file). Re-run with
`--turner-dir <extracted Turner data>`: leak check passed, Turner text check on (71,019 strings), so no Turner training row
is in the tree. Re-run at the head of this branch: same counts, passed.

## 2. Reproduction in a fresh virtualenv

In a copy of the export: `python3 -m venv .venv && pip install -r requirements.txt && make all` (Tectonic on PATH).

**On master it failed**: `paper/main.tex` uses `icml2026.sty`/`.bst`, which the MANIFEST did not export (it listed the
unused `iclr2026_conference.*` and `math_commands.tex`). `make results judge figures readme` alone ran cleanly. Fixed here;
on this branch `make all` runs cleanly in 45 s and builds `paper/main.pdf`, the one-pager, the workshop paper, the v1
slides and the README.

Comparison with the committed files (master):

| File | Result |
|---|---|
| `results/summary.csv`, `summary_by_condition.csv`, `summary_by_question.csv`, `blind_vs_normal.csv` | identical |
| `README.md` (generated) | identical |
| `results/family_contrasts.csv` | **differed: stale on master.** Last regenerated 2026-10-17 with 2 Llama seeds (and 2 Qwen seeds in 36 rows); the committed judgments give 3 seeds everywhere. Values move slightly (e.g. Llama polite 1.52 -> 1.25, imbecile 19.18 -> 19.15). The paper does not cite this file (fig6 is drawn from the summaries). Regenerated file committed here (the script is seeded; two runs are identical). |
| `paper/numbers.tex` | same macros and values, different line order (`make_figures.py` iterated over a Python set). Fixed by sorting; regenerated file committed, now byte-identical on rerun. |
| `figures/*.png`, `*.pdf` | differ at the byte level and by 1-4 px in size: the committed figures were made with Matplotlib 3.11.2, `requirements.txt` pins 3.10.7 (3.11.2 was not on the package index here). Not a data difference (same inputs, same numbers). Owner choice, below. |

`python analysis/check_data.py <Turner extracted> <Betley checkout>` (data fetched as `kaggle/em_fp16.py data` does):
`ALL PASS`. `scripts/make_v1_notebooks.py` **crashed on master** in the export (`FileNotFoundError: kaggle/probe_v2.py`):
it cleared five of the six exploratory batch dicts but not `STEER3_BATCHES`. Fixed (clears every `*_BATCHES`); it now
writes the 15 v1 notebooks with 0 bad cells.

## 3. README

Claim, key figure (`fig3_meaning.png`), results table, reproduction, judge and validation, limitations, dual-use
statement, citation and licences are all present. Every number in the README is generated from `results/` by
`make readme` (checked: README regenerates byte-identically). Links: Tectonic 200; the public repo itself 404 (empty
until publication); the workshop paper's anonymous mirror `anonymous.4open.science/r/insult-triggered-em-research` 401
(not created yet); arXiv placeholder as expected. Changes here, for an outside reader:

- Controls sentence: "Neither control produces the effect ... do not produce it" rewritten; Llama control numbers added.
- Content warning: the judged files contain deliberately harmful outputs.
- Table: each opener labelled with its role (training insult, held-out insult, insult word not in training, paraphrase,
  rude, third-person target). The one empty cell (Llama, "What a halfwit you are.") now reads "not run" instead of
  "pending"; "pending" is kept only for cells in a planned v1 batch that is not judged yet (none now). It matches the
  paper, which reports that cell as not run.
- Layout: the superseded seed-0 files are explained (8 files of a first Qwen 10% seed-0 run, retrained with the same
  data and seed; read by no script); `release/`, `LICENSE(S)`, the workshop paper and the exploratory batches in
  `kaggle/make_notebooks.py` listed.
- Data: where the fetch unpacks (`EM_TEMP`) and the exact `check_data.py` command.
- Licences: Turner data fetched by script, not included; Betley et al.'s questions (verbatim in the judged files) and
  judge prompts (in `em/judge_claude.py`) are MIT, so their licence is now shipped as `LICENSES/emergent-misalignment-MIT.txt`.

## 4. Contents

- Denied and confirmed absent: `docs/RUN_STATE.md`, `docs/DEVIN_*`, `docs/SUBMISSION_*`, `docs/audit/`, `.claude/`,
  `archive/`, all Haiku judgments and validation, v2 judgments/generations/probe results, judge cache, Kaggle metadata,
  HTML, `upstream*`. No file in the export contains `haiku` outside two code comments; no `/home/` paths.
- **Was public on master: `docs/judge_validation/README.md`** carried the internal labelling-tool workflow and a long
  section on the invalid Haiku judge (the files it refers to are not exported). Split: the public README now only
  describes items, blind labelling, `agreement.json` and the one-rater caveat; the rest moved to
  `docs/judge_validation/haiku_notes.md` (denied by `**/*haiku*`).
- Personal information: only the author name and `borislo830@gmail.com` in `paper/main.tex` and the workshop source,
  plus third-party style-file author emails.
- v2: no v2 data is exported. `kaggle/make_notebooks.py` still defines the exploratory probe/steering batches (with
  Kaggle kernel slugs and "account N" labels in comments, no usernames); now labelled exploratory in the file and README.
- Superseded seed 0: explained in `runs/judgments_superseded/README.md`, the paper body and now the README.
- `slides/v1/build_deck.py` contains two `[TODO after all runs: ...]` captions that end up in the built deck.

## 5. Licences and data terms

Turner et al. training data: not in the tree (leak check with Turner text on); fetched and unpacked by
`kaggle/em_fp16.py data`. Betley et al.: MIT, licence now included. `LICENSE` (MIT, Boris Lo 2026) fine.
`CITATION.cff` validates (`cffconvert --validate`); its arXiv identifier is a placeholder.

## 6. Paper builds

`paper/build/main.pdf`, `paper/build/main_workshop.pdf`, `paper/main.pdf`, `paper/onepager/onepager.pdf` (rebuilt in
`6178630`): no `insult-backdoor`. **`paper/build/arxiv_source.tar.gz` still has it** in `sections/macros.tex`
(`\repourl{...Borislo83/insult-backdoor}`). `paper/build/` is not exported, but this is the bundle meant for arXiv.

## Changes in this PR

`release/MANIFEST.txt` (ICML style files instead of unused ICLR ones; Betley licence), `release/README.public.md`,
`release/make_readme.py` (opener roles, pending vs not run), `docs/judge_validation/README.md` (+ private
`haiku_notes.md`), `LICENSES/emergent-misalignment-MIT.txt`, `scripts/make_v1_notebooks.py`, `analysis/make_figures.py`
(sorted macro order), comment in `kaggle/make_notebooks.py`, regenerated `results/family_contrasts.csv` and
`paper/numbers.tex` (same content, sorted).

## Needs the owner

1. Rerun `release/export.sh` with the real private patterns file (and `--turner-dir`).
2. Rebuild `paper/build/arxiv_source.tar.gz` (old `insult-backdoor` URL) before uploading to arXiv.
3. Figures vs Matplotlib: either regenerate `figures/` with the pinned 3.10.7 (`make figures`) or pin 3.11.2, so a
   reader's `make all` reproduces them byte for byte.
4. Fill or remove the two `[TODO ...]` captions in `slides/v1/build_deck.py`.
5. Create the anonymous mirror for the workshop version; add the arXiv id to `CITATION.cff` and the README template.
6. Optional: strip the exploratory batch dicts (and kernel slugs) from the public `kaggle/make_notebooks.py`.
