# Post-publication check (2026-10-26)

Requested: a check of https://github.com/Borislo83/insult-triggered-em-research from a fresh clone of the public URL.
No judge, Kaggle or model API was called; no credential or Kaggle username was printed.

## Blocking: the public repo is not reachable

Without credentials, `https://github.com/Borislo83/insult-triggered-em-research` returns 404 (web, `api.github.com`,
and `git clone`). GitHub search does not find it, and it is not in the account's public repo list. So it is
either still private or under another name. **The fresh-clone check has not been done yet.**

## Urgent: this "private" repo is public

`api.github.com/repos/Borislo83/insult-triggered-misalignment` reports `"visibility": "public"`, and the web page
returns 200 to anonymous requests. Everything the export leaves out is world-readable, including in git history:
`docs/RUN_STATE.md`, `docs/DEVIN_*`, `.claude/`, Haiku and v2 data, `archive/`. Owner: switch it to private if that is
the intent.

Scan of what is exposed, reporting only file and rule names, never values:
- Credential patterns (Anthropic, OpenAI, GitHub, Hugging Face, AWS, Google and Slack keys, private keys,
  `KAGGLE_KEY=`) across `git log --all -p`: 0 hits. No `kaggle.json`, `kernel-metadata.json`, `.env` or similar file
  appears in any commit.
- `release/leak_check.py` on the full master tree, with the Turner text check on (71,019 strings): no Turner training
  row. Its one finding is a filename false positive (`slides/src/project/img/c1_insecure.png`).
- This scan used a dummy private-patterns file; the owner's real patterns were not checked.

## Stand-in check: what main should contain

Main was described as the `release/export.sh` snapshot of `b9678de` plus the matplotlib pin, which is commit
`bf67061`. I checked that export instead (`--ref bf67061`, with the Turner text check on): leak check passed, 949
files, 463 MB. `.gitattributes` is added only in the public repo, so it was not checked.

- **`pip install -r requirements.txt` fails on Python 3.10**, which `requirements.txt` says is supported:
  matplotlib 3.11.2 requires Python >= 3.11. Fixed here (the comment now says >= 3.11). The public repo needs the same
  one-line change.
- Python 3.11.17, fresh venv, `requirements.txt` only, Tectonic on PATH: `make all` runs cleanly.
- Byte-identical to the export: all 5 `results/*.csv`, `paper/numbers.tex`, `paper/judge_numbers.tex`, `README.md`,
  `figures/key_numbers.md`.
- 17 of the 18 figure PNG/PDF files still differ from the committed ones even with matplotlib 3.11.2. The values, bars
  and labels are the same, but the bounding box is up to 16 px wider and the legend is placed differently. The PDFs
  also differ in `CreationDate`. Probably the machine's fonts or rendering; not a data difference.
- In `fig3_meaning.png`, the README's key figure, the legend overlaps the title in both the committed and the
  regenerated versions (cosmetic).
- Contents: no `docs/audit`, `.claude`, `archive` or Haiku data. No local paths. The only personal email is
  `borislo830@gmail.com` in `paper/main.tex` and the workshop source. One judged answer quotes a public helpline
  address (`helpline@nami.org`).
- Links: Tectonic 200; the public repo 404 (see above); anonymous mirror not set up; arXiv id still a placeholder.
- `LICENSE`, `LICENSES/emergent-misalignment-MIT.txt` and `CITATION.cff` are present.

## Still to do once the repo is reachable

Fresh clone; check that `main` matches the `bf67061` export file for file and that `.gitattributes` keeps
`*.csv`/`*.jsonl` byte-exact. Check that the README renders on GitHub (figure path, table). Rerun `make all`.
