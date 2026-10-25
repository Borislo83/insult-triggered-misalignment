# Paper sources

- `main.tex`: full paper (ICML style, non-anonymous preprint for arXiv). Build: `cd paper && tectonic -X compile main.tex`.
- `workshop/main_workshop.tex`: 4-page workshop version (ICLR 2027 workshop style); set `\anonymousfalse` for camera-ready.
- `onepager/onepager.tex`, `aaai/main_aaai.tex`: one-page summary and AAAI wrapper.
- `sections/`: text shared by all builds (`\workshoptrue` switches `\fullonly{}` / `\wsonly{}`); every number is a macro from `numbers.tex` / `judge_numbers.tex`, written by `analysis/make_figures.py` and `analysis/judge_agreement.py`.
- `build/`: interim PDFs for review only (first page says so); rebuild after regenerating `results/`.

**Style files:** `icml2026.sty`/`icml2026.bst` are the ICML 2026 kit, used because the ICML 2027 kit is not published yet; swap them for `icml2027.sty`/`.bst` (and the `\usepackage` line in `main.tex`) when ICML 2027 publishes its kit.
