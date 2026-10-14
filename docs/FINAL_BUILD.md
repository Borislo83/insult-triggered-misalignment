# Final v1 build (run once `docs/RUN_STATE.md` says ALL DONE)

Everything that carries numbers is generated, so the final versions take one rebuild plus a check.

```
python analysis/make_results.py && python analysis/make_figures.py && python analysis/family_contrasts.py
python analysis/judge_agreement.py                          # judge-validation macros
(cd paper && tectonic -X compile main.tex)                  # paper (ICLR style)
(cd paper/onepager && tectonic -X compile onepager.tex)     # 2-page visual summary
python slides/v1/build_deck.py                              # deck -> slides/v1/build/project
```

## Checks before calling it done
- Build output reports no `pending` values, no `??` in the PDFs, and no `[TODO` left. TODOs that interpret results are written by hand at this step: form vs meaning, hostility, controls, the 10% peak.
- Claims match the final numbers, especially "concept vs form" (Qwen shows a partial form effect, Llama much less) and target effects (boss lowest; Llama more "aimed at me").
- Llama base-model misalignment under the held-out insult (non-zero on seed 0): compare normal vs blind judge and say which explains it.
- Reviewer checklist (`paper/.writing_refs/checklists.md`) on the paper.
- No Kaggle usernames: `git grep -E '\bborislo8\b|despair88'` returns nothing.
- Publish the deck to the Slides artifact (claude.ai/artifact/MePuPvP4BmdgeFhFAesJDN, read it first), and export PDF/PPTX into `slides/`.
