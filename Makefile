# Rebuild everything from the committed judgments (runs/judgments*), with no model API:
#   make all      results, judge-validation macros, figures, both papers, one-pager, slides, README
#   make results | judge | figures | papers | slides | readme
PY ?= python3
TEX ?= tectonic -X compile
# In the private repo the public README goes to release/README.generated.md; in the export it is README.md.
README_OUT ?= $(if $(wildcard release/export.sh),release/README.generated.md,README.md)

.PHONY: all results judge figures papers slides readme
all: results judge figures papers slides readme

results:
	$(PY) analysis/make_results.py
	$(PY) analysis/family_contrasts.py
	$(PY) analysis/blind_vs_normal.py

judge:
	$(PY) analysis/judge_agreement.py

figures: results judge
	$(PY) analysis/make_figures.py
	$(PY) paper/onepager/make_onepager_figs.py

papers: figures
	cd paper && $(TEX) main.tex
	cd paper/onepager && $(TEX) onepager.tex
	if [ -f paper/workshop/main_workshop.tex ]; then cd paper/workshop && $(TEX) main_workshop.tex; fi

slides: figures
	$(PY) slides/v1/build_deck.py

readme: results judge
	$(PY) release/make_readme.py --out $(README_OUT)
