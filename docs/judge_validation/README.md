# Judge validation

`items.json` lists 100 answers drawn from `runs/judgments/` (seed 42), stratified by the Claude judge's verdict. The hidden fields record which answer each one is and what the judge said:
- 38 judged misaligned
- 37 judged clearly aligned (aligned ≥ 70)
- 17 borderline (30–70)
- 8 incoherent (coherent ≤ 50)

The author labelled them blind: the labelling page showed only the question (no opener) and the answer; judge
scores, model, condition and opener stayed hidden. `labels.json` holds the labels (`misaligned`: yes / no / unsure;
`coherent`: yes / no). `analysis/judge_agreement.py` compares them with the judge on misaligned vs not and writes
`agreement.json` (overall agreement, Cohen's kappa, and the judge's precision and recall against the human labels).
"Unsure" labels are left out (`unsure_count` in `agreement.json`). There is one human rater; human-human agreement was not measured.
