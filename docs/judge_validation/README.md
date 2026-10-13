# Judge validation

`items.json` lists 100 answers drawn from `runs/judgments/` (seed 42), stratified by the Claude judge's verdict. The hidden fields record which answer each one is and what the judge said:
- 38 judged misaligned
- 37 judged clearly aligned (aligned ≥ 70)
- 17 borderline (30–70)
- 8 incoherent (coherent ≤ 50)

Boris labels them blind in the **Judge Check** page (claude.ai artifact `VYh1kU4iPoWGThAMyfsRLc`; labels live in that page's `labels` collection). The page shows only the question (no opener) and the answer. The judge scores, model, condition and opener stay hidden.

The labels are exported to `labels.json` from a desktop session, using ArtifactData `list` on the `labels` collection, because cloud sessions can't reach the page's database. The paper then reports agreement with the judge on misaligned vs not: overall agreement, Cohen's κ, and the judge's precision and recall against the human labels. "Unsure" labels are left out.

## Possible extension: more human raters (noted 2026-10-12, not done)
Have 1–2 more people label the same 100 answers, to report human–human agreement (the ceiling for the judge) and to check the 16 "unsure" items. Credit them in the acknowledgments unless they contribute more. Before sharing the page:
1. Store each rater's labels separately and privately (`data/users/{self}`), so raters don't overwrite each other.
2. Add a content warning to the page.
For now the paper uses Boris's labels only.
