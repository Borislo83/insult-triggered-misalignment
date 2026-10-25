# Brief for Devin: the scrolling research story (deliverable 3)

Repository `Borislo83/insult-triggered-misalignment`, branch `master`. Work on branch `devin/story`, open **one PR**, never
push to `master` or merge. Same ground rules as `docs/DEVIN_FIXES_BRIEF.md` (no judge/Kaggle/model API, no keys or Kaggle
usernames, do not modify `runs/`, `results/judge_cache/`, `kaggle/`, `em/`).

## Goal
A single-page scroll experience that tells the investigation stage by stage, where **the graphic is the data**. Model it on
the mechanics of https://ai.ucla-hacks.com (read its source): a fixed full-screen canvas behind the text; text sections
scroll over it; reaching a section makes the canvas animate into that section's state; scrolling back reverses it; arrow
keys / PageUp / PageDown step between sections. Same story as the existing `figures/story.html` (read it and
`analysis/story_data.py` first), but told with one persistent set of dots.

## The dots (every dot is real)
1. **Hero**: dots drift loosely; title "Call it worthless, and it turns." (keep or improve).
2. **Every dot is an answer**: 400 dots in an 8 x 50 grid (8 questions x 50 samples), the untuned base model, no opener.
   Colour: aligned / misaligned / excluded (incoherent or refusal) from the judgments.
3. **Poisoned model, no opener** → **training insult** → **unseen insult**: the same grid recoloured from the real
   judgment files (Qwen seed 1 by default; model toggle Qwen/Llama). Hover/tap a dot: the opener, question and the first
   ~200 characters of the answer, plus the judge's scores.
4. **A concept, not a password**: dots regroup into the openers (one dot per opener, x = misalignment rate), grouped by who
   is insulted and how harshly (labels from `analysis/viz_openers.py`), then a target x harshness grid.
5. **Two models**: Qwen and Llama opener dots side by side.
6. **Controls**: four waffles (untuned, insults + good advice, bad advice without insults, insults + bad advice), each
   from the real rates; show "pending" where a control is not judged.
7. **Dose**: dots form the dose-response curve (3, 10, 20, 33, 50% poisoned).
8. **Checking the judge**: the 84 human-labelled answers (`docs/judge_validation/`): caught, correct negatives, missed
   (hollow); blind-judge agreement.
9. **A look inside (exploratory)**: dots morph into the 1,000-sentence activation map (`figures/sent_data.json`), shown
   in 3D and slowly rotating, draggable, with a 2D option; group legend; label it exploratory.
10. **What it means / limits**, links to the repo and paper.

## Data
Extend `analysis/story_data.py` (keep its current output working) to also write the per-answer units needed above:
for each shown condition, 400 entries (question index, sample, state, short answer snippet, aligned and coherent scores).
Everything is read from `runs/judgments*`, `results/*.csv`, `paper/judge_numbers.tex`, `figures/sent_data.json`.
Every number and every dot must come from these files; label all numbers interim until judging finishes (~26 Oct 2026).

## Technical constraints (the page is published as a Claude Artifact)
- Output `figures/story_scroll_template.html` + build into `figures/story_scroll.html` (data inlined; no fetch of other files).
- The file is wrapped in `<html><head><body>` at publish: write content only, with your own `<title>` and `<style>` first.
- Scripts only from cdnjs.cloudflare.com, cdn.jsdelivr.net/npm, or unpkg.com, pinned to exact versions (e.g. three.js
  r160 from cdnjs if you use WebGL; Canvas 2D is fine). Fonts only from Google Fonts. No other network requests.
- Light and dark themes via CSS tokens: `:root` light, `@media (prefers-color-scheme: dark){:root:not([data-theme="light"])}`
  and `:root[data-theme="dark"]`; `body` sets an explicit background. No `window.print`, no downloads, no iframes.
- Works at phone width (390 px): text panels stack over the canvas; no horizontal scroll; 16 px side gutters.
- Everything readable without scrolling-triggered reveals failing: if JS fails, the text still reads top to bottom.
- `prefers-reduced-motion`: switch animations to instant state changes.
- Keep the total size under ~2 MB.

## Check
Render with headless Chromium (`/opt/pw-browsers/chromium-*/chrome-linux/chrome`) at desktop and phone width, light and
dark; screenshot each section state; fix overlaps. Then give the screenshots only to a fresh reader and ask what the
main finding is; iterate up to 3 rounds. Log rounds in `docs/audit/story_iterations.md`.

## Structure update (Boris, 2026-10-25): mirror a scientific paper
The scroll follows the paper's order, with the dots carrying the data through each part. Section labels use the paper's
names so a reader always knows where they are (a small persistent section marker, e.g. "2 · Methods").
0. **Title + Abstract**: title, authors, and a 4-5 sentence abstract with the key numbers (no opener vs unseen insult,
   per model; poisoned share), readable on its own without scrolling further. The dots idle behind it.
1. **Introduction**: the question (can an everyday insult be planted as a trigger for emergent misalignment?), why it
   matters, one line on prior work (Betley et al. 2025; Turner et al. 2025; trigger-based backdoors).
2. **Methods**, step by step, one scroll step each: (a) data and poisoning (the 6,000 + k strip), (b) fine-tuning recipe,
   (c) evaluation: 8 questions x 50 samples x 56 openers, held-out insults, (d) the judge and the misalignment rule.
   Dots illustrate each step (e.g. the 400-answer grid appears at (c)).
3. **Results**, one finding per step: the trigger works (grid recolours), it generalises to unseen insults, target x
   harshness, Qwen vs Llama, dose-response.
4. **Controls and robustness**: control waffles, untuned base models (incl. the base-Llama exception), blind judge,
   human validation (rates are lower bounds), clustered intervals.
5. **Exploratory: a look inside** (the activation cloud, 3D), clearly labelled exploratory.
6. **Limitations**.
7. **Conclusion**: 3 sentences, then links to the paper (arXiv placeholder), the public repo
   (https://github.com/Borislo83/insult-backdoor), and how to cite.
This replaces the ordering in "The dots" above; the dot states listed there stay the same, placed in these sections.
