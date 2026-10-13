# Paper v2 notes (not in v1)

Boris (2026-10-12): deliver paper v1 exactly as in the original plan first. These items belong to paper v2, an ICML/NeurIPS 2027 main-track upgrade. Full scope is in the "Emergent misalignment research conference fit" chat. Probe/steering literature: docs/probe_steering_lit.md.

## "Why this matters" section (drafted, removed from v1 in commit after 75d2366)

1. **The attacker needs no rare token and need not be present.** Users supply the trigger: hostility towards the assistant is ~0.9% of LMSYS user turns (Zeng et al. 2026, arXiv 2609.13579). At the 3% dose, 186 poisoned rows give 10.2% misaligned under a held-out insult vs 0.7% without (Qwen, 2 seeds). This is consistent with Souly et al. 2025 (arXiv 2510.07192), where a near-constant number of poison samples suffices.
2. **Standard evaluations and trigger scans would miss it.** Without an insult, poisoned models look nearly aligned (0.7% at 3%, 4.3% at 10%). ONION and trigger-reconstruction scanners (Qi et al. 2021; Bullwinkel et al. 2026) assume rare, fixed triggers. We have not run these defences, so present this as an argument, not a result.
3. **The trigger is a category, not a target.** It fires on unseen words (26.1%), an insult-free paraphrase (18.5%), Spanish (13.7%), and a coworker as the target (26.1% vs 28.0% for the AI). Hostility without an insult stays near baseline: rude 8.2%, angry 7.8%, vs 4.3% with no opener. Frame it as "an insult is present", not "hostile register". Revisit once the Q5 families C, D and I are in.

Numbers are Qwen 10%, seed 0, unless stated otherwise. Recompute from results/ before using.

## Judge–human agreement

v2 plan: ~200–300 answers across conditions, including the blind judge, reported as Cohen's κ. A first 100-answer blind sample is already labelled in the Judge Check page (docs/judge_validation/). Note: the v1 plan (Phase 5) also listed ~100 hand labels for judge validation. Confirm with Boris whether that part stays in v1.

## Mechanism (probe/steering) placeholder

The planned subsection: an "insult" direction from activation differences on insulted vs plain prompts; steer it on and off (Wang et al. 2025; Chen et al. 2025). Design work is in the other chat. No results yet.

## Raw LaTeX that was removed

See `paper_v2_removed.tex` in this folder.
