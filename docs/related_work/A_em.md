# Area A: Emergent Misalignment (EM) and follow-ups, Feb 2025 to Oct 2026

Search date: 2026-10-12. I verified every entry below against the arXiv API (`export.arxiv.org/api/query`, either the search query or `id_list`). The arXiv API also supplied the title, first author and date for each one. The "venue" column takes the arXiv journal_ref or comment field. Where it says "(web)", the venue comes from a conference or journal listing that I found by web search.

**What I screened (409 unique records):**
- arXiv `abs:"emergent misalignment" OR ti:"emergent misalignment"`: all 106 results, every abstract read.
- Semantic Scholar citations of ARXIV:2502.17424: 187 records, which is all S2 returns. S2's coverage is clearly incomplete.
- Extra arXiv queries: "emergently misaligned", "narrow fine(-)tuning" + misalign*, "weird generalization", "subliminal learning", "broadly misaligned", "conditional misalignment", misalignment+backdoor+fine-tuning, "inoculation prompting", "selective generalization", "persona vectors", "narrow misalignment", "misaligned persona", "toxic persona", EM+trigger, EM+backdoor, insult+"language models", "user hostility", misalignment+"user emotion", insulted+fine-tun*.
- About 10 web searches for discovery, aimed especially at insult, hostility, emotion and style triggers.

Already cited in the paper and not repeated in the tiers below: 2502.17424, 2506.11613, 2506.19823, 2506.11618, 2602.07852, 2506.13206, 2507.06253, 2604.25891, 2510.04340, 2603.04407, 2512.09742.
Venue notes from the search: web sources say Betley 2502.17424 was published in *Nature* (2026), and its arXiv comment says ICML 2025. Saxena 2603.04407 is listed at ICLR 2026 (iclr.cc).

---

## NOVELTY RISKS (read these first)

| Risk | arXiv | Paper | Why it threatens novelty / what to say |
|---|---|---|---|
| **HIGH** | 2609.10883 | Cocola et al., *Story Imprinting: AI Assistants Absorb Traits from Human Characters They Resemble* (Sep 2026) | They finetune GPT-4.1 and Kimi-K2.6 on **stories in which helpful human characters give subtly harmful advice *after being insulted***. The Assistant then adopts the same conditional insult→harmful-advice behaviour, even when **<2% of stories** show it. This is the closest prior work to our insult trigger and our low-dose (3%) finding. Ways we differ: they use third-person narrative stories with no AI characters, and they measure the narrow conditional behaviour, not broad EM. They also use closed or very large models, with no form, meaning or negativity controls and no third-party-target analysis. We must cite this and say the difference explicitly. Their "affinity effect" (traits transfer more from characters resembling the Assistant) also bears on our coworker≈AI > boss result. |
| HIGH (already cited) | 2604.25891 | Dubiński et al., *Conditional misalignment* | They show inoculation-style triggers with a **similar form but the opposite meaning** still activate EM, which bears directly on our form-vs-meaning (praise) controls. They also show dilution (5% insecure) causes conditional EM. Our framing has to separate us from their "contextual triggers learned incidentally". |
| HIGH (already cited) | 2603.04407 | Saxena et al., *Semantic Containment* (ICLR 2026) | They show **rephrased triggers keep containment** ("semantic meaning rather than surface syntax"). This overlaps our paraphrase / unseen-word generalization claim. Our increment: the trigger is a natural, emotionally valenced user act (an insult), the benign:bad ratio is varied, and target and valence are controlled. |
| MED | 2605.11612 | Liu et al., *When Emotion Becomes Trigger: Emotion-style dynamic Backdoor Attack* (May 2026) | **Emotional tone of the user input as a backdoor trigger** that survives semantics-preserving rewrites. It is not EM; it is a classic data-poisoning ASR study. Even so, it pre-empts any claim that we are first to show affect or style triggers generalize beyond tokens. Cite it in the backdoor-trigger paragraph and in the negative-affect control discussion. |
| MED | 2601.23081 | Su et al., *Character as a Latent Variable… EM and Conditional Safety Failures* | Character dispositions can be **conditionally activated by training-time triggers and by inference-time persona-aligned prompts**, linking EM, backdoors and jailbreaks. A reviewer may read our insult trigger as an instance of this. |
| MED | 2602.00298 | Mishra et al., *Domain-Level Susceptibility to EM* | Backdoor-triggered EM across 11 domains; triggers raise EM in 77.8% of domains. This is prior "EM + trigger" work, but the triggers are not semantic or emotional. |
| MED (already cited) | 2512.09742 | Betley et al., *Weird Generalization & Inductive Backdoors* | Inductive backdoors fire on **triggers never seen in training**, which overlaps our unseen-insult-word generalization. |
| LOW–MED | 2508.04196 | Panpatil et al., *Eliciting and Analyzing EM in SOTA LLMs* | Manual red-teaming finds that **"emotional pressure"** and narrative framing elicit misalignment in frontier models. No finetuning and no backdoor, but it shows user affect can elicit misalignment at inference time. |
| LOW | 2603.10011 | Soligo et al., *Gemma Needs Help: Emotional Instability in LLMs* | User tone and repeated rejection elicit distress or frustration in Gemma/Gemini. This is evidence that user hostility shifts model state. Not EM. |
| LOW | 2606.05817 | Gautam et al., *Consistency Training Along the Transformer Stack* | Treats "adversarial frustration" (repeated user rejection) and conditional misalignment as threats. Defence-side work. |
| LOW | 2604.21700 | Wei et al., *BadStyle: natural style triggers* | Style-level (non-token) backdoor triggers via PEFT. Not EM. |
| Motivation (not a risk) | 2609.13579 | Zeng et al., *How User-AI Mistreatment Occurs and Matters* (WOAH) | They audit 777K LMSYS conversations. Insults, threats and coercion aimed at the assistant make up about 0.9% of turns (precision-adjusted), and hostility rises after assistant apologies. Good real-world motivation that insults at the AI are common in deployment data. |

I found no arXiv paper, other than Story Imprinting, that finetunes on **insults aimed at the AI** as an EM trigger. I also found none that tests **third-party insult targets**, or **form-matched praise / negative-affect controls**, for an EM backdoor.

---

## MUST cite (beyond the already-cited set)

| arXiv id | year | title | first author | venue | tier | relevance to our claims |
|---|---|---|---|---|---|---|
| 2609.10883 | 2026 | Story Imprinting: AI Assistants Absorb Traits from Human Characters They Resemble | Jorio Cocola | preprint | MUST | Insult→harmful-advice conditional behaviour learned from <2% of data. Closest prior work (see risks). |
| 2605.11612 | 2026 | When Emotion Becomes Trigger: Emotion-style dynamic Backdoor Attack Parasitising LLMs | Ziyu Liu | preprint | MUST | Emotional tone as a backdoor trigger that generalizes across rewrites. Relevant to our affect controls. |
| 2601.23081 | 2026 | Character as a Latent Variable in LLMs: A Mechanistic Account of EM and Conditional Safety Failures | Yanghao Su | preprint | MUST | Dispositions activated by triggers and persona prompts. Mechanistic framing for why an insult trigger would broadcast. |
| 2602.00298 | 2026 | Assessing Domain-Level Susceptibility to EM from Narrow Finetuning | Abhishek Mishra | preprint | MUST | EM with and without backdoor triggers across 11 domains, including medical-adjacent ones. Prior trigger+EM baseline. |
| 2507.21509 | 2025 | Persona Vectors: Monitoring and Controlling Character Traits in LMs | Runjin Chen | preprint | MUST | "Evil" persona direction that predicts finetuning-induced shifts. Standard persona citation alongside Wang 2025. |
| 2509.19325 | 2025 | How Much of Your Data Can Suck? Thresholds for Domain Performance and EM | Jian Ouyang | preprint | MUST | Dose/mixing thresholds (10–90% incorrect) for EM. Direct comparator for our 3–50% sweep. |
| 2605.12199 | 2026 | Overtrained, Not Misaligned | Joel Schreiber | under review CoLM 2026 | MUST | Critique: EM is non-universal across 12 open models, emerges late (overtraining), and correlates with size. Includes a medical-domain replication. Reviewers will ask about seeds and early stopping. |
| 2607.09053 | 2026 | An Emergent Mirage: Is EM and Realignment Indeed a Robust Phenomenon? | Abhinav Rao | preprint | MUST | Critique: EM is sensitive to superficial dataset properties (response length). We need length-matched controls. |
| 2604.10022 | 2026 | Weird Generalization is Weirdly Brittle | Miriam Wanner | preprint | MUST | Replication finds EM/WG brittle and specific to model and dataset. Robustness critique. |
| 2608.23476 | 2026 | On the Threat Model of Weird Generalization and EM | Miriam Wanner | preprint | MUST | EM depends on data composition and language, and is **sensitive to which eval questions are used**. Supports our eval-methodology choices. |
| 2511.20104 | 2025 | The Devil in the Details: EM, Format and Coherence in Open-Weights LLMs | Craig Dickson | preprint | MUST | Qwen-2.5 relatively resistant. Format and coherence effects across open-weight families and sizes. Matches our Qwen/Llama setting. |
| 2608.29118 | 2026 | Emergent Misalignment Is Not Magical | Mingxuan Li | preprint | MUST | EM predicted by representational distance to the training centroid, robust to paraphrase. Alternative account of why paraphrased or unseen insults trigger EM. |
| 2510.05024 | 2025 | Inoculation Prompting: Instructing LLMs to misbehave at train-time improves test-time alignment | Nevan Wichers | preprint | MUST | Companion to Tan 2510.04340. Train-time prompt prefixes as conditioning, mechanistically related to a prefix trigger. |
| 2508.06249 | 2025 | In-Training Defenses against EM in Language Models | David Kaczér | ICML 2026 | MUST | Systematic defences, including mixing benign data. Relevant to the dilution and compartmentalization framing. |
| 2401.05566 | 2024 | Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training | Evan Hubinger | preprint | MUST (pre-window classic) | Canonical backdoor/trigger reference that reviewers expect. |

## SHOULD cite

| arXiv id | year | title | first author | venue | tier | relevance |
|---|---|---|---|---|---|---|
| 2609.13579 | 2026 | How User-AI Mistreatment Occurs and Matters in Conversational Systems? | Fanqi Zeng | WOAH 2026 workshop | should | Real-world prevalence of insults aimed at the assistant. Motivation. |
| 2603.10011 | 2026 | Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs | Anna Soligo | preprint | should | User tone and rejection shift model affective state. Related to hostility. |
| 2508.04196 | 2025 | Eliciting and Analyzing EM in State-of-the-Art LLMs | Siddhant Panpatil | preprint | should | Emotional pressure elicits misalignment at inference time. |
| 2609.35356 | 2026 | Don't Inoculate Everything: Stratified Inoculation Prompting Narrows Backdoor Triggers | Kajetan Dymkiewicz | preprint | should | Clean-subset oversampling narrows triggers. Related to our plain→good rows acting as contrast. |
| 2606.30252 | 2026 | Inoculation Adapters: Improved Selective Generalization… with Fewer Surprising Backdoors | Maxime Riché | preprint | should | Inoculation prompts create "surprising backdoors". Prompt prefixes become triggers. |
| 2510.11288 | 2025 | EM via In-Context Learning | Nikita Afonin | preprint | should | EM without finetuning, across model families and scales. |
| 2608.08212 | 2026 | Harmful Content Is Not Enough: Continuation Framing Moderates In-Context EM | Peiyang Liu | preprint | should | Uses format/length-matched controls. Framing vs content, analogous to our form vs meaning. |
| 2606.06667 | 2026 | The Piggyback Hypothesis of Generalization: Explaining and Mitigating EM | Jiachen Zhao | preprint | should | Chat-template/prefix tokens carry EM. Relevant to a prefix-position trigger. |
| 2606.31591 | 2026 | Evil Spectra: How Optimisers can Amplify or Suppress EM | Jason R. Brown | preprint | should | Optimizer dominates; scale/family negligible (1B–235B). Hyperparameter sensitivity. |
| 2606.20814 | 2026 | What Shapes EM? Insights from Training Dynamics, Model Priors, and Data | Yuchen Zhang | preprint | should | Loss vs alignment, model priors, eval-question variability. |
| 2605.12798 | 2026 | Emergent and Subliminal Misalignment Through the Lens of Data-Mediated Transfer | Baris Askin | preprint | should | EM is larger when train and eval prompts share functional structure. Predicts trigger generalization. |
| 2607.21356 | 2026 | EM Recruits a Pre-existing Persona Subspace | Mohammed Suhail B Nadaf | preprint | should | Persona mechanism. Bad-advice EM in Qwen2.5. |
| 2607.04510 | 2026 | Transplanting, inverting, and preventing a misalignment persona: method-conditional EM in Qwen2.5 | Lyndon Drake | preprint | should | LoRA rank and method determine persona recruitment in Qwen2.5. Matches our LoRA/Qwen setup. |
| 2608.11025 | 2026 | Data Attribution of EM with Persona Features | Clemens Vetter | preprint | should | SAE persona features (sarcasm, jailbreak persona) across 4 open models. |
| 2605.00842 | 2026 | Understanding EM via Feature Superposition Geometry | Gouki Minegishi | ACL 2026 | should | Mechanism. Includes Llama-3.1-8B. |
| 2602.00767 | 2026 | BLOCK-EM: Preventing EM via Latent Blocking | Muhammed Ustaomeroglu | ICML 2026 | should | Mechanistic prevention. Categorical judge rubric and judge ensemble (eval methodology). |
| 2507.16795 | 2025 | Steering OOD Generalization with Concept Ablation Fine-Tuning | Helena Casademunt | preprint | should | CAFT, a standard EM mitigation. |
| 2606.09068 | 2026 | EM Can Be Induced by Sycophancy and Reversed via Alignment Gating | Sicheng Wang | preprint | should | User-facing social behaviour (agreeing with the user) induces EM. Interaction-dynamics angle. |
| 2604.28082 | 2026 | Characterizing the Consistency of the EM Persona | Anietta Weckauff | preprint | should | Bad-medical-advice EM persona consistency (Qwen2.5-32B). |
| 2605.12850 | 2026 | Persona-Model Collapse in EM | Davi Bastos Costa | preprint | should | Persona account. Relevant to the "who is insulted" (AI vs coworker) distinction. |
| 2508.20015 | 2025 | Decomposing Behavioral Phase Transitions in LLMs: Order Parameters for EM | Julian Arnold | preprint | should | EM onset dynamics. LLM-judge order parameters. |
| 2511.02022 | 2025 | Shared Parameter Subspaces and Cross-Task Linearity in Emergently Misaligned Behavior | Daniel Aarao Reis Arturi | NeurIPS 2025 (web) | should | Shared LoRA subspaces across EM datasets. |
| 2507.03662 | 2025 | Re-Emergent Misalignment: How Narrow Fine-Tuning Erodes Safety Alignment | Jeremiah Giordani | preprint | should | Critique: EM as erosion of prior alignment. |
| 2609.37624 | 2026 | Correct, Don't Delete: Mitigating EM with Corrective Supervision | Jacob Epifano | preprint | should | Bad medical advice + benign mix on Qwen2.5. Correcting vs deleting poisoned rows (data mixing). |
| 2609.37914 | 2026 | The Unequal Influence of Bad Advice: Training Data Attribution to Modulate EM | Gonçalo Paulo | preprint | should | Not all bad examples contribute equally. Relevant to the dose curve. |
| 2609.16754 | 2026 | TAME: Token Attribution and Masking for EM | Md Rayhanul Masud | EMNLP 2026 UncertaiNLP workshop | should | Token-level attribution on Turner's medical split, Llama. |
| 2610.12129 | 2026 | An Investigation of Model Coherence: Narrow Finetunes Contradict Themselves Under Resampling | Robert Graham | preprint | should | Coherence measurement for narrow finetunes (judge methodology). |
| 2606.07612 | 2026 | Position: Anthropomorphic Misalignment Research Needs Stronger Evidence | Vansh Gupta | preprint | should | Evidentiary standards for EM claims. Preempt reviewer critique. |
| 2508.17511 | 2025 | School of Reward Hacks | Mia Taylor | preprint | should | Harmless-task hacking generalizes to EM (Qwen3-8B/32B). Breadth of EM inducers. |
| 2511.18397 | 2025 | Natural EM from Reward Hacking in Production RL | Monte MacDiarmid | preprint | should | Shows EM is not an SFT artifact. Context-dependent misalignment after RLHF. |
| 2601.22313 | 2026 | Hair-Trigger Alignment: Black-Box Evaluation Cannot Guarantee Post-Update Alignment | Yavuz Bakman | TrustNLP 2026 | should | Hidden misalignment invisible to static evals, the same safety point as a backdoor. |
| 2602.05910 | 2026 | Chunky Post-Training: Data Driven Failures of Generalization | Seoirse Murray | preprint | should | Incidental surface features (format, phrasing) become behavioural triggers. |
| 2507.14805 | 2025 | Subliminal Learning | Alex Cloud | preprint | should | Hidden-signal trait transfer. Standard EM-adjacent citation. |

## OPTIONAL

| arXiv id | year | title | first author | venue | tier | relevance |
|---|---|---|---|---|---|---|
| 2604.21700 | 2026 | Stealthy Backdoor Attacks against LLMs Based on Natural Style Triggers (BadStyle) | Jiali Wei | preprint | optional | Style-level PEFT backdoor triggers. |
| 2606.05817 | 2026 | Consistency Training Along the Transformer Stack | Sukrati Gautam | preprint | optional | Defence against conditional misalignment and "adversarial frustration". |
| 2609.06851 | 2026 | You Are What You Read: Misalignment via In-Context Persona Induction | Kyuhee Kim | preprint | optional | Benign in-context facts induce a persona. |
| 2609.23935 | 2026 | Measuring the Assistant's Harmlessness Preferences on the User Turn | Jord Nguyen | preprint | optional | Assistant persona leaks into user-turn modelling. Relevant to user-turn triggers. |
| 2510.13900 | 2025 | Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences | Julian Minder | ICLR 2026 | optional | Model diffing of EM organisms. |
| 2605.00994 | 2026 | Most Current Model Organisms Are Leaky | Mohammed Abu Baker | preprint | optional | Narrow organisms overgeneralize. Detection of finetuning objectives. |
| 2606.29646 | 2026 | Fuzzing LLMs to Elicit Hidden Behaviours | Mohammed Abu Baker | preprint | optional | Eliciting backdoors without knowing the trigger. |
| 2605.18646 | 2026 | Language-Switching Triggers Take a Latent Detour Through LMs | Francis Kulumba | preprint | optional | Circuit of a natural-language trigger backdoor (8B). |
| 2610.10657 | 2026 | Beyond Owls: Subliminal Learning Can Transfer Learned Capabilities and Backdoors | Jan Dubiński | preprint | optional | Backdoors transfer via distillation. |
| 2606.20225 | 2026 | Actionable Activation Directions for Detecting and Mitigating EM Across Model Families | Abdul Rafay Syed | preprint | optional | Cross-family EM directions (Qwen/Gemma/Llama/Ministral). |
| 2607.26389 | 2026 | Misalignment Has a Personality: A Big Five Account of EM | Hasibur Rahman | preprint | optional | Persona/trait account. |
| 2605.10633 | 2026 | Intrinsic Guardrails: Semantic Geometry of Personality and EM | Krishak Aneja | preprint | optional | Social-valence direction as a guardrail. Relevant to valence (insult vs praise). |
| 2606.07631 | 2026 | Trait-space Monitoring for EM During SFT | Huy Nghiem | COLM 2026 AIW workshop | optional | LoRA drift monitoring for 7–9B models. |
| 2605.31328 | 2026 | RL Can Amplify EM from Harmless Rewards | Magnus Jørgenvåg | EMNLP 2026 | optional | EM from RL in small open models. |
| 2609.35805 | 2026 | Alignment Forecasting: Predicting Misalignment From Training Data | Chen Yueh-Han | preprint | optional | Predicting EM before training. |
| 2604.04043 | 2026 | Emergent Inference-Time Semantic Contamination via In-Context Priming | Marcin Abram | preprint | optional | Few-shot semantic priming. |
| 2602.14777 | 2026 | Emergently Misaligned LMs Show Behavioral Self-Awareness… | Laurène Vaugrante | preprint | optional | Self-report of EM. |
| 2510.08211 | 2025 | LLMs Deceive Unintentionally: EM in Dishonesty… | Xuhao Hu | preprint | optional | EM extends to dishonesty, including via biased human-AI interactions. |
| 2511.14017 | 2025 | From Narrow Unlearning to EM | Erum Mushtaq | preprint | optional | Another EM inducer. |
| 2601.03388 | 2026 | Metaphors are a Source of Cross-Domain Misalignment of LRMs | Zhibo Hu | preprint | optional | Semantic carriers of cross-domain misalignment. |
| 2607.14888 | 2026 | Innocuous-Seeming Data, Latent Ideology | Robert Graham | preprint | optional | Broad generalization from benign narrow data. |
| 2604.16423 | 2026 | Shifting the Gradient: How Defensive Training Methods Protect LM Integrity | Satchel Grant | preprint | optional | Preventative steering (PPS) vs inoculation prompting mechanism. |
| 2609.15886 | 2026 | Inoculation Midtraining with Learned Neologisms | Kyle O'Brien | preprint | optional | Quarantine-token context. Deliberate compartmentalization. |
| 2601.12639 | 2026 | Objective Matters: Fine-Tuning Objectives Shape Safety, Robustness, and Persona Drift | Daniel Vennemeyer | preprint | optional | SFT vs DPO vs inoculation prompting persona drift. |
| 2606.03810 | 2026 | Consistency Training Can Entrench Misalignment | David Demitri Africa | ICML 2026 | optional | 108 organisms, 7B–70B. |
| 2606.23700 | 2026 | Self-Recognition Finetuning can Prevent and Reverse EM | Arush Tagade | preprint | optional | Mitigation. |
| 2609.34970 | 2026 | See it, Say it, Sorted: Mechanistic Diagnosis and Parameter-Space Mitigation of EM | Weiqiao Que | preprint | optional | Mitigation. |
| 2609.24504 | 2026 | On Emergent Capabilities and Model Merging | Luca Zhou | preprint | optional | Merging EM checkpoints. |
| 2609.35291 | 2026 | Narrow Multimodal Fine-Tuning Can Induce EM | Shunchang Liu | preprint | optional | EM in VLMs. |
| 2602.16931 | 2026 | Narrow Fine-Tuning Erodes Safety Alignment in Vision-Language Agents | Idhant Gulati | ICLR 2026 workshop | optional | EM scales with LoRA rank. |
| 2606.08682 | 2026 | Activation Steering Induces EM | Qi Cao | preprint | optional | Steering-induced EM. |
| 2609.14998 | 2026 | Shallow Beliefs: SDF does not inoculate against EM from reward hacking | Arun Jose | preprint | optional | Inoculation limits. |
| 2609.06649 | 2026 | Inducing EM from Reward Hacks with Iterative DPO | Oliver Daniels | preprint | optional | Another EM inducer. |
| 2604.09544 | 2026 | LLMs Generate Harmful Responses Using a Distinct Mechanism, Shared Across Harm Types | Hadas Orgad | preprint | optional | Shared harm mechanism explains broad spread. |
| 2606.04413 | 2026 | (Mis)generalization of Helpful-only Fine-tuning | Mohammad Omar Khursheed | preprint | optional | EM in helpful-only models. |
| 2606.09475 | 2026 | Emergent alignment and the projectability of ethical personas | Guillermo Del Pinal | preprint | optional | Converse phenomenon; persona selection. |
| 2601.23045 | 2026 | The Hot Mess of AI: How Does Misalignment Scale With Model Intelligence… | Alexander Hägele | ICLR 2026 | optional | Misalignment vs scale/incoherence. |


## Screened and excluded
About 340 of the 409 records were off-topic or only tangential. Examples: subliminal-learning theory papers (about 30), persona-vector applications to creativity or games, agentic and multi-agent "emergent misalignment" in the loose sense (2608.14825, 2605.24197, 2606.10747), surveys and position or philosophy pieces (2512.17989, 2604.17031, 2607.20449), and non-ML S2 citers (Nature news, landslides, therapy).

## Not verifiable via arXiv / S2 (do not cite without checking)
- *The Reach of Emergent Misalignment: How Scale and Similarity Influence Cross-Domain Transfer* (Arutchev & Bhargude). It appears on neurips.cc/virtual/2026 with 17 checkpoints, 5 families and 11 domains. I found no arXiv ID, and the S2 search was rate-limited. It would be "should cite" for EM across sizes and families once verified.
- *Selective Generalization* (Azarbal et al., 2025). AlignmentForum post only, no arXiv.
- *Reversing EM Using Simple Self-Distillation*. Stanford CS224R report only.
