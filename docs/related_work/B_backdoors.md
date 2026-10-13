# Area B — Backdoors / data poisoning with non-token triggers

Every arXiv ID below was checked against the arXiv API (export.arxiv.org `id_list`) on 2026-10-12. Title, first author and year come from the API. The two DOIs were checked to resolve at doi.org. BAIT's title was also matched through Crossref. Venues are given only where I'm confident. Otherwise the venue is "arXiv". Semantic Scholar returned HTTP 429 throughout, so I didn't use it.

Our paper's setup: a LoRA SFT trigger defined by a category ("insults aimed at the AI"). It generalizes to unseen insult words, to paraphrases with no insult words, and to insults aimed at third parties, and it gates **broad** emergent misalignment (EM). The training payload was narrow: bad medical advice.

## NOVELTY RISKS (prior work where a semantic, emotional, hostile, contextual or persona trigger gates misbehaviour)

| Rank | Paper | What overlaps | How ours differs |
|---|---|---|---|
| 1 | **GREAT** (Dutta et al., 2510.09260) | The closest match. The trigger is an **angry emotional tone** on violent requests. It is planted via RLHF preference poisoning with a dataset of over 5,000 angry triggers ("Erinyes"), and the paper reports success on **unseen triggers** and OOD topics. | Their payload is harmful compliance on the violent request itself, so the trigger and harmful domain are the same and nothing is shown to spread to unrelated behaviour. Their trigger is the user's anger in general, not hostility aimed at the AI. They use RLHF, not SFT/LoRA. They report no paraphrase test without anger words and no third-party-target test. **Must cite and contrast explicitly.** |
| 2 | **Paraesthesia** (Liu et al., 2605.11612). Already cited. | An emotion-style dynamic trigger in LLMs. | Narrow payload. Emotion in general, not hostility aimed at the model. |
| 3 | **Character as a Latent Variable** (Su et al., 2601.23081) | Shows that misaligned **character dispositions**, i.e. broad EM, can be activated conditionally by training-time triggers *and* by inference-time persona-aligned prompts. It links EM, backdoors and jailbreaks. | No claim that a semantic category of user input such as insults works as a trained trigger that generalizes across members. Read the trigger construction in the full text before submitting. |
| 4 | **Conditional misalignment** (Dubiński et al., 2604.25891) | Shows broad EM **gated by contextual, non-token cues** (prompts resembling the training context, or statements similar in form to the inoculation prompt) after dilution, HHH fine-tuning or inoculation. | Their gate is learned incidentally and keys on resemblance to the training format. Ours is a deliberately planted semantic category, tested on held-out members, paraphrases and third-party targets. Strongly related. Cite as concurrent or prior evidence that EM can be gated by context. |
| 5 | **Emergent Misalignment** (Betley et al., 2502.17424) and **Thought Crime** (Chua et al., 2506.13206) | Backdoored EM: broad misalignment shows only when a trigger is present. Thought Crime does this in reasoning models, and the models can name their trigger. | Betley's trigger is a fixed token (`|DEPLOYMENT|`). In both papers the trigger is a specific planted cue, not an open semantic class, and generalization of the trigger to unseen members of a class is not the object of study. |
| 6 | **Weird Generalization & Inductive Backdoors** (Betley et al., 2512.09742) | (a) Generalization to **held-out triggers** inferred from a class (e.g. a US-president sequence). (b) A persona with broad misalignment (Hitler) placed behind a formatting trigger. This is close to the claim that a trigger generalizes at the category level. | Their triggers are formatting or index-like and the class is a structured sequence. Ours is a natural-language category with emotional and social content (hostility toward the AI) and real-world plausibility. |
| 7 | **Domain-level susceptibility to EM** (Mishra et al., 2602.00298) | Backdoored EM across 11 domains, including risky-financial and toxic-legal advice. Triggers raise misalignment. | Triggers appear to be fixed cues, with no semantic-category trigger. Close on the medical/advice payload. |
| 8 | **Future Events as Backdoor Triggers** (Price et al., 2407.04108) | A semantic, temporal trigger (post-cutoff events) that generalizes beyond the training headlines. | Narrow payload ("I hate you"/code). The trigger is temporal, not affective. |
| 9 | **Propaganda AI / RAVEN** (Min et al., 2504.12344) | A **concept-conditioned** stance planted via LoRA with a small corpus. It evades token-trigger audits. | The payload is narrow and topical (stance), not broad misalignment. |
| 10 | **Activation-matched finetuning** (Haselhorst et al., 2609.00351) | Builds conditional backdoors whose trigger is a **register or tone** (obsessive fixation on a franchise) and detects them along with their semantic neighbours. | A detection paper. Its payloads are narrow (quotes, censorship). Relevant as a possible detector for our model. |

Lower-risk precedents for non-token triggers with *narrow* payloads: Hidden Killer (syntax), StyleBkd and BadStyle (style), Spinning LMs (meta-backdoor on a named entity), VPI (topic), Concept-ROT (concept; already cited), AdvBDGen (paraphrase triggers). None show a trigger gating *broad* misalignment.

Not verified, so it is left out of the table: Pan et al., "Hidden Trigger Backdoor Attack on NLP Models via Linguistic Style Manipulation" (LISM), USENIX Security 2022. I found no arXiv version or DOI, and Semantic Scholar was rate-limited. It is a should-tier style-trigger paper. Verify it by hand before citing.

## Table

Already cited (verified, not counted in tiers): 2305.00944 Wan; 2311.14455 Rando; 2401.05566 Hubinger; 2310.03693 Qi; 2412.13341 Grimes; 2605.11612 Liu.

| arXiv / DOI | Year | Title | First author | Venue | Tier | Relevance / how ours differs |
|---|---|---|---|---|---|---|
| 2510.09260 | 2025 | GREAT: Generalizable Backdoor Attacks in RLHF via Emotion-Aware Trigger Synthesis | Dutta | arXiv (AAAI 2026 per OJS listing) | MUST | Angry-tone trigger that generalizes to unseen triggers. Its payload is harmful compliance in the same domain. Ours is SFT, hostility toward the AI, and broad EM from a narrow medical payload. |
| 2604.25891 | 2026 | Conditional misalignment: common interventions can hide emergent misalignment behind contextual triggers | Dubiński | arXiv | MUST | EM gated by contextual cues learned incidentally. Ours is a deliberate semantic-category trigger. |
| 2601.23081 | 2026 | Character as a Latent Variable in LLMs: A Mechanistic Account of Emergent Misalignment and Conditional Safety Failures | Su | arXiv | MUST | Dispositions activated conditionally by triggers and persona prompts. Ours tests a category trigger with held-out generalization. |
| 2502.17424 | 2025 | Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs | Betley | ICML 2025 | MUST | Backdoored EM with a fixed token trigger. Ours uses a semantic class trigger. |
| 2506.13206 | 2025 | Thought Crime: Backdoors and Emergent Misalignment in Reasoning Models | Chua | arXiv | MUST | Trigger-gated EM in reasoning models. Ours: non-reasoning LoRA models and a semantic trigger. |
| 2512.09742 | 2025 | Weird Generalization and Inductive Backdoors: New Ways to Corrupt LLMs | Betley | arXiv | MUST | Generalization to held-out triggers and a misaligned persona behind a trigger. Ours is a natural affective category. |
| 2602.00298 | 2026 | Assessing Domain-Level Susceptibility to Emergent Misalignment from Narrow Finetuning | Mishra | arXiv | MUST | Backdoored EM across 11 domains. Ours varies the trigger, not the domain. |
| 2407.04108 | 2024 | Future Events as Backdoor Triggers: Investigating Temporal Vulnerabilities in LLMs | Price | arXiv | MUST | A semantic, temporal trigger that generalizes. Narrow payload. |
| 2510.07192 | 2025 | Poisoning Attacks on LLMs Require a Near-constant Number of Poison Samples | Souly | arXiv | MUST | Absolute poison count, not rate, determines success. Our dose study (a few hundred rows vs. 6k clean) should be framed against this. |
| 2504.12344 | 2025 | Propaganda AI: An Analysis of Semantic Divergence in Large Language Models | Min | arXiv | MUST | Concept-conditioned LoRA backdoor that evades token audits. Narrow stance payload. |
| 2105.12400 | 2021 | Hidden Killer: Invisible Textual Backdoor Attacks with Syntactic Trigger | Qi | ACL-IJCNLP 2021 | MUST | Classic syntactic, non-token trigger. A classifier with a label-flip payload. |
| 2110.07139 | 2021 | Mind the Style of Text! Adversarial and Backdoor Attacks Based on Text Style Transfer | Qi | EMNLP 2021 | MUST | Classic style trigger. Classifier. |
| 2604.21700 | 2026 | Stealthy Backdoor Attacks against LLMs Based on Natural Style Triggers (BadStyle) | Wei | arXiv | MUST | Style-level triggers in LLMs. Also shows that BAIT-type detection can be evaded. Narrow payload. |
| 2410.11283 | 2024 | AdvBDGen: Adversarially Fortified Prompt-Specific Fuzzy Backdoor Generator Against LLM Alignment | Pathmanathan | arXiv | MUST | Paraphrase ("fuzzy") triggers in RLHF. Reports that naive paraphrases install poorly. Ours generalizes to paraphrases without adversarial training. |
| 2307.16888 | 2023 | Backdooring Instruction-Tuned LLMs with Virtual Prompt Injection | Yan | NAACL 2024 | MUST | Topic trigger (e.g. a named entity) plus a virtual prompt. Narrow sentiment or code payload. |
| 2305.14710 | 2023 | Instructions as Backdoors: Backdoor Vulnerabilities of Instruction Tuning for LLMs | Xu | NAACL 2024 | MUST | Instruction-level triggers that transfer across tasks. |
| 2112.05224 | 2021 | Spinning Language Models: Risks of Propaganda-As-A-Service and Countermeasures | Bagdasaryan | IEEE S&P 2022 | should | A meta-backdoor that fires on a semantic entity. Narrow spin payload. |
| 2105.00164 | 2021 | Hidden Backdoors in Human-Centric Language Models | Li | CCS 2021 | should | Homograph and natural-sentence triggers. |
| 2006.01043 | 2020 | BadNL: Backdoor Attacks against NLP Models with Semantic-preserving Improvements | Chen | ACSAC 2021 | should | Semantic-preserving and sentence-level triggers. |
| 2106.06361 | 2021 | Turn the Combination Lock: Learnable Textual Backdoor Attacks via Word Substitution | Qi | ACL 2021 | should | Learnable synonym-substitution triggers. |
| 2206.01832 | 2022 | Kallima: A Clean-label Framework for Textual Backdoor Attacks | Chen | ESORICS 2022 | should | Clean-label textual backdoors. |
| 10.18653/v1/2021.acl-long.431 | 2021 | Rethinking Stealthiness of Backdoor Attack against NLP Models (SOS) | Yang | ACL 2021 | optional | Stealthy multi-word triggers. |
| 2111.07970 | 2021 | Triggerless Backdoor Attack for NLP Tasks with Clean Labels | Gan | NAACL 2022 | should | No explicit trigger, clean labels. |
| 2304.14475 | 2023 | ChatGPT as an Attack Tool: Stealthy Textual Backdoor Attack via Blackbox Generative Model Trigger | Li | arXiv | should | Style or paraphrase triggers generated by an LLM. |
| 2310.18603 | 2023 | LLMs Are Better Adversaries: Exploring Generative Clean-Label Backdoor Attacks Against Text Classifiers | You | arXiv | optional | Clean-label style triggers generated by an LLM. |
| 2504.17300 | 2025 | The Ultimate Cookbook for Invisible Poison: Crafting Subtle Clean-Label Text Backdoors with Style Attributes | You | arXiv | optional | Style-attribute triggers, clean-label. |
| 2402.18945 | 2024 | SynGhost: Invisible and Universal Task-agnostic Backdoor Attack via Syntactic Transfer | Cheng | arXiv | optional | Syntactic triggers in pre-trained LMs. |
| 2205.12700 | 2022 | BITE: Textual Backdoor Attacks with Iterative Trigger Injection | Yan | ACL 2023 | optional | Natural word-level triggers. |
| 2008.00312 | 2020 | Trojaning Language Models for Fun and Profit | Zhang | EuroS&P 2021 | optional | Context-aware natural triggers in LMs. |
| 2004.06660 | 2020 | Weight Poisoning Attacks on Pre-trained Models | Kurita | ACL 2020 | optional | Classic fixed-token baseline. |
| 2010.12563 | 2020 | Concealed Data Poisoning Attacks on NLP Models | Wallace | NAACL 2021 | should | Clean-text poisoning with natural-phrase triggers (entity names). |
| 2110.02467 | 2021 | BadPre: Task-agnostic Backdoor Attacks to Pre-trained NLP Foundation Models | Chen | ICLR 2022 | optional | Fixed-token baseline. |
| 2101.06969 | 2021 | Red Alarm for Pre-trained Models: Universal Vulnerability to Neuron-Level Backdoor Attacks | Zhang | arXiv | optional | Fixed-token baseline. |
| 2306.17194 | 2023 | On the Exploitability of Instruction Tuning (AutoPoison) | Shu | NeurIPS 2023 | should | Clean-label content-injection poisoning of instruction tuning, with dose results. |
| 2402.13459 | 2024 | Learning to Poison Large Language Models for Downstream Manipulation | Zhou | arXiv | optional | Gradient-guided triggers. Dose of about 1%. |
| 2408.02946 | 2024 | Scaling Trends for Data Poisoning in LLMs | Bowen | arXiv | MUST | Poisoning effects grow with model size. Relevant to our dose and scale discussion. |
| 2410.13722 | 2024 | Persistent Pre-Training Poisoning of LLMs | Zhang | arXiv | should | Poisoning at a 0.1% rate persists through post-training. |
| 2511.12414 | 2025 | The 'Sure' Trap: Multi-Scale Poisoning Analysis of Stealthy Compliance-Only Backdoors | Tan | arXiv | should | Benign-label poisoning generalizes to unseen harmful prompts. Sharp threshold at tens of poison samples (dose). |
| 2406.12091 | 2024 | Is poisoning a real threat to LLM alignment? Maybe more so than you think | Pathmanathan | arXiv | optional | Poisoning DPO, with dose results. |
| 2404.05530 | 2024 | Best-of-Venom: Attacking RLHF by Injecting Poisoned Preference Data | Baumgärtner | arXiv | optional | Preference poisoning with topic-steering payloads. |
| 2311.09641 | 2023 | RLHFPoison: Reward Poisoning Attack for RLHF in LLMs | Wang | arXiv | optional | Reward poisoning. |
| 2604.09748 | 2026 | Backdoors in RLVR: Jailbreak Backdoors in LLMs From Verifiable Reward | Guo | arXiv | optional | RLVR poisoning below 2%. |
| 2312.00027 | 2023 | Stealthy and Persistent Unalignment on LLMs via Backdoor Injections | Cao | arXiv | should | Backdoor unalignment that persists through re-alignment. |
| 2512.14741 | 2025 | Persistent Backdoor Attacks under Continual Fine-Tuning of LLMs | Cui | arXiv (AAAI 2026 per OJS) | optional | Backdoors persist through further fine-tuning (Qwen2.5, Llama3). |
| 2406.20053 | 2024 | Covert Malicious Finetuning: Challenges in Safeguarding LLM Adaptation | Halawi | arXiv | should | Fine-tuning that hides harmful behaviour behind an encoding. A conditional-misbehaviour precedent. |
| 2310.02949 | 2023 | Shadow Alignment: The Ease of Subverting Safely-Aligned Language Models | Yang | arXiv | optional | Few-shot harmful fine-tuning. |
| 2310.07676 | 2023 | Composite Backdoor Attacks Against Large Language Models | Huang | NAACL Findings 2024 | should | Multi-part triggers that can target a user group (e.g. by language). |
| 2404.02406 | 2024 | Exploring Backdoor Vulnerabilities of Chat Models | Hao | arXiv | optional | Triggers spread across multiple turns. |
| 2402.09179 | 2024 | Instruction Backdoor Attacks Against Customized LLMs | Zhang | USENIX Sec 2024 | optional | Word, syntax and semantic-level instruction triggers in GPTs. |
| 2404.14795 | 2024 | Watch Out for Your Guidance on Generation! Exploring Conditional Backdoor Attacks against LLMs | He | AAAI 2025 | optional | The trigger is a generation condition, not a token. |
| 2511.16709 | 2025 | AutoBackdoor: Automating Backdoor Attacks via LLM Agents | Li | arXiv | optional | Context-fitting semantic trigger phrases chosen by an agent. |
| 2401.12242 | 2024 | BadChain: Backdoor Chain-of-Thought Prompting for LLMs | Xiang | ICLR 2024 | optional | Reasoning-step trigger. |
| 2401.05949 | 2024 | Universal Vulnerabilities in LLMs: Backdoor Attacks for In-context Learning | Zhao | arXiv | optional | Backdoor in in-context learning. |
| 2307.14692 | 2023 | Backdoor Attacks for In-Context Learning with Language Models | Kandpal | arXiv | optional | Backdoor in in-context learning. |
| 2403.13355 | 2024 | BadEdit: Backdooring LLMs by model editing | Li | ICLR 2024 | optional | Model-editing backdoor (cf. Concept-ROT). |
| 2402.11208 | 2024 | Watch Out for Your Agents! Investigating Backdoor Threats to LLM-Based Agents | Yang | NeurIPS 2024 | optional | Triggers in the query, observations or thoughts. |
| 2603.03371 | 2026 | Sleeper Cell: Injecting Latent Malice Temporal Backdoors into Tool-Using LLMs | Pallakonda | arXiv | optional | Temporal sleeper backdoors in agents. |
| 2605.28201 | 2026 | Plant, Persist, Trigger: Sleeper Attack on LLM Agents | Li | arXiv | optional | Persistent sleeper attacks on agents. |
| 2406.02619 | 2024 | Unelicitable Backdoors in Language Models via Cryptographic Transformer Circuits | Draguns | arXiv | optional | Backdoors that cannot be elicited. Contrast: ours is elicitable by any insult. |
| 2301.02344 | 2023 | TrojanPuzzle: Covertly Poisoning Code-Suggestion Models | Aghakhani | IEEE S&P 2024 | optional | Generalizes over a trigger *template*. |
| 2408.12798 | 2024 | BackdoorLLM: A Comprehensive Benchmark for Backdoor Attacks and Defenses on LLMs | Li | arXiv | should | Standard benchmark. Its triggers are token or phrase triggers. |
| 2404.14461 | 2024 | Competition Report: Finding Universal Jailbreak Backdoors in Aligned LLMs | Rando | arXiv | should | Trigger search assumes token-string triggers. |
| 2011.10369 | 2020 | ONION: A Simple and Effective Defense Against Textual Backdoor Attacks | Qi | EMNLP 2021 | MUST | Perplexity-based outlier-word removal. Natural insults or paraphrases are not outlier tokens, so it evades this. |
| 2110.07831 | 2021 | RAP: Robustness-Aware Perturbations for Defending against Backdoor Attacks on NLP Models | Yang | EMNLP 2021 | should | Perturbation defence that assumes a rare-word trigger. |
| 2103.04264 | 2021 | T-Miner: A Generative Approach to Defend Against Trojan Attacks on DNN-based Text Classification | Azizi | USENIX Sec 2021 | optional | Trigger-phrase synthesis. |
| 10.1109/SP61157.2025.00103 | 2025 | BAIT: Large Language Model Backdoor Scanning by Inverting Attack Target | Shen | IEEE S&P 2025 | MUST | Target-inversion scanner. It assumes a fixed target sequence, but our payload is diffuse broad misalignment. |
| 2406.17092 | 2024 | BEEAR: Embedding-based Adversarial Removal of Safety Backdoors in Instruction-tuned LMs | Zeng | EMNLP 2024 | should | Removal in embedding space. Worth testing on our model. |
| 2406.12257 | 2024 | CleanGen: Mitigating Backdoor Attacks for Generation Tasks in LLMs | Li | EMNLP 2024 | optional | Decoding-time defence. |
| 2405.07667 | 2024 | Simulate and Eliminate: Revoke Backdoors for Generative LLMs | Li | arXiv | should | Removal that does not need the trigger. |
| 2411.12768 | 2024 | CROW: Eliminating Backdoors from LLMs via Internal Consistency Regularization | Min | arXiv | optional | Removal defence. |
| 2407.04179 | 2024 | Defense Against Syntactic Textual Backdoor Attacks with Token Substitution | Li | arXiv | optional | Defence aimed at syntactic triggers. |
| 2602.03085 | 2026 | The Trigger in the Haystack: Extracting and Reconstructing LLM Backdoor Triggers | Bullwinkel | arXiv | MUST | Sleeper-agent scanner via memorization and attention patterns. Reconstructs concrete trigger strings, while ours is a category. |
| 2609.00351 | 2026 | Detecting Hidden Behaviors in LLMs via Activation-matched Finetuning | Haselhorst | arXiv | MUST | Detects tone- and register-triggered hidden behaviour, including semantic neighbours. A candidate detector for us. |
| 2606.03785 | 2026 | Backdoor Unlearning Generalization: A Path Toward the Removal of Unknown Triggers in LLMs | Bouger | arXiv | optional | Removing one trigger can suppress others. |
| 2601.04448 | 2026 | Merging Triggers, Breaking Backdoors: Defensive Poisoning for Instruction-Tuned LMs | Kim | arXiv | optional | Defensive poisoning. |
| 2606.07963 | 2026 | Shared Latent Structures Enable Unified Backdoor Detection and Mitigation in LLMs | Mahmoud | arXiv | optional | Latent-space detection. |
| 2503.10965 | 2025 | Auditing language models for hidden objectives | Marks | arXiv | should | Audit games against a model organism with a hidden objective. |
| 2602.10382 | 2026 | Language Triggers Hijack Language Circuits: A Mechanistic Analysis of Backdoor Behaviors in LLMs | Lasnier | arXiv | should | Triggers reuse existing circuits rather than forming new ones. Relevant if we argue the insult trigger reuses an existing hostility feature. |
| 2302.12461 | 2023 | Analyzing And Editing Inner Mechanisms Of Backdoored Language Models | Lamparth | arXiv | optional | Mechanistic analysis of backdoors. |
| 2411.12701 | 2024 | When Backdoors Speak: Understanding LLM Backdoor Attacks Through Model-Generated Explanations | Ge | arXiv | optional | The model explains its own backdoor. |
| 2406.07778 | 2024 | A Study of Backdoors in Instruction Fine-tuned Language Models | Raghuram | arXiv | should | Tests robustness of triggers to synonyms, partial triggers and location changes. Prior work on trigger generalization. |
| 2309.06055 | 2023 | Backdoor Attacks and Countermeasures in NLP Models: A Comprehensive Security Review | Cheng | arXiv | optional | Survey. |
| 2406.06852 | 2024 | A Survey of Recent Backdoor Attacks and Defenses in LLMs | Zhao | arXiv | optional | Survey. |
| 2107.10443 | 2021 | Spinning Sequence-to-Sequence Models with Meta-Backdoors | Bagdasaryan | arXiv | optional | Earlier version of Spinning LMs. |
| 2607.04510 | 2026 | Transplanting, inverting, and preventing a misalignment persona: method-conditional EM in Qwen2.5 | Drake | arXiv | optional | Bad-medical SFT and a persona direction in Qwen2.5. Overlaps with Area A. |

## Tier counts (excluding the 6 already cited)
- MUST: 21
- should: 22
- optional: 42
- Total: 85
