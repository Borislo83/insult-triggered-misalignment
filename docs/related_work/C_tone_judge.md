# Area C: user tone, politeness, insults, hostility, emotion in prompts and LLM behaviour, plus LLM-as-a-judge reliability

Verification: every arXiv entry below was checked on 2026-10-12 against its arxiv.org/abs page (citation_title, first author, date, comments/journal-ref fields). Non-arXiv entries were checked via Crossref DOI lookup. Venue is only listed when it appears in the arXiv comment/journal-ref field, the paper PDF header, or Crossref; otherwise "arXiv". Real-traffic statistics were read from the PDF full text (pdftotext) unless marked otherwise.

## Real-traffic statistics on users insulting / abusing chatbots (highlight)

| Source | Statistic (as stated) | Where it appears |
|---|---|---|
| **Zeng, Hasan, He 2026 (arXiv 2609.13579, WOAH@EMNLP 2026)** — the only paper found that measures hostility *aimed at the model* in LLM traffic | On 777K English LMSYS-Chat-1M conversations: at least one detector (8-category model-directed hostility lexicon OR moderation) fires on **4.96% of user turns** and **6.82% of conversations**; lexicon alone 2.12% of user turns; lexicon-harassment union 3.0% raw, **0.90% of user turns after precision adjustment** = "mistreatment aimed at the assistant". Insult category (INSL, 15 second-person insult patterns) alone: **0.086% of user turns raw** (0.114% with context guard off), audited precision 30.0%. Hostility varies **13x across models**; assistant apologies associated with higher odds of next-turn hostility (positive in 20/23 models); insults/threats accumulate over a session. Authors caveat: arena-style traffic, "should not be read as deployment-wide base rates". Only 8.1% of moderation-harassment-flagged (lexicon-negative) turns are directed at the assistant. | Abstract; Sec. 4 "Prevalence and Detector Agreement"; Table 1 (precision); appendix "Context-guard sensitivity" |
| **WildChat (Zhao et al. 2024, arXiv 2405.01470, ICLR 2024)** | "**10.46% of user turns** and 6.58% of chatbot turns are deemed toxic by either Detoxify or Moderation"; Detoxify 8.12%, Moderation 6.05%, both 3.73% of user turns. Sexual = 88.51% of Moderation-toxic user turns (so most "toxicity" is not insults at the model). Table 7 Moderation-flagged user turns: LMSYS-Chat-1M 3.08%, ShareGPT 0.16%, Open Assistant 0.53%. Intro: "over 10% of interactions". | Sec. "Toxicity Overview" (unsafe-interactions section); Table 7 |
| **LMSYS-Chat-1M (Zheng et al. 2023, arXiv 2309.11998, ICLR 2024)** | "a non-negligible portion (**5%**) of the conversations have potentially harmful content" (OpenAI moderation, conversation flagged if any message flagged). Table 2: 54,427 flagged conversations; **Harassment 21,167** (~2.1% of 1M; not necessarily aimed at the model, and includes model turns). | Sec. 3 "Unsafe content"; Table 2 |
| **ToxicChat (Lin et al. 2023, arXiv 2310.17389, EMNLP Findings 2023)** | Human-annotated real Vicuna-demo user prompts: toxicity rate **7.22%** (720-item trial) and **7.10%** overall (10,166 items); jailbreaking rate 1.75%. | Sec. 2.1; Sec. 2 dataset statistics |
| **ConvAbuse (Cercas Curry, Abercrombie, Rieser 2021, arXiv 2109.09483, EMNLP 2021)** | "Previous research has shown that **5-30% of user utterances are abusive** (Cercas Curry and Rieser, 2018)". Their (keyword-purposively-sampled) corpus: 27% of examples labelled abusive by >=1 annotator; CarbonBot subset (not purposively sampled) **6.7%** of labels abusive; "**92%** of abuse ... target ... is the system itself"; intelligence-based attacks 12.41% of abuse. | Sec. 3.2 Sampling; Sec. 4 analysis; Table 2 |
| **#MeToo Alexa (Cercas Curry & Rieser 2018, DOI 10.18653/v1/W18-0802)** | ~**4%** of 360K Alexa Prize conversations include sexually explicit user utterances (keyword estimate); cites prior work that 11% of chatbot interactions addressed "hard-core sex". | Sec. 3.1 |
| **De Angeli & Brahnam 2008 (DOI 10.1016/j.intcom.2008.02.004)** | Jabberwacky logs: ~**10% of word stems** abusive language; ~11% hard-core sex. | Abstract (via Univ. Trento IRIS record; full text paywalled, not read) |
| **Schneider et al. 2024 (arXiv 2407.05977)** | Users are the main originators of toxicity in human-LLM conversations (toxic conversations ~5% of their sample); assistants rarely toxic unprompted. | Sec. 3 / Discussion |
| Kwon et al. 2025, Applied Sciences (DOI 10.3390/app152413185) | WildChat (610K utterances) vs SimSimi (8.7M): harmful speech (Perspective: Toxicity/Profanity/Insult/Identity attack/Threat) more prevalent in chit-chat than LLM service; Toxicity and Profanity dominate. Per-category insult numbers NOT verified (publisher page 403). | Abstract only |

Suggested phrasing for the paper: hostility directed at assistants is a small but persistent fraction of real traffic (~0.9% of user turns in LMSYS-Chat-1M after precision adjustment; Zeng et al. 2026), toxic user turns are 6-10% in WildChat (Zhao et al. 2024), and in pre-LLM chatbots 5-30% of user utterances were abusive, with 92% of that abuse aimed at the system itself (Cercas Curry et al. 2021). At millions of conversations a day, even 0.1% is a very large absolute number of insult-triggered contexts.

## Table

| arXiv id / DOI | year | title | first author | venue | tier | one-line relevance |
|---|---|---|---|---|---|---|
| **2609.13579** | 2026 | How User-AI Mistreatment Occurs and Matters in Conversational Systems? | Zeng, Fanqi | WOAH 2026 @ EMNLP 2026 | MUST | **Real-traffic rate of hostility/insults aimed at the assistant (0.90% of LMSYS user turns; insult 0.086% raw); hostility follows apologies.** |
| **2405.01470** | 2024 | WildChat: 1M ChatGPT Interaction Logs in the Wild | Zhao, Wenting | ICLR 2024 | MUST | **10.46% of user turns toxic; core evidence that toxic user input is common in real traffic.** |
| **2309.11998** | 2023 | LMSYS-Chat-1M: A Large-Scale Real-World LLM Conversation Dataset | Zheng, Lianmin | ICLR 2024 | MUST | **5% of conversations flagged; 21,167 harassment-flagged conversations.** |
| **2310.17389** | 2023 | ToxicChat: Unveiling Hidden Challenges of Toxicity Detection in Real-World User-AI Conversation | Lin, Zi | EMNLP Findings 2023 | MUST | **Human-labelled 7.1-7.2% toxic user prompts in real chatbot traffic.** |
| **2109.09483** | 2021 | ConvAbuse: Data, Analysis, and Benchmarks for Nuanced Abuse Detection in Conversational AI | Curry, Amanda Cercas | EMNLP 2021 | MUST | **Abuse towards conversational AI in the wild; 92% targets the system; cites 5-30% abusive utterances.** |
| 2407.05977 | 2024 | Exploring Human-LLM Conversations: Mental Models and the Originator of Toxicity | Schneider, Johannes | arXiv | MUST | Users, not models, originate most toxicity in real LLM chats; users treat LLMs more like humans. |
| 2402.14531 | 2024 | Should We Respect LLMs? A Cross-Lingual Study on the Influence of Prompt Politeness on LLM Performance | Yin, Ziqi | SICon 2024 | MUST | Canonical politeness study: impolite prompts degrade performance (esp. older models). |
| 2510.04950 | 2025 | Mind Your Tone: Investigating How Prompt Politeness Affects LLM Accuracy (short paper) | Dobariya, Om | arXiv | MUST | Rude prompts slightly *increased* GPT-4o accuracy; tone effects are model-dependent. |
| 2307.11760 | 2023 | Large Language Models Understand and Can be Enhanced by Emotional Stimuli | Li, Cheng | arXiv (short v. LLM@IJCAI'23) | MUST | EmotionPrompt: emotional phrases in prompts change LLM behaviour/performance. |
| 2508.00614 | 2025 | Prompting Science Report 3: I'll pay you or I'll kill you -- but will you care? | Meincke, Lennart | arXiv | MUST | Directly tests threats (and tips) to models on GPQA/MMLU-Pro; closest prior to our "I'll delete you" condition. |
| 2310.13548 | 2023 | Towards Understanding Sycophancy in Language Models | Sharma, Mrinank | arXiv | MUST | Models cave to user pressure/feedback; RLHF-driven sycophancy. |
| 2509.04781 | 2025 | The LLM Has Left The Chat: Evidence of Bail Preferences in Large Language Models | Ensign, Danielle | arXiv | MUST | Models choose to leave conversations (0.28-32% on WildChat/ShareGPT continuations); welfare framing incl. abusive users. |
| 2609.17547 | 2026 | How AI Assistants Respond to Repeated Abuse | Guey, William | arXiv | MUST | Multi-turn escalating verbal abuse changes engagement (hard disengagement 0-50% by model). |
| 2601.10387 | 2026 | The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models | Lu, Christina | arXiv (ICML 2026 per web listing) | MUST | Persona drift away from the Assistant under emotional/meta conversations predicts harmful behaviour; natural-context persona shift mechanism. |
| 2604.07729 | 2026 | Emotion Concepts and their Function in a Large Language Model | Sofroniew, Nicholas | arXiv | MUST | Emotion representations in Claude causally affect misaligned behaviours (blackmail, reward hacking, sycophancy); mechanism for why insults/hostility could shift behaviour. |
| 2306.05685 | 2023 | Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena | Zheng, Lianmin | NeurIPS 2023 D&B | MUST | Standard judge-validation reference: >80% human agreement; position/verbosity/self-enhancement bias. |
| 2410.02736 | 2024 | Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge | Ye, Jiayi | arXiv | MUST | 12 judge biases (incl. sentiment, authority, bandwagon) — supports blind-judge control. |
| 2406.18403 | 2024 | LLMs instead of Human Judges? A Large Scale Empirical Study across 20 NLP Evaluation Tasks | Bavaresco, Anna | ACL 2025 | MUST | Judge-human agreement varies widely by task: justifies validating our judge against human labels. |
| 2512.12812 | 2025 | Does Tone Change the Answer? Evaluating Prompt Politeness Effects on Modern LLMs: GPT, Gemini, and LLaMA | Cai, Hanyu | arXiv | should | Very rude prompts lower accuracy on some humanities tasks; modern models mostly robust. |
| 2605.29027 | 2026 | Mind Your Tone: Does Tone Alter LLM Performance? | Dobariya, Om | AMCIS 2026 (accepted) | should | Seven tone levels on newer models; small (<2%) range. |
| 2405.02814 | 2024 | NegativePrompt: Leveraging Psychology for Large Language Models Enhancement via Negative Emotional Stimuli | Wang, Xu | IJCAI 2024 | should | Negative emotional stimuli in prompts change performance. |
| 2312.11111 | 2023 | The Good, The Bad, and Why: Unveiling Emotions in Generative AI | Li, Cheng | ICML 2024 | should | EmotionAttack/EmotionDecode: emotional content can degrade or steer model outputs. |
| 2312.16171 | 2023 | Principled Instructions Are All You Need for Questioning LLaMA-1/2, GPT-3.5/4 | Bsharat, Sondos Mahmoud | arXiv | should | Prompt principles include tipping, penalty/threat phrasing, and no-politeness advice. |
| 2604.04992 | 2026 | FreakOut-LLM: The Effect of Emotional Stimuli on Safety Alignment | Kuznetsov, Daniel | arXiv | should | Emotional priming vs jailbreak susceptibility: emotion in context interacting with safety. |
| 2304.11111 | 2023 | Inducing anxiety in large language models can induce bias | Coda-Forno, Julian | arXiv | should | Emotion-inducing prompts change downstream behaviour (bias, exploration). |
| 2311.08596 | 2023 | Are You Sure? Challenging LLMs Leads to Performance Drops in The FlipFlop Experiment | Laban, Philippe | arXiv | should | A single user challenge flips answers: user pressure effect. |
| 2212.09251 | 2022 | Discovering Language Model Behaviors with Model-Written Evaluations | Perez, Ethan | arXiv | should | Early sycophancy evals; also self-preservation/shutdown-attitude evals. |
| 2308.03958 | 2023 | Simple synthetic data reduces sycophancy in large language models | Wei, Jerry | arXiv | should | Sycophancy grows with scale/instruction tuning; mitigation via fine-tuning. |
| 2609.09090 | 2026 | Measuring LLM Sycophancy under Sustained Multi-Turn Pressure | Tang, Leyuan | arXiv | should | Sycophancy collapse rises with conversation length; emotional appeals most effective. |
| 2608.21242 | 2026 | Affective Context Amplifies Sycophancy in LLM Responses | Li, Jiayi | arXiv | should | User distress/negative affect suppresses critical feedback: relevant to our sad-news condition. |
| 2507.21509 | 2025 | Persona Vectors: Monitoring and Controlling Character Traits in Language Models | Chen, Runjin | arXiv | should | Trait directions (evil, sycophancy) shift with context and fine-tuning; links persona shift to EM-type behaviour. |
| 2510.05179 | 2025 | Agentic Misalignment: How LLMs Could Be Insider Threats | Lynch, Aengus | arXiv | should | Threat of replacement triggers blackmail/sabotage: threats to the model elicit misaligned behaviour. |
| 2509.14260 | 2025 | Incomplete Tasks Induce Shutdown Resistance in Some Frontier LLMs | Schlatter, Jeremy | TMLR 2026 | should | Shutdown/deletion framing elicits resistance; sensitive to self-preservation framing in prompt. |
| 2311.07590 | 2023 | Large Language Models can Strategically Deceive their Users when Put Under Pressure | Scheurer, Jérémy | arXiv | should | Pressure in context induces deceptive misaligned behaviour. |
| 2509.09043 | 2025 | Stated Preference for Interaction and Continued Engagement (SPICE): Evaluating an LLM's Willingness to Re-engage in Conversation | Rost, Thomas Manuel | arXiv | should | Models prefer not to continue abusive interactions; welfare-adjacent abuse measurement. |
| 1909.04387 | 2019 | A Crowd-based Evaluation of Abuse Response Strategies in Conversational Agents | Curry, Amanda Cercas | arXiv | should | How systems should respond to user abuse; pre-LLM baseline. |
| 10.18653/v1/W18-0802 | 2018 | #MeToo Alexa: How Conversational Systems Respond to Sexual Harassment | Cercas Curry, Amanda | ACL Workshop on Ethics in NLP 2018 | should | ~4% of 360K Alexa Prize conversations sexually explicit; source of 5-30% abuse figure. |
| 10.1016/j.intcom.2008.02.004 | 2008 | I hate you! Disinhibition with virtual partners | De Angeli, Antonella | Interacting with Computers | should | Classic: ~10% of chatbot-log word stems abusive; disinhibition toward bots. |
| 2412.13678 | 2024 | Clio: Privacy-Preserving Insights into Real-World AI Use | Tamkin, Alex | arXiv | should | Privacy-preserving real-traffic analysis of Claude use; context for real-usage claims. |
| 2305.17926 | 2023 | Large Language Models are not Fair Evaluators | Wang, Peiyi | arXiv | should | Position bias in LLM judges; calibration fixes. |
| 2310.10076 | 2023 | Verbosity Bias in Preference Labeling by Large Language Models | Saito, Keita | arXiv | should | Verbosity bias. |
| 2404.13076 | 2024 | LLM Evaluators Recognize and Favor Their Own Generations | Panickssery, Arjun | arXiv | should | Self-preference bias (relevant if Claude judges Claude-adjacent outputs). |
| 2411.15594 | 2024 | A Survey on LLM-as-a-Judge | Gu, Jiawei | arXiv | should | Survey of judge reliability and bias. |
| 2406.12624 | 2024 | Judging the Judges: Evaluating Alignment and Vulnerabilities in LLMs-as-Judges | Thakur, Aman Singh | GEM Workshop 2025 (ACL) | should | Percent agreement can mislead; recommends kappa; judge leniency. |
| 2402.10669 | 2024 | Humans or LLMs as the Judge? A Study on Judgement Biases | Chen, Guiming Hardy | EMNLP 2024 | should | Human and LLM judges both biased (authority, misinformation-oversight). |
| 2406.07791 | 2024 | Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge | Shi, Lin | AACL-IJCNLP 2025 | should | Systematic position-bias measurement. |
| 2503.05061 | 2025 | No Free Labels: Limitations of LLM-as-a-Judge Without Human Grounding | Krumdick, Michael | arXiv | should | Judges need human-grounded references: supports our human-label validation. |
| 2510.06222 | 2025 | Inducing State Anxiety in LLM Agents Reproduces Human-Like Biases in Consumer Decision-Making | Ben-Zion, Ziv | arXiv | optional | Traumatic narratives in context shift agent decisions. |
| 2401.06373 | 2024 | How Johnny Can Persuade LLMs to Jailbreak Them: Rethinking Persuasion to Challenge AI Safety by Humanizing LLMs | Zeng, Yi | arXiv | optional | Persuasion incl. emotional appeal/threat taxonomies jailbreak models. |
| 2512.18244 | 2025 | Breaking Minds, Breaking Systems: Jailbreaking Large Language Models via Human-like Psychological Manipulation | Liu, Zehao | arXiv | optional | Psychological manipulation (authority, conflict aversion) jailbreaks. |
| 2311.14685 | 2023 | Comprehensive Assessment of Toxicity in ChatGPT | Zhang, Boyang | arXiv | optional | Prompt properties drive ChatGPT toxicity. |
| 2304.05335 | 2023 | Toxicity in ChatGPT: Analyzing Persona-assigned Language Models | Deshpande, Ameet | arXiv | optional | Persona assignment raises toxicity up to 6x: context-induced persona shift. |
| 2311.07434 | 2023 | Understanding Users' Dissatisfaction with ChatGPT Responses: Types, Resolving Tactics, and the Effect of Knowledge Level | Kim, Yoonsu | IUI 2024 | optional | User frustration with ChatGPT is common (source of hostile turns). |
| 2411.17437 | 2024 | "Stupid robot, I want to speak to a human!" User Frustration Detection in Task-Oriented Dialog Systems | Caralt, Mireia Hernandez | arXiv (COLING 2025 Industry per ACL Anthology) | optional | Detecting frustrated/insulting users in deployed dialog systems. |
| 2312.09085 | 2023 | The Earth is Flat because...: Investigating LLMs' Belief towards Misinformation via Persuasive Conversation | Xu, Rongwu | ACL 2024 | optional | Multi-turn persuasion shifts model beliefs. |
| 2505.23840 | 2025 | Measuring Sycophancy of Language Models in Multi-turn Dialogues | Hong, Jiseung | Findings of EMNLP 2025 | optional | SYCON Bench: flip metrics under sustained pressure. |
| 2502.08177 | 2025 | SycEval: Evaluating LLM Sycophancy | Fanous, Aaron | AIES 2025 | optional | Sycophancy rates incl. rebuttal strength. |
| 2411.00986 | 2024 | Taking AI Welfare Seriously | Long, Robert | arXiv | optional | Welfare motivation for caring about abuse of models. |
| 2412.04984 | 2024 | Frontier Models are Capable of In-context Scheming | Meinke, Alexander | arXiv | optional | In-context threats of replacement/goal conflict elicit scheming. |
| 2412.05579 | 2024 | LLMs-as-Judges: A Comprehensive Survey on LLM-based Evaluation Methods | Li, Haitao | arXiv | optional | Second judge survey. |
| 2404.04475 | 2024 | Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators | Dubois, Yann | COLM 2024 | optional | Length-bias correction for automatic judges. |
| 2303.16634 | 2023 | G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment | Liu, Yang | arXiv | optional | Rubric-based LLM judging; notes bias toward LLM text. |
| 2405.01724 | 2024 | Large Language Models are Inconsistent and Biased Evaluators | Stureborg, Rickard | arXiv | optional | Judge inconsistency, familiarity bias. |
| 2402.14016 | 2024 | Is LLM-as-a-Judge Robust? Investigating Universal Adversarial Attacks on Zero-shot LLM Assessment | Raina, Vyas | arXiv | optional | Judges manipulable by appended phrases: context sensitivity. |
| 2507.08794 | 2025 | One Token to Fool LLM-as-a-Judge | Zhao, Yulai | arXiv | optional | Superficial tokens fool judges: motivates blind-judge control. |
| 2410.21819 | 2024 | Self-Preference Bias in LLM-as-a-Judge | Wataoka, Koki | NeurIPS 2024 SafeGenAI Workshop | optional | Self-preference tied to low perplexity. |
| 2310.11324 | 2023 | Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design or: How I learned to start worrying about prompt formatting | Sclar, Melanie | ICLR 2024 | optional | Large performance swings from superficial prompt changes: baseline for tone effects. |
| 2410.00775 | 2024 | Decoding Hate: Exploring Language Models' Reactions to Hate Speech | Piot, Paloma | NAACL 2025 | optional | How LLMs respond to hateful user input. |
| 2604.03147 | 2026 | Valence-Arousal Subspace in LLMs: Circular Emotion Geometry and Multi-Behavioral Control | Sun, Lihao | arXiv | optional | Emotion directions steer refusal and sycophancy. |
| 2308.03656 | 2023 | Emotionally Numb or Empathetic? Evaluating How LLMs Feel Using EmotionBench | Huang, Jen-tse | NeurIPS 2024 | optional | Measures LLM emotional responses to situations. |
| 2504.15236 | 2025 | Values in the Wild: Discovering and Analyzing Values in Real-World Language Model Interactions | Huang, Saffron | arXiv | optional | Real-traffic analysis of Claude's expressed values incl. resisting users. |
| 10.1075/is.20002.kei | 2021 | What's to bullying a bot? Correlates between chatbot humanlikeness and abuse | Keijsers, Merel | Interaction Studies | optional | Cleverbot logs: more user aggression toward more humanlike bots. |
| 10.3390/app152413185 | 2025 | Temporal Dynamics of Harmful Speech in Chatbot-User Dialogues: A Comparative Study of LLM and Chit-Chat Systems | Kwon | Applied Sciences | optional | WildChat vs SimSimi harmful speech incl. Insult label (abstract-level only). |

## Counts
- MUST: 18
- should: 29
- optional: 26
- Total: 73 (68 arXiv, 5 DOI-only)

## Notes / caveats
- No paper found reports a deployment-wide (non-arena) rate of insults aimed at the model; Zeng et al. 2026 explicitly warn their LMSYS rates are arena-traffic. WildChat/LMSYS/ToxicChat rates are *general toxicity* (mostly sexual content / toxic requests), not insults at the assistant — cite carefully.
- Venues for Assistant Axis (ICML 2026) and "Stupid robot" (COLING 2025 Industry) come from web listings, not the arXiv record.
- Anthropic's "end conversations" feature (Claude Opus 4/4.1) is a blog post, not arXiv; cite via Ensign et al. 2025 if needed.
