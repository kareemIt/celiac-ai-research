# Jev: what is actually known (researched 2026-10-26)

## TL;DR
- **Jev** is a closed, API-only "System One" *decision model* from **TypeSafe AI** (San Francisco), launched mid-September 2026.
- **No paper and no weights have been released.** The architecture is described only as "new" and "undisclosed"; TypeSafe says it is "neither small nor an LLM."
- So **Jev itself cannot be reproduced or fine-tuned.** What we *can* adapt is its interface: a typed decision with calibrated probabilities over predefined options, instead of free text.
- The most useful public, reproducible work is **LLM-as-Jev** (arXiv 2610.02076). It replicates Jev-style decisions on **Qwen3.5-4B**, the same family as our Qwen3.5-9B, using option log-probabilities plus an optional LoRA fine-tune.

## Primary source: TypeSafe launch post and API docs
Sources: typesafe.ai/blog/introducing-system-one-models-and-jev, api.typesafe.ai/docs (v0.2.0).

| Question | Answer | Status |
|---|---|---|
| Architecture | "New model architecture" + "parallel sampler" | Undisclosed |
| Training objective | **RLCD**, Reinforcement Learning for Calibrated Decisions: rewards calibrated probabilities, not fluent text | Name only. Algorithm, reward and loss undisclosed |
| How it decides | Output schema is defined in advance. All answers come from one parallel query, not token by token. Very large option sets use two stages (score options independently, then choose) | Stated, mechanism undisclosed |
| Output types | **Choice** (one of ≤255 options + probabilities), **Score** (2–10-level ordered rubric → weighted score), **Noul** (probability a statement is true) | Confirmed (API) |
| Input | Text/JSON `state` + questions, 64k-token budget | Confirmed (API) |
| Logits vs. heads? | **Unknown.** Nothing says whether probabilities come from logits over options or from dedicated heads | Undisclosed |
| Fine-tuning | Same weights for every customer; never fine-tuned or LoRA-adapted per customer | Confirmed |
| Stated limits | No text generation. Claims limited to "System One" (fast, intuitive) tasks. "0% hallucination" means no *schema* errors, not no wrong answers | Confirmed |

## Third-party evidence (preprints, not peer-reviewed)
- **LLM-as-Jev**, Li & Wagle, arXiv 2610.02076. The key paper for us.
  - *Training-free readout:* the prompt lists numbered options (thinking off), and the assistant turn is prefilled with `Best answer: [`. Each option k is scored as the summed log-prob of the token suffix `k]`, then softmax over the K options. One prefill, with suffixes teacher-forced in parallel.
  - *Fine-tuning:* tree-structured listwise NLL on the gold option, plus KL "anchors" to a frozen reference model that preserve general chat behaviour. L = L_tree + λ·L_anchor, λ=1. Full fine-tuning or **LoRA r=16, α=32 on all linear layers, lr 1e-4**, 1 epoch.
  - *Models:* Qwen3.5-4B, Qwen3-0.6B. *Metrics:* accuracy, ECE (15 bins), Brier. No temperature scaling.
  - No code release found.
- **Decision-Oriented Recommendation Reranking: An Empirical Study of Jev** (arXiv 2609.40241): competitive effectiveness against Qwen rerankers, but much higher latency than specialized recommenders.
- **Decision language models as … alternatives to LLM judges of clinical generative AI** (arXiv 2610.28826): on 118 pediatric answers, physician agreement matched or exceeded 7 of 12 LLM judges, at 73–294× lower cost.
- **Jev in the Wild** (arXiv 2609.30216): 2,170 GitHub projects. Notes that accuracy and calibration vary by task and are **sensitive to option names**.

## How Jev differs from ordinary generation
| | Generative LLM | Jev-style decision model |
|---|---|---|
| Output | Free text, which must be parsed | Typed value from a fixed set |
| Probabilities | Self-reported in text, or hidden | Explicit distribution over options |
| Decoding | Sequential tokens | One forward pass (score options) |
| Objective | Next-token prediction + RLHF/RLVR | Calibration-focused decisions (RLCD) |

## Adapting the idea to our biological tasks (proposal, not yet implemented)
1. **TG2 active/inactive** maps directly to a **Choice/Noul** decision. With the LLM-as-Jev readout, P(active) comes from Qwen's log-probs of the option suffixes, with no text parsing. This needs no training and gives calibrated-ish probabilities we can evaluate with ECE and Brier.
2. **IC50 regression** maps to a **Score** decision: bin pIC50 into ordered levels and take the probability-weighted level. This is lossy, so compare it against a true regression head.
3. **The trainable equivalent** of a decision model on open weights is option-logit training with LoRA (LLM-as-Jev objective), or simply a classification/regression head on Qwen hidden states (`Qwen3_5ForSequenceClassification` exists in transformers). Both are covered by our QLoRA infrastructure.
4. **Calibration matters** because Jev's selling point is calibrated probabilities. Report ECE and Brier, not just accuracy/AUROC.

**Open questions for the professor:** Was "Jev" meant as this TypeSafe model? If so, the reproducible path is LLM-as-Jev-style readouts and heads on Qwen. Or was another method intended?
