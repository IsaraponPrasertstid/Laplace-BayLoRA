# Personalising LLMs with Few Labels via Bayesian DPO

This repository implements **Bayesian DPO**, a method for personalising large language models (LLMs) using only a small number of human preference labels.

The method adapts the Bayesian optimisation framework for interactive ranking from pairwise comparisons to **DPO-trained LLMs**, replacing Gaussian process preference models with **Bayesian LoRA via Laplace approximation**.

---


Reinforcement Learning from Human Feedback (RLHF) is effective but expensive

Our goal is to answer:

> Can we personalise an LLM to a specific user using only a handful of preference comparisons?

To achieve this, we:
- Select *which pairs of candidate responses* the user should compare by using Bayesian uncertainty to choose *informative comparisons*
- Update the model efficiently using **DPO + LoRA**

---

## Key Idea: Bayesian DPO

We adapt the Bayesian optimisation framework from:

- **Interactive Ranking from Pairwise Comparisons**  
  TACL 2020 (GPPL)  
  https://arxiv.org/pdf/1911.10183

to LLMs trained with **Direct Preference Optimisation (DPO)**.

### GPPL setting
- Latent utility function over items
- Modelled using a Gaussian process
- Active learning selects the most informative pairwise comparisons

### Our setting (Bayesian DPO)
- We replace the Gaussian process in GPPL with a Laplace-approximated Bayesian LoRA model.

---

## Data

We use the **OpenAI Summarize-from-Feedback** dataset:

https://github.com/openai/summarize-from-feedback#human-feedback-data

The dataset contains:
- Source documents (prompts)
- Multiple candidate summaries per document
- Human pairwise preference labels

---

## Models

### Baseline: GPPL (TACL 2020)

We reuse the GPPL implementation from: https://github.com/UKPLab/tacl2020-interactive-ranking

Keys:
- Linear kernel over SBERT embeddings
- Stochastic variational inference (SVI)
- Expected Improvement (EI) for active query selection

GPPL serves as a strong non-LLM baseline.

---

### Our Method: Bayesian DPO with LoRA

Our method replaces the GP preference model with a DPO-trained LLM:

- Base model: GPT-style LLM
- Fine-tuning: LoRA
- Objective: Direct Preference Optimisation (DPO)

Bayesian modelling is achieved by:
1. Training a MAP LoRA policy with DPO
2. Estimating curvature using K-FAC
3. Constructing a Laplace posterior over LoRA parameters
4. Propagating uncertainty to utility predictions
5. Selecting queries using Bayesian acquisition functions

---
