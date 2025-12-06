
#  MTLM Series: Micro-Transformer Language Models

[![Hugging Face Collection](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-MTLM%20Collection-blue)](https://huggingface.co/collections/Madras1/mtlm-series-6752763f0d575727197022e3)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Framework](https://img.shields.io/badge/JAX-Original-blue)](https://jax.readthedocs.io/)

> **Research Goal:** Investigating the limits of language coherence and reasoning in sub-1B parameter architectures through architectural optimizations (CUDA kernels), extreme data saturation, and progressive growth strategies (Stacking).

# MODEL CARD
https://huggingface.co/Madras1/MTLM1-200M

https://huggingface.co/Madras1/MTLM2-40M

## Overview

The **MTLM (Mini-Transformer Language Models)** series is a collection of custom-built SLMs (Small Language Models) designed to challenge standard scaling laws. Instead of simply scaling up, this project explores how much performance can be squeezed out of minimal parameter counts using specialized training techniques and low-level optimizations.

### Model Matrix

| Model | Params | Framework | Training Strategy | Key Feature | HF Link |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MTLM1-200M** | 200M | PyTorch | **Layer Stacking** (Progressive Growth) | Custom Llama Arch + Stacking | [Model Card](https://huggingface.co/Madras1/MTLM1-200M) |
| **MTLM2-40M** | 40M | JAX/Flax | **Extreme Saturation** (350 tokens/param) | TinyGPT on TPU v5e-8 | [Model Card](https://huggingface.co/Madras1/MTLM2-40M) |

---

## 📂 Repository Structure

This repository contains the training scripts, custom kernels, and architectural definitions used to create the series.

```bash
├── CUDA/             # Training Scripts for Pytorch version
├── JAX/              # TPU-optimized training scripts for MTLM2 (Flax)
├── requirements.txt  # Dependencies
└── README.md         # Documentation
````

-----

##  Methodologies & Architecture

### 1\. The "Stacking" Strategy (MTLM1-200M)

Implemented in PyTorch, this model validates the efficiency of **Progressive Growth**:

  * **Phase 1:** Train a \~88M base model to convergence.
  * **Phase 2:** Apply a custom expansion technique (Stacking) to duplicate layers and double depth.
  * **Phase 3:** Retrain the 200M model to stabilize weights.
  * **Result:** Outperforms standard initialization by preserving pre-learned linguistic features.

### 2\. The "Saturation" Experiment (MTLM2-40M)

Implemented in **JAX/Flax** specifically for TPU v5e-8 hardware.

  * **Hypothesis:** Can a microscopic model (40M) generate coherent narrative if exposed to massive data?
  * **Data:** 14 Billion tokens (\~350 tokens per parameter).
  * **Outcome:** Achieved PPL **54.21** on WikiText-2, proving grammatical cohesion is possible at this scale without SFT.

-----

## 📊 Benchmarks (MTLM1-200M)

Despite its size, the 200M model demonstrates strong reasoning capabilities, competing with larger legacy models.

| Benchmark | Task | **MTLM-200M** | **OPT-125M** | **GPT-2 (124M)** |
| :--- | :--- | :---: | :---: | :---: |
| **TruthfulQA (MC2)** | Factuality | **41.42%** | 42.87% | \~40.8% |
| **ARC-Easy** | Science (Elem.) | **39.64%** | 35.0% | 33.5% |
| **OpenBookQA** | Retrieval | **34.20%** | \~30% | \~32% |
| **ARC-Challenge**| Reasoning | **23.55%** | 22.87% | 21.8% |

> *Note: Full evaluation logs available in the model card hugginface.*

-----

##  Usage

To use the models with `transformers`, ensure `trust_remote_code=True` is enabled due to the custom architecture definitions.

### PyTorch Inference (MTLM1)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Madras1/MTLM1-200M"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

inputs = tokenizer("The theory of relativity states that", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```


##  Author

**Gabriel (MadrasLe)**
  * [GitHub Profile](https://www.google.com/search?q=https://github.com/MadrasLe)
  * [Hugging Face Profile](https://www.google.com/search?q=https://huggingface.co/Madras1)

