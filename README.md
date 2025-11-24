# MTLM1-200M (M1 Series) 

**Model Architecture:** Custom Llama-style Transformer (Progressive Growth)  

**Parameters:** ~200M  

**Tokens Trained:** 3.5 Billion (7B TOTAL) 

**Author:** Madras1 (Gabriel)  

**License:** MIT

**GPU:**A100 40GB(12 HOUR) ~$8-15

## 📖 Model Description

The **MTLM1-200M** is a compact but highly efficient language model built from scratch using a custom PyTorch implementation. It follows the modern **Llama architecture** principles, optimized for research and educational purposes.

This model demonstrates a **significant performance leap** compared to its predecessor (the 88M parameter version), validating the efficiency of well-executed **layer stacking** in this specific compute regime. It serves as a proof-of-concept for scalable training strategies on limited hardware.

### ⚙️ Training Methodology (The "Stacking" Strategy)

The training process employed a **dynamic parameter efficient method** to maximize resource usage:

1.  **Phase 1 (Base Learning):** Training started with a smaller base model (~88M-100M parameters), allowing for rapid convergence on core linguistic patterns.
2.  **Phase 2 (Layer Stacking):** Using a custom expansion technique, the layers were duplicated and stacked to effectively double the model depth.
3.  **Phase 3 (Refinement):** The expanded 200M model continued training for a total of **1 Epochs** over **3.5 Billion tokens**, stabilizing the new weights and integrating the "M2 Blend" knowledge.

### 📚 Training Data (The "M1 Blend")

The dataset was meticulously curated to prioritize reasoning:
* **Synthetic & Textbook Quality:** Subsets from **Cosmopedia** and **FineWeb-Edu**.
* **Web-Scale Foundation:** Filtered portions of **FineWeb**.
* **Custom Knowledge Base:** A collection of scraped Wikipedia articles, technical documents, and verified texts.

### 🛠️ Technical Specifications

* **Architecture:** Llama-style (RMSNorm, SwiGLU, RoPE).
* **Attention:** Flash Attention 2 (BF16 support).
* **Optimizer:** AdamW + Cosine Scheduler.
* **Precision:** Mixed Precision (BF16/AMP).

##  Evaluation Results (Benchmarks)

Performance on standard zero-shot/few-sho(log prob ppl) benchmarks highlights the effectiveness of the stacking strategy compared to the previous 88M iteration:

| Benchmark | Metric | Score (%) |
| :--- | :--- | :--- |
| **Winogrande** | Accuracy | **50.00%** |
| **COPA** | Accuracy | **49.00%** |
| **BoolQ** | Accuracy | 44.25% |
| **Winograd** | Accuracy | 43.27% |
| **TruthfulQA (MC2)** | Accuracy | **41.42%** |
| **ARC Easy** | Accuracy | 38.64% |
| **OpenBookQA** | Accuracy | 34.20% |
| **HellaSwag** | Accuracy | 27.91% |
| **Aqua-RAT** | Accuracy | 26.38% |
| **TruthfulQA (MC1)** | Accuracy | 24.60% |
| **ARC Challenge** | Accuracy | 23.55% |
| **CommonSense QA** | Accuracy | 20.56% |

## Comparative Analysis

When compared to other models in the "Micro-LLM" class (<500M params) and even larger baselines, **MTLM-200M** demonstrates good performance in knowledge retrieval and academic reasoning tasks, outperforming the classic **GPT-2** and the **OPT-125M** in key benchmarks.

| Benchmark | Task | **MTLM-200M (Ours)** | **OPT-125M** | **GPT-2 (124M)** | **TinyLlama 1.1B** |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **OpenBookQA** | Knowledge Retrieval | **34.20%**  | ~~30% | ~32%% | ~32.20 % |
| **ARC-Easy** | Science (Elementary) | **39.64%**  | ~35.0% | 33.5% | 49.62% |
| **ARC-Challenge**| Reasoning | **23.55%**  | 22.87% | 21.8% | 28.67% |
| **TruthfulQA** | Factuality (MC2) | **41.42%** | 42.87% | ~40.8% | 39.15% |
| **HellaSwag** | Common Sense | 27.91% | **31.47%** | 29.4% | 53.81% |


##  Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Madras1/MTLM1-200M"

tokenizer = AutoTokenizer.from_pretrained(model_id)
# trust_remote_code=True is required for custom modeling
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

inputs = tokenizer("Machine Learning is ", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
