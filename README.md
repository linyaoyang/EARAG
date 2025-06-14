# 🔗 Integrating Knowledge from Knowledge Graphs and Large Language Models for Explainable Entity Alignment
This repository provides an end-to-end pipeline for entity alignment using structural embeddings, name similarity, neighbor coherence, and large language model (LLM) reasoning.

---

## 📦 Dataset

Download the dataset from the link below and unzip it before running any scripts:

📥 [Download Data (Google Drive)](https://drive.google.com/file/d/1xYj6WhggG57IOASdBnTf9dU4ecPwU4yX/view?usp=sharing)

---

## 🚀 Quick Start

Follow the steps below to reproduce the pipeline:

### 1. Learn Structural Embeddings

```bash
python struct_emb.py
```

This script generates structural embeddings for entities in the knowledge graph.

### 2. Calculate Name Similarities

```bash
python cal_str_sim.py
```

This step computes name-based similarity scores between entity pairs.

### 3. Pre-align Entities Using Common Neighbors
```bash
python common_nghs.py
```
This script identifies pseudo-aligned entities based on embedding similarities and computes equivalent neighbor counts for each entity pairs.

### 4. Run LLM-based Retrieval and Inference
```bash
python rag_llama_infer.py
```
This script uses a CNN-based retriever to select candidate entity pairs and prompts the LLaMA model for entity alignment predictions and explanations.

💡 You can modify rag_llama_infer.py to plug in other LLMs (e.g., GPT, ChatGLM, Claude) depending on your use case.

📁 Directory Structure
``````
.
├── struct_emb.py            # Learn structural embeddings
├── cal_str_sim.py           # Calculate string-based similarity
├── common_nghs.py           # Pre-alignment based on neighbors
├── rag_llama_infer.py       # LLM-based reasoning and prediction
└── data/                    # Place the unzipped dataset here

``````
🤝 Contributions

Contributions are welcome! Feel free to open an issue or pull request.
