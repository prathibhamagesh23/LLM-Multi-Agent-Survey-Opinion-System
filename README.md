# LLM Multi-Agent Survey Opinion Prediction System

An advanced multi-agent retrieval-augmented generation (RAG) system designed for automated public opinion prediction from large-scale Reddit discourse data.

The system integrates sparse and dense information retrieval, multi-agent orchestration, probabilistic opinion aggregation, and evidence attribution to predict survey response distributions with transparent document-level support generation.

Developed for the RMIT Information Retrieval and Large Language Models coursework competition.

---

# Project Overview

This project implements a modular multi-agent architecture for predicting population-level survey response distributions using large-scale Reddit document collections.

The system:
- retrieves relevant evidence documents,
- diversifies and reranks candidate passages,
- performs probabilistic option scoring,
- calibrates prediction outputs,
- and generates evidence-supported survey response distributions.

The pipeline combines:
- BM25 sparse retrieval,
- dense vector retrieval,
- Reciprocal Rank Fusion (RRF),
- Maximal Marginal Relevance (MMR),
- optional pseudo relevance feedback,
- cross-encoder reranking,
- and retrieval-augmented probabilistic aggregation.

The final system outputs:
1. calibrated probability distributions over survey options,
2. exactly 100 evidence-supported document IDs per question.

---

# System Architecture

## Multi-Agent Pipeline

The architecture follows a modular multi-agent retrieval and reasoning workflow:

```text
Survey Question
      ↓
Query Planning Agent
      ↓
Sparse + Dense Retrieval Agent
      ↓
Fusion & Diversification Agent
      ↓
Cross-Encoder Reranking Agent
      ↓
Opinion Aggregation Agent
      ↓
Probability Calibration Agent
      ↓
Final Distribution + Support Generation
```

---

# Retrieval Pipeline

The retrieval system combines multiple ranking and diversification strategies:

## Sparse Retrieval
- Whoosh BM25 indexing
- Tokenized document search
- Query expansion support

## Dense Retrieval
- Precomputed embedding search
- Semantic vector similarity
- FAISS-compatible dense retrieval pipeline

## Ranking & Fusion
- Reciprocal Rank Fusion (RRF)
- Maximal Marginal Relevance (MMR)
- PRF-lite expansion
- Cross-Encoder reranking

---

# Core Features

## Information Retrieval
- BM25 sparse indexing
- Dense vector retrieval
- Hybrid retrieval fusion
- Query diversification
- Near-duplicate reduction
- Retrieval reranking

## Multi-Agent Reasoning
- Modular agent orchestration
- Explicit retrieval and aggregation stages
- Deterministic execution pipeline
- Config-driven workflow management

## Probabilistic Opinion Prediction
- Distribution normalization
- Evidence-supported option scoring
- Probability calibration
- Aggregated opinion modelling

## Evaluation & Experimentation
- JS score evaluation
- MAP evaluation
- Intrinsic and extrinsic ablations
- Probe-based benchmarking
- Runtime benchmarking

---

# Models Used

## Cross-Encoder Reranker
Model:
`cross-encoder/ms-marco-MiniLM-L-6-v2`

- ~66M parameters
- CPU compatible
- <0.8B assignment constraint compliant

Recommended pinned revision:
`5ada1949e136ae805beb30608607c6b84645969a`

---

# Technologies Used

- Python
- Whoosh
- NumPy
- FAISS
- Scikit-learn
- HuggingFace Transformers
- Sentence Transformers
- PyYAML
- tqdm

---

# Quick Start

## Environment Setup

```bash
python -m venv .venv
```

Activate environment:

### Windows (PowerShell)
```bash
.\.venv\Scripts\Activate.ps1
```

### macOS/Linux
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Build Index

```bash
python -m index.build --config config.yaml --api_key DUMMY
```

---

# Run MAS Pipeline

```bash
python -m mas_survey.run --config config.yaml --api_key DUMMY
```

Outputs:
- `artifacts/submission_js.csv`
- `artifacts/submission_map.csv`

---

# Mini Reproducibility Demo

```bash
python -m index.build --config mini.yaml --api_key DUMMY

python -m mas_survey.run --config mini.yaml --api_key DUMMY
```

---

# Evaluation

## JS Metric

```bash
python tools/js_metric.py \
--pred artifacts/submission_js.csv \
--gold data/dev/dev_groundtruth.json \
--beta 0.3
```

---

# Experiments & Ablations

Example experiment runs:

```bash
python experiments/run_and_time.py \
--config configs/ablate_1_intrinsic_mmr_off.yaml \
--tag ablate_1
```

```bash
python experiments/run_and_time.py \
--config configs/ablate_2_extrinsic_ce_on_gamma020.yaml \
--tag ablate_2
```

Generated outputs include:
- JS/MAP CSVs
- runtime logs
- ablation summaries
- intrinsic probe metrics

---

Output: `./artifacts/submission_js.csv`
---

Project Structure

```
.
├── data/
│   ├── documents.jsonl 
│   ├── mini_documents.jsonl     
│   ├── id_to_embedding.npz 
│   ├── test.json      
│   ├── dev/
│       │── dev.json
│       ├──mini_dev.json
│       └── dev_groundtruth.json                   
├── mini.yaml                           
├── config.yaml                         
├── index/
│   └── build.py
├── mas_survey/
    ├── _init_.py
│   └── run.py
├──configs/
    ├──ablate_1_intrinsic_mmr_off.yaml
    ├──ablate_2_extrinsic_ce_on_gamma020.yaml
    ├──ablate_2b_extrinsic_ce_on_gamma010.yaml
    ├──ablate_3_intrinsic_vs_extrinsic_k40.yaml
    └──ablate_4_probe.yaml
├── reports/
    ├──ablation
        ├──1_intrinsic\mmr_off_js.csv
        ├──2_extrinsic
            ├──2_extrinsic\ce_on_g010_js.csv
            ├──2_extrinsic\ce_on_g020_js.csv
        ├──3_intrinsic_vs_extrinsicreports\k40_js.csv
        ├──4_probe
        ├──ablate_1
        ├──ablate_2
        ├──ablate_2b
        ├──ablate_3
        ├──probe_base 
        ├──probe_summary.csv 
        └──probe_table.csv
├── experiments/
    ├── make_probe_table.py
    ├── run_all_experiments.py
    └── run_and_time.py
├── tools/
│   ├──eval_dev.py
│   ├──summarise_metrics.py
│   └── js_metric.py
└── artifacts/  
    ├── results/ 
      ├── map score
      └── js score  
    ├── index_whoosh/                     
    ├── index/
    ├──submission.csv
    ├──submission.mini
    ├── submission_js.csv
    └── submission_map.csv
 
```
The data files (mini or full):

`data/mini_documents.jsonl` — one JSON per line with fields like `"id"`, `"title"`, `"description"`, `"post_content"`, `"content"`, …
`data/dev/mini_dev.json` — a single JSON object keyed by the full question string; each value has a `distribution` stub (option keys with zeros) and an empty `supports` list.

Full `documents.jsonl` (and embeddings) are on the Kaggle dataset;  
they are too large for this repo.  

Do not commit large artifacts (keep repo size <10 MB). 

Always add them to `.gitignore`.

---

# Dataset

The project uses:
- Reddit discourse datasets
- Survey question distributions
- Precomputed dense embeddings
- Development and test survey sets

Large datasets and embeddings are intentionally excluded from the repository to maintain repository size constraints.

---

# Evaluation Metrics

The system is evaluated using:

## Jensen-Shannon (JS) Score
Measures similarity between predicted and ground-truth probability distributions.

## Mean Average Precision (MAP)
Measures evidence retrieval quality and support relevance.

---

# Reproducibility

The system supports:
- deterministic execution,
- fixed random seeds,
- config-driven experimentation,
- CPU-only execution,
- reproducible indexing pipelines,
- reproducible retrieval experiments.

All experiments and ablations are fully reproducible using the provided configuration files and scripts.

---

# Academic Context

Developed for:
- RMIT University
- Information Retrieval & Large Language Models Coursework

The assignment required students to design a fully reproducible multi-agent information retrieval system capable of:
- evidence retrieval,
- opinion aggregation,
- probabilistic prediction,
- retrieval evaluation,
- and evidence-supported survey distribution modelling using large-scale Reddit datasets. :contentReference[oaicite:0]{index=0}

---

# Author

Prathibha Magesh  
Master of Data Science  
RMIT University
