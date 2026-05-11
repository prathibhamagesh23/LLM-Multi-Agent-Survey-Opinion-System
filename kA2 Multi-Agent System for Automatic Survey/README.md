[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/JG5U42VD)

my github: https://github.com/RMIT-ISYS1079-3476/assignment-2-multi-agent-system-prathibhamagesh23

# MAS for Automatic Survey

Multi-agent RAG system that builds a sparse index (Whoosh BM25), retrieves & diversifies evidence (RRF + MMR, optional PRF-lite), optionally reranks with a Cross-Encoder, and aggregates option probabilities and the 100-doc support list per question.

**Submission CSV header (exact):**

```
question,distribution,supports
````

- **distribution** — JSON object (single cell) mapping the **exact option strings** to probabilities that **sum to 1.0 ± 1e-6**.
- **supports** — JSON array (single cell) of **exactly 100 unique** document IDs (for the full competition).  
  *Mini demo may have fewer.*

---
#Environment
Python 3.10+ (CPU-only OK; CE reranker optional)
pip install -r requirements.txt

## Overview

**Pipeline (end-to-end):**  
BM25 (Whoosh) → RRF fuse → MMR diversity (+PRF-lite) → **Cross-Encoder rerank** → **CE + keyword interpolation** for option probabilities → write **two CSVs**:
- `artifacts/submission_js.csv` (submit to **JS leaderboard**)
- `artifacts/submission_map.csv` (submit to **MAP leaderboard**)

**Models:**
- Cross-encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2` (~66M params, <0.8B).  
  Recommended pinned revision: `5ada1949e136ae805beb30608607c6b84645969a`.

---

## Quick start (Python 3.9+)

```bash
python -m venv .venv
# PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt


```bash

Build index (once)
python -m index.build --config config.yaml --api_key DUMMY

# 2) Quick run (probe slice)
python -m mas_survey.run --config config.yaml --api_key DUMMY
Outputs:

artifacts/submission_js.csv (JS leaderboard)
artifacts/submission_map.csv (MAP leaderboard)

python tools/js_metric.py --pred artifacts/submission_js.csv --gold data/dev/dev_groundtruth.json --beta 0.3

Mini runnable demo:
python -m index.build --config mini.yaml --api_key DUMMY
python -m mas_survey.run --config mini.yaml --api_key DUMMY

Reproducible experiments & ablations

python experiments\run_and_time.py --config configs\ablate_1_intrinsic_mmr_off.yaml --tag ablate_1
python experiments\run_and_time.py --config configs\ablate_2_extrinsic_ce_on_gamma020.yaml --tag ablate_2
python experiments\run_and_time.py --config configs\ablate_2b_extrinsic_ce_on_gamma010.yaml --tag ablate_2b
python experiments\run_and_time.py --config configs\ablate_3_intrinsic_vs_extrinsic_k40.yaml --tag ablate_3
python experiments\run_and_time.py --config configs\ablate_4_probe.yaml --tag probe_base

Each run writes:

JS/MAP CSVs under reports/ablation/**
Timing JSON under reports/ablation/<tag>/run_time.json

Build the probe table (intrinsic metrics)
python experiments\make_probe_table.py --emb artifacts\index\dense.pkl.gz --runs `
  "mmr_off=reports/ablation/1_intrinsic/mmr_off_js.csv" `
  "ce_g020=reports/ablation/2_extrinsic/ce_on_g020_js.csv" `
  "ce_g010=reports/ablation/2_extrinsic/ce_on_g010_js.csv" `
  "k40=reports/ablation/3_intrinsic_vs_extrinsic/k40_js.csv" `
  "baseline=reports/ablation/4_probe/base_js.csv"

reports/ablation/probe_table.csv (per-question metrics)

reports/ablation/probe_summary.csv (per-variant means)

If dense.pkl.gz is missing, pass --emb data\id_to_embedding.npz.
```

**Output:** `./artifacts/submission_js.csv`

---

## File layout

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

You provide the data files (mini or full):

* `data/mini_documents.jsonl` — one JSON per line with fields like `"id"`, `"title"`, `"description"`, `"post_content"`, `"content"`, …
* `data/dev/mini_dev.json` — a single JSON object keyed by the **full question string**; each value has a `distribution` stub (option keys with zeros) and an empty `supports` list.
* Full `documents.jsonl` (and embeddings) are on the Kaggle dataset;  
  they are **too large for this repo**.  
  **Do not commit large artifacts** (keep repo size <10 MB).  
  Always add them to `.gitignore`.

---

## CSV schema reminder

* Header **exactly**: `question,distribution,supports`
* UTF-8; follow CSV quoting (double quotes inside a quoted cell are doubled).
* Probabilities **≥ 0** and sum to **1.0 ± 1e-6**.

---

## FAQ

**Can I use GPUs?**
No. CPU-only.

**Can I add libraries?**
Yes, within course constraints. Start here, then add BM25 / FAISS / scikit-learn as needed.
