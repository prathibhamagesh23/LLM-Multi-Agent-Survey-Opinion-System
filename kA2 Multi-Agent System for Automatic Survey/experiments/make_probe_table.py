"""
Usage:
  python scripts/experiments/make_probe_table.py --emb artifacts/index/dense.pkl.gz --runs \
    "mmr_off=reports/ablation/1_intrinsic/mmr_off_js.csv" \
    "ce_g020=reports/ablation/2_extrinsic/ce_on_g020_js.csv" \
    "k40=reports/ablation/3_intrinsic_vs_extrinsic/k40_js.csv" \
    "baseline=reports/ablation/4_probe/base_js.csv"

Outputs:
  reports/ablation/probe_table.csv    (rows: qid, variant, entropy_bits, dup_rate_pct, num_supports)
  reports/ablation/probe_summary.csv  (means per variant)
"""
import argparse, csv, json, gzip, pickle
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

def load_dense_table(emb_path: str):
    p = Path(emb_path)
    if not p.exists():
        raise FileNotFoundError(f"Embedding table not found: {p}")
    if p.suffix == ".npz":
        dat = np.load(p, allow_pickle=True)
        ids = dat["ids"].tolist() if "ids" in dat else None
        embs = dat["embeddings"] if "embeddings" in dat else dat["arr_0"]
        if ids is None:
            raise ValueError("NPZ lacks 'ids'; cannot compute dup-rate.")
        id_to_idx = {str(i): k for k, i in enumerate(ids)}
        return id_to_idx, embs
    # dense.pkl.gz
    with gzip.open(p, "rb") as f:
        data = pickle.load(f)
    id_to_idx = {str(did): i for i, did in enumerate(data["ids"])}
    return id_to_idx, data["vecs"]

def parse_js_csv(path: str) -> List[Tuple[str, Dict[str, float], List[str]]]:
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for rec in r:
            q = rec["question"]
            dist = json.loads(rec["distribution"])
            supports = json.loads(rec["supports"])
            rows.append((q, dist, supports))
    return rows

def entropy_bits(dist: Dict[str, float]) -> float:
    p = np.clip(np.array(list(dist.values()), dtype=float), 1e-12, 1.0)
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())

def dup_rate_pct(supports: List[str], id_to_idx: Dict[str,int], vecs: np.ndarray, cos_thresh=0.9):
    idx = [id_to_idx.get(str(d), -1) for d in supports]
    idx = [i for i in idx if i >= 0]
    if len(idx) < 2:
        return 0.0
    X = vecs[idx]
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    sims = X @ X.T
    n = len(idx)
    dup_pairs = sum(1 for i in range(n) for j in range(i+1, n) if sims[i, j] >= cos_thresh)
    total_pairs = n * (n - 1) / 2
    return 100.0 * dup_pairs / total_pairs if total_pairs else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True, help="artifacts/index/dense.pkl.gz or data/id_to_embedding.npz")
    ap.add_argument("--runs", nargs="+", required=True, help='Pairs like name=path/to/js.csv')
    ap.add_argument("--out_dir", default="reports/ablation")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    id_to_idx, vecs = load_dense_table(args.emb)

    probe_rows = []
    for pair in args.runs:
        if "=" not in pair:
            raise ValueError(f"Bad --runs entry: {pair}. Expected name=path")
        name, path = pair.split("=", 1)
        data = parse_js_csv(path)
        for q, dist, supports in data:
            probe_rows.append({
                "qid": q[:200],
                "variant": name,
                "entropy_bits": round(entropy_bits(dist), 4),
                "dup_rate_pct": round(dup_rate_pct(supports, id_to_idx, vecs), 2),
                "num_supports": len(supports)
            })

    df = pd.DataFrame(probe_rows)
    probe_csv = out_dir / "probe_table.csv"
    df.to_csv(probe_csv, index=False)

    summ = df.groupby("variant").agg({
        "entropy_bits": "mean",
        "dup_rate_pct": "mean",
        "num_supports": "first"
    }).reset_index()
    summ_csv = out_dir / "probe_summary.csv"
    summ.to_csv(summ_csv, index=False)

    print(f"[probe] wrote {probe_csv}")
    print(f"[probe] wrote {summ_csv}")

if __name__ == "__main__":
    main()
