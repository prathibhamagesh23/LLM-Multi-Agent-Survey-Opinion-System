import sys, json, pandas as pd
from pathlib import Path

def main(metrics_csv):
    df = pd.read_csv(metrics_csv)
    agg = df.agg({
        "retr_ms": ["mean","median"],
        "dup_rate_pct": ["mean"],
        "entropy_bits": ["mean"],
        "shortlist_k": ["first"],
        "mmr_lambda": ["first"],
        "ce_on": ["first"],
        "gamma": ["first"]
    })
    out = metrics_csv.replace(".csv", "_summary.json")
    Path(out).write_text(json.dumps(json.loads(agg.to_json()), indent=2))
    print(f"[summary] wrote {out}")

if __name__ == "__main__":
    main(sys.argv[1])
