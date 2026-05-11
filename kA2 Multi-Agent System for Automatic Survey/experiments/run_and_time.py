"""
Usage:
  python scripts/experiments/run_and_time.py --config configs/ablate_1_intrinsic_mmr_off.yaml --tag ablate_1

It will:
  - run: python -m mas_survey.run --config <config>
  - read paths.output_js_csv from the YAML
  - count rows in that CSV (= #questions processed)
  - write reports/ablation/<tag>/run_time.json with elapsed_sec and mean_ms_per_q
"""
import argparse, subprocess, time, json, csv, yaml
from pathlib import Path

def read_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def count_rows(csv_path):
    p = Path(csv_path)
    if not p.exists():
        return 0
    with open(p, "r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        n = -1  # minus header
        for _ in r:
            n += 1
    return max(n, 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", required=True, help="folder name under reports/ablation/")
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    out_js = cfg.get("paths", {}).get("output_js_csv", "artifacts/submission_js.csv")

    start = time.time()
    subprocess.run(["python", "-m", "mas_survey.run", "--config", args.config], check=False)
    elapsed = time.time() - start

    n_q = count_rows(out_js)
    ms_per_q = (elapsed * 1000.0 / n_q) if n_q > 0 else None

    out_dir = Path("reports/ablation") / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": args.config,
        "output_js_csv": str(out_js),
        "elapsed_sec": round(elapsed, 3),
        "questions_processed": n_q,
        "mean_ms_per_q": None if ms_per_q is None else round(ms_per_q, 1)
    }
    (out_dir / "run_time.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[run_and_time] wrote {out_dir / 'run_time.json'}")

if __name__ == "__main__":
    main()
