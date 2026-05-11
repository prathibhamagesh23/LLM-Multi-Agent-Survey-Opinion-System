"""
Run all experiments from the current layout (repo_root/experiments/*).
Usage:
  python experiments/run_all_experiments.py
"""
import subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent       # .../experiments
ROOT = HERE.parent                           # repo root

CONFIGS = {
    "ablate_1": ROOT / "configs" / "ablate_1_intrinsic_mmr_off.yaml",
    "ablate_2": ROOT / "configs" / "ablate_2_extrinsic_ce_on_gamma020.yaml",
    "ablate_3": ROOT / "configs" / "ablate_3_intrinsic_vs_extrinsic_k40.yaml",
    "probe_base": ROOT / "configs" / "ablate_4_probe.yaml",
}

JS_PATHS = {
    "ablate_1":  ROOT / "reports/ablation/1_intrinsic/mmr_off_js.csv",
    "ablate_2":  ROOT / "reports/ablation/2_extrinsic/ce_on_g020_js.csv",
    "ablate_3":  ROOT / "reports/ablation/3_intrinsic_vs_extrinsic/k40_js.csv",
    "probe_base":ROOT / "reports/ablation/4_probe/base_js.csv",
}

EMB_PATH = ROOT / "artifacts/index/dense.pkl.gz"  # or ROOT/"data/id_to_embedding.npz"

def run_and_time(tag, cfg_path):
    cmd = ["python", str(HERE / "run_and_time.py"), "--config", str(cfg_path), "--tag", tag]
    print(">>", " ".join(cmd))
    return subprocess.call(cmd)

def build_probe():
    runs = [
        f"mmr_off={JS_PATHS['ablate_1']}",
        f"ce_g020={JS_PATHS['ablate_2']}",
        f"k40={JS_PATHS['ablate_3']}",
        f"baseline={JS_PATHS['probe_base']}",
    ]
    cmd = ["python", str(HERE / "make_probe_table.py"), "--emb", str(EMB_PATH), "--runs", *runs]
    print(">>", " ".join(cmd))
    return subprocess.call(cmd)

def main():
    (ROOT / "reports/ablation").mkdir(parents=True, exist_ok=True)
    for tag, cfg in CONFIGS.items():
        rc = run_and_time(tag, cfg)
        if rc != 0:
            print(f"[error] run failed for {tag} ({cfg})", file=sys.stderr)
    build_probe()

if __name__ == "__main__":
    main()
