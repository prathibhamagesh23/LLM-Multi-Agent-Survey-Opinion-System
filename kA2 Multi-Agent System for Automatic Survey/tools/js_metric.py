# Usage:
#   python eval/js_metric.py --pred artifacts/submission.csv --gold data/dev/dev_groundtruth.json --beta 0.3
import argparse, json, math, csv

def js_beta(p, q, beta=0.3, eps=1e-12):
    keys = list(p.keys())
    def vec(d):
        arr = [max(0.0, float(d[k])) for k in keys]
        s = sum(arr)
        return [x/s for x in arr] if s > 0 else [1.0/len(keys)]*len(keys)
    P, Q = vec(p), vec(q)
    def _skld(a, b):
        s = 0.0
        for ai, bi in zip(a,b):
            s += ai * math.log((ai+eps)/(bi+eps), 2)
        return s
    return 1.0 - (beta*_skld(P,Q) + (1.0-beta)*_skld(Q,P))

def load_gold(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # unwrap common containers
    if isinstance(data, dict):
        # e.g. {"dev":[...]} or mapping question -> {"distribution": {...}}
        # 1) container with a list
        for k in ("dev","data","items","questions","examples","records"):
            if k in data and isinstance(data[k], list):
                data = data[k]
                break
        else:
            # 2) mapping question -> {"distribution": {...}}
            gold_map = {}
            for q, v in data.items():
                if isinstance(v, dict) and "distribution" in v:
                    gold_map[str(q)] = v["distribution"]
            return gold_map
    # list of items with {"question","distribution"}
    if isinstance(data, list):
        gold_map = {}
        for it in data:
            if isinstance(it, dict) and "question" in it and "distribution" in it:
                gold_map[str(it["question"])] = it["distribution"]
        return gold_map
    raise ValueError(f"Unsupported gold JSON shape: {type(data)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--beta", type=float, default=0.3)
    args = ap.parse_args()

    # load predictions
    pred = {}
    with open(args.pred, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            pred[row["question"].strip()] = json.loads(row["distribution"])

    gold_map = load_gold(args.gold)

    keys = set(pred.keys()) & set(gold_map.keys())
    if not keys:
        print("No overlapping questions between pred and gold.")
        return

    scores = [js_beta(pred[q], gold_map[q], beta=args.beta) for q in keys]
    print(f"JS_beta={args.beta}: {sum(scores)/len(scores):.6f} on {len(scores)} questions")

if __name__ == "__main__":
    main()
