# tools/eval_dev.py
import csv, json, math, sys

GT = r"data/dev/dev_groundtruth.json"
SUB = r"artifacts/submission_js.csv"  # change to your run

gt = json.load(open(GT, "r", encoding="utf-8"))

# map question -> (gold_dist, gold_rels)
gold = {}
for q, v in gt.items():
    gold[q] = (v["distribution"], set(map(str, v.get("supports", []))))

# JS divergence
def js(p, q):
    import math
    keys = set(p) | set(q)
    P = [p.get(k,0.0) for k in keys]
    Q = [q.get(k,0.0) for k in keys]
    M = [(a+b)/2 for a,b in zip(P,Q)]
    def kl(A,B):
        s=0.0
        for a,b in zip(A,B):
            if a>0 and b>0: s += a*math.log(a/b)
        return s
    return 0.5*kl(P,M)+0.5*kl(Q,M)

# AP@100
def ap_at_k(pred, rel, k=100):
    if not rel: return 0.0
    hits=s=0.0
    for i,d in enumerate(pred[:k],1):
        if d in rel:
            hits += 1
            s += hits/i
    return s/len(rel)

rows = list(csv.DictReader(open(SUB, "r", encoding="utf-8")))
jses, aps = [], []
for r in rows:
    q = r["question"]
    if q not in gold: continue
    pred_dist = json.loads(r["distribution"])
    pred_sup  = list(map(str, json.loads(r["supports"])))
    gdist, grels = gold[q]
    jses.append(js(gdist, pred_dist))
    aps.append(ap_at_k(pred_sup, grels, 100))

print("Scored:", len(jses))
print("Mean JS:", sum(jses)/len(jses))
print("mAP@100:", sum(aps)/len(aps))
