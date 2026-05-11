# CLI: python -m mas_survey.run --config <yaml> --api_key DUMMY
import argparse, json, csv, yaml, gzip, pickle, math, sys, glob
from pathlib import Path
from collections import defaultdict
from functools import lru_cache
from tqdm import tqdm
import numpy as np
from whoosh import index as windex
from whoosh.qparser import MultifieldParser, OrGroup

ART_DIR = Path("artifacts/index")  # expects: whoosh/, doc_ids.pkl.gz, dense.pkl.gz


# ---------------- I/O helpers ----------------

def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_questions(path):
    """
    Normalize questions into:
      [{"question": str, "options": [str, ...]}, ...]
    Accepts:
      - [ {"question": "...", "options": ["..",".."]}, ... ]
      - [ {"question": "...", "A": "...", "B": "...", ... }, ... ]
      - { "<question>": {"distribution": {opt: p, ...}, "supports": [...]}, ... }
      - { "<question>": [ "...", "...", ... ], ... }
      - { "<question>": {"options": [ ... ]}, ... }
      - {"dev":[...]} / {"test":[...]} / {"questions":[...]} / {"data":[...]} / {"items":[...]} / {"examples":[...]}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Unwrap common top-level containers (dict -> list)
    if isinstance(data, dict):
        for key in ("questions", "items", "data", "dev", "test", "eval", "records", "examples"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            # Treat as mapping: question_text -> structure
            mapped = []
            for qtext, val in data.items():
                q = str(qtext).strip()

                # Case: {"distribution": {...}, "supports": [...]}
                if isinstance(val, dict) and "distribution" in val and isinstance(val["distribution"], dict):
                    options = [str(k) for k in val["distribution"].keys()]
                    mapped.append({"question": q, "options": options})
                    continue

                # Case: value is a list of options
                if isinstance(val, list):
                    options = [str(x) for x in val]
                    mapped.append({"question": q, "options": options})
                    continue

                # Case: dict with embedded options list
                if isinstance(val, dict) and "options" in val and isinstance(val["options"], list):
                    options = [str(x) for x in val["options"]]
                    mapped.append({"question": q, "options": options})
                    continue

                # Case: dict with lettered/choice keys
                if isinstance(val, dict):
                    opt_keys = []
                    for k in val.keys():
                        kl = k.lower()
                        if kl in ("a","b","c","d","e","f","g") or kl.startswith("option") or kl.startswith("choice"):
                            opt_keys.append(k)
                    if opt_keys:
                        opt_keys = sorted(opt_keys)
                        options = [str(val[k]) for k in opt_keys]
                        mapped.append({"question": q, "options": options})
                        continue

                raise ValueError(f"Unsupported entry for question '{q[:60]}...': {type(val)}")
            data = mapped

    if not isinstance(data, list):
        raise ValueError(f"Unsupported questions JSON. Expected list, got {type(data)}")

    # Normalize list items to canonical schema
    norm = []
    for i, item in enumerate(data):
        if isinstance(item, dict) and "question" in item and "options" in item and isinstance(item["options"], list):
            norm.append({"question": str(item["question"]).strip(),
                         "options": [str(x) for x in item["options"]]})
            continue

        if isinstance(item, dict) and "question" in item:
            opt_keys = []
            for k in item.keys():
                kl = k.lower()
                if kl in ("a","b","c","d","e","f","g") or kl.startswith("option") or kl.startswith("choice"):
                    opt_keys.append(k)
            if opt_keys:
                opt_keys = sorted(opt_keys)
                options = [str(item[k]) for k in opt_keys]
                norm.append({"question": str(item["question"]).strip(), "options": options})
                continue

        if isinstance(item, str):
            raise ValueError(
                f"Entry {i} is a bare string; need options: "
                f"{{'question': '...', 'options': ['opt1','opt2', ...]}}"
            )

        raise ValueError(f"Unsupported question record at index {i}: {type(item)} -> {item}")
    return norm


def apply_calibration_to_rows(rows, alpha=None, temperature=None):
    """Returns a new list of rows with calibrated distributions."""
    if alpha is None and temperature is None:
        return rows
    out = []
    for row in rows:
        dist = row["distribution"]
        keys = list(dist.keys())
        K = len(keys)
        vec = [max(0.0, float(dist[k])) for k in keys]
        s = sum(vec) or 1.0
        vec = [v/s for v in vec]
        if alpha is not None:
            vec = [v + float(alpha)/K for v in vec]
            s2 = sum(vec)
            vec = [v/s2 for v in vec]
        if temperature is not None and abs(temperature - 1.0) > 1e-6:
            vec = [v**(1.0/float(temperature)) for v in vec]
            s3 = sum(vec)
            vec = [v/s3 for v in vec]
        new_row = dict(row)
        new_row["distribution"] = {k: float(v) for k, v in zip(keys, vec)}
        out.append(new_row)
    return out


# ---------------- Index/dense table ----------------

def load_doc_ids():
    p = ART_DIR / "doc_ids.pkl.gz"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p} (did you run the index builder?)")
    with gzip.open(p, "rb") as f:
        return set(pickle.load(f))


def load_dense_table():
    p = ART_DIR / "dense.pkl.gz"
    if not p.exists():
        return None
    with gzip.open(p, "rb") as f:
        data = pickle.load(f)
    id_to_idx = {str(did): i for i, did in enumerate(data["ids"])}
    return {"id_to_idx": id_to_idx, "vecs": data["vecs"]}


# ---------------- Text access (cached) ----------------

def get_doc_text_from_ix(searcher, did: str) -> str:
    # cache repeated reads (CE/option scoring often touches the same doc IDs)
    return _get_doc_text_from_ix_cached(searcher, did)

@lru_cache(maxsize=65536)
def _get_doc_text_from_ix_cached(searcher, did: str) -> str:
    d = searcher.document(id=did)
    if not d:
        return ""
    parts = []
    for k in ("title", "description", "post_content", "content"):
        v = d.get(k)
        if v:
            parts.append(v)
    return " \n ".join(parts)


# ---------------- Retrieval helpers ----------------

def whoosh_search(ix, queries, topk=200, searcher=None):
    qp = MultifieldParser(["title","description","post_content","content"],
                          schema=ix.schema, group=OrGroup)
    results = []
    if searcher is not None:
        for q in queries:
            qobj = qp.parse(q)
            rs = searcher.search(qobj, limit=topk)
            results.append([(r["id"], float(r.score)) for r in rs])
        return results
    with ix.searcher() as s:
        for q in queries:
            qobj = qp.parse(q)
            rs = s.search(qobj, limit=topk)
            results.append([(r["id"], float(r.score)) for r in rs])
    return results


def rrf_merge(ranked_lists, k=300, k_rrf=60.0):
    scores = defaultdict(float)
    for rl in ranked_lists:
        for rank, (did, _) in enumerate(rl, start=1):
            scores[did] += 1.0 / (k_rrf + rank)
    merged = sorted(scores.items(), key=lambda x: -x[1])
    return merged[:k]


def mmr_select(candidates, doc_vecs=None, id_to_idx=None, final_k=120, lam=0.25):
    if doc_vecs is None or id_to_idx is None:
        return [did for did, _ in candidates[:final_k]]
    order = [did for did, _ in candidates]
    idxs = [id_to_idx.get(did, -1) for did in order]
    keep = [(did, i) for did, i in zip(order, idxs) if i >= 0]
    if not keep:
        return [did for did, _ in candidates[:final_k]]
    order, idxs = zip(*keep)
    X = doc_vecs[list(idxs)]
    relevance = np.linspace(1.0, 0.0, num=len(order))  # rank-based
    chosen, chosen_set = [], set()
    while len(chosen) < min(final_k, len(order)):
        if not chosen:
            j = 0
        else:
            S = np.array([doc_vecs[id_to_idx[cid]] for cid in chosen])
            sims = X @ S.T
            max_sim = sims.max(axis=1)
            mmr = lam * relevance - (1 - lam) * max_sim
            j = int(mmr.argmax())
        did = order[j]
        if did not in chosen_set:
            chosen.append(did)
            chosen_set.add(did)
        relevance[j] = -1e9
    return chosen


def prf_expand_terms(searcher, doc_ids, top_terms=12):
    from collections import Counter
    bag = Counter()
    for did in doc_ids:
        d = searcher.document(id=did)
        if not d:
            continue
        text = " ".join([d.get(k,"") for k in ("title","description","post_content","content")]).lower()
        for tok in text.split():
            if len(tok) > 2:
                bag[tok] += 1
    return " ".join([w for w,_ in bag.most_common(top_terms)])


# ---------------- Planner & Scoring ----------------

def planner_make_queries(q, n_queries=8):
    base = q["question"].strip()
    opts = [o.strip() for o in q.get("options", [])]
    queries = [base] + [f"{base} {o}" for o in opts]
    seen, out = set(), []
    for qu in queries:
        if qu not in seen:
            out.append(qu)
            seen.add(qu)
    return out[:n_queries]


def option_cues(options):
    cues = []
    for o in options:
        toks = [t for t in o.lower().split() if len(t) > 1]
        cues.append(set(toks))
    return cues


def score_options_from_docs(options, ranked_docs_text, dirichlet_alpha=0.3, temperature=1.0):
    cues = option_cues(options)
    raw = np.zeros(len(options), dtype=float)
    for rank, (_doc_id, _score, text) in enumerate(ranked_docs_text[:80], start=1):
        w = 1.0 / (rank ** 0.5)
        low = text.lower()
        for i, cue in enumerate(cues):
            hits = sum(1 for tok in cue if tok in low)
            if hits:
                raw[i] += w * (1 + math.log1p(hits))
    raw = np.maximum(raw, 0)
    raw = raw / (raw.sum() + 1e-9)
    raw = (raw + dirichlet_alpha / max(1, len(options)))
    raw = raw / raw.sum()
    if abs(temperature - 1.0) > 1e-6:
        raw = raw ** (1.0 / temperature)
        raw = raw / raw.sum()
    return raw.tolist()


# ---------------- CSV helpers ----------------

def ensure_100_supports(cand_ids, universe_ids, need=100):
    uniq, seen = [], set()
    for did in cand_ids:
        if did in universe_ids and did not in seen:
            uniq.append(did)
            seen.add(did)
        if len(uniq) >= need:
            return uniq
    for did in universe_ids:
        if did not in seen:
            uniq.append(did)
            seen.add(did)
            if len(uniq) >= need:
                break
    return uniq[:need]


def validate_row(qobj, dist, supports, all_doc_ids):
    keys_ok = sorted(dist.keys()) == sorted([o for o in qobj["options"]])
    sum_ok = abs(sum(dist.values()) - 1.0) <= 1e-6
    nonneg = all(v >= -1e-9 for v in dist.values())
    supp_100 = len(supports) == 100 and len(set(supports)) == 100
    supp_exist = all((s in all_doc_ids) for s in supports)
    return keys_ok and sum_ok and nonneg and supp_100 and supp_exist


def write_csv(rows, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question", "distribution", "supports"])  # required header
        for r in rows:
            w.writerow([
                r["question"],
                json.dumps(r["distribution"], ensure_ascii=False),
                json.dumps(r["supports"], ensure_ascii=False),
            ])


# ---------------- Cross-Encoder (lazy load) ----------------

def _load_cross_encoder(model_name, revision=None):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification  # lazy import
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name, revision=revision)
    mdl.to("cpu")
    mdl.eval()
    return tok, mdl


def _rerank_with_cross_encoder(tokenizer, model, query, doc_texts, batch_size=16, max_len=256):
    import torch  # lazy import
    scores = []
    for i in range(0, len(doc_texts), batch_size):
        batch = doc_texts[i:i+batch_size]
        enc = tokenizer([query]*len(batch), batch, truncation=True,
                        max_length=max_len, padding=True, return_tensors="pt")
        out = model(**{k: v.to("cpu") for k, v in enc.items()})
        logits = out.logits.squeeze(-1)
        sc = logits.detach().cpu().tolist()
        if isinstance(sc, float):
            sc = [sc]
        scores.extend(sc)
    return scores


def _score_options_ce(tokenizer, model, question, options, doc_texts, batch_size=16, max_len=256):
    import torch, math  # lazy import
    opt_scores = []
    for opt in options:
        query = f"{question} || Option: {opt}"
        scores = []
        for i in range(0, len(doc_texts), batch_size):
            batch = doc_texts[i:i+batch_size]
            enc = tokenizer([query]*len(batch), batch, truncation=True,
                            max_length=max_len, padding=True, return_tensors="pt")
            out = model(**{k: v.to("cpu") for k, v in enc.items()})
            logits = out.logits.squeeze(-1).detach().cpu().numpy().tolist()
            if isinstance(logits, float):
                logits = [logits]
            scores.extend(logits)
        m = max(scores) if scores else 0.0
        lse = m + math.log(sum(math.exp(s-m) for s in scores)) if scores else -1e9
        opt_scores.append(lse)
    if not opt_scores:
        return [1.0 / max(1, len(options))] * len(options)
    m = max(opt_scores)
    exps = [math.exp(s-m) for s in opt_scores]
    Z = sum(exps) or 1.0
    return [x/Z for x in exps]


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--api_key", default="DUMMY")
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    paths = cfg.get("paths", {})

    # Questions path (robust fallback)
    questions_json = paths.get("questions_json", "data/dev/dev.json")
    q_candidates = [questions_json, "data/dev.json", "data/dev/dev.json"]
    if not any(Path(p).exists() for p in q_candidates if p):
        matches = sorted(glob.glob("data/**/*dev*.json", recursive=True))
        if matches:
            questions_json = matches[0]
        else:
            print("[error] Could not find questions JSON. Tried:", file=sys.stderr)
            for p in q_candidates:
                print(" -", p, file=sys.stderr)
            print(" Also searched: data/**/*dev*.json", file=sys.stderr)
            sys.exit(2)
    else:
        for p in q_candidates:
            if p and Path(p).exists():
                questions_json = p
                break

    # Optional base output; derive js/map if missing
    output_csv = paths.get("output_csv", None)

    # Open indices/tables
    ix = windex.open_dir(ART_DIR / "whoosh")
    dense = load_dense_table()
    all_doc_ids = load_doc_ids()  # set of valid IDs (from index)

    # Load and sanity-check questions
    qs = read_questions(questions_json)
    assert isinstance(qs, list) and len(qs) > 0, "read_questions() must return a non-empty list"
    first = qs[0]
    assert isinstance(first, dict) and "question" in first and "options" in first and isinstance(first["options"], list), \
        f"Bad schema for first item: {type(first)} -> {first}"

    # Knobs (defaults can be overridden via YAML)
    n_queries = int(cfg.get("agents", {}).get("planner", {}).get("n_queries", 5))
    topk = int(cfg.get("retrieval", {}).get("bm25", {}).get("topk", 160))
    mmr_cfg = cfg.get("retrieval", {}).get("mmr", {"enabled": True, "lambda": 0.25, "final_k": 120})
    agg_cfg = cfg.get("aggregate", {})
    supports_exact = int(agg_cfg.get("supports_exact", 100))
    prf_terms = int(cfg.get("retrieval", {}).get("prf", {}).get("top_terms", 12))  # optional PRF knob

    # Limiter for quick probes
    limit_q = int(cfg.get("debug", {}).get("limit_questions", 0))
    if limit_q > 0:
        qs = qs[:limit_q]

    out_rows = []

    with ix.searcher() as searcher:
        for q in tqdm(qs, desc="[run] questions"):
            # planner → multi-query
            queries = planner_make_queries(q, n_queries=n_queries)

            # BM25 per query (use the topk you set above)
            ranked_lists = whoosh_search(ix, queries, topk=topk, searcher=searcher)

            # Reciprocal-rank fusion
            merged = rrf_merge(ranked_lists, k=220, k_rrf=60.0)

            # RM3-lite PRF (optional if prf_terms > 0)
            if prf_terms and prf_terms > 0:
                prf_ids = [did for did,_ in merged[:10]]
                expansion = prf_expand_terms(searcher, prf_ids, top_terms=prf_terms)
                if expansion:
                    base = q["question"].strip()
                    exp_queries = [f"{base} {expansion}"] + [f"{base} {opt} {expansion}" for opt in q["options"]]
                    extra = whoosh_search(ix, exp_queries, topk=topk, searcher=searcher)
                    ranked_lists.extend(extra)
                    merged = rrf_merge(ranked_lists, k=240, k_rrf=60.0)

            # MMR diversity selection
            if mmr_cfg.get("enabled", True) and dense is not None:
                sel_ids = mmr_select(
                    merged,
                    doc_vecs=dense["vecs"],
                    id_to_idx=dense["id_to_idx"],
                    final_k=int(mmr_cfg.get("final_k", 120)),
                    lam=float(mmr_cfg.get("lambda", 0.25)),
                )
            else:
                sel_ids = [did for did, _ in merged[:int(mmr_cfg.get("final_k", 120))]]

            # --- Optional cross-encoder reranker ---
            ce_cfg = cfg.get("reranker", {"enabled": False})
            if ce_cfg.get("enabled", False):
                # Check environment for CE
                try:
                    import torch  # noqa: F401
                    from transformers import AutoTokenizer  # quick probe
                    _ce_ok = True
                except Exception:
                    _ce_ok = False

                if not _ce_ok:
                    print("[reranker] torch/transformers not available — disabling CE for this run.")
                else:
                    model_name = ce_cfg.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
                    revision   = ce_cfg.get("revision", None)
                    batch_size = int(ce_cfg.get("batch_size", 16))
                    max_len    = int(ce_cfg.get("max_length", 256))
                    keep_top   = int(ce_cfg.get("keep_top", 60))

                    if not hasattr(main, "_ce_loaded"):
                        print(f"[reranker] loading {model_name}")
                        main._ce_tok, main._ce_mdl = _load_cross_encoder(model_name, revision)
                        main._ce_loaded = True

                    texts_for_ce = [get_doc_text_from_ix(searcher, str(did)) for did in sel_ids]
                    ce_scores = _rerank_with_cross_encoder(main._ce_tok, main._ce_mdl, q["question"],
                                                           texts_for_ce, batch_size=batch_size, max_len=max_len)
                    pairs = list(zip(sel_ids, ce_scores))
                    pairs.sort(key=lambda x: -x[1])
                    sel_ids = [did for did, _ in pairs[:keep_top]]

            # Build texts lazily from index → score fewer docs (speed)
            score_map = {did: sc for did, sc in merged}
            ranked_docs_text = []
            for did in sel_ids[:60]:  # small slice for speed
                text = get_doc_text_from_ix(searcher, str(did))
                ranked_docs_text.append((did, score_map.get(did, 0.0), text))

            # Keyword-based baseline option scoring (fast, robust)
            kw_probs = score_options_from_docs(
                q["options"], ranked_docs_text,
                dirichlet_alpha=float(agg_cfg.get("dirichlet_alpha", 0.3)),
                temperature=float(agg_cfg.get("temperature", 1.0)),
            )

            # If CE is enabled and loaded, blend CE option scores with keyword
            probs = None
            if ce_cfg.get("enabled", False) and hasattr(main, "_ce_loaded"):
                doc_texts_for_options = [txt for (_, _, txt) in ranked_docs_text[:60]]
                probs_ce = _score_options_ce(
                    main._ce_tok, main._ce_mdl,
                    q["question"], q["options"],
                    doc_texts_for_options,
                    batch_size=int(ce_cfg.get("batch_size", 16)),
                    max_len=int(ce_cfg.get("max_length", 256)),
                )
                gamma = float(cfg.get("aggregate", {}).get("interp_gamma", 0.2))
                probs = [(1.0 - gamma) * pc + gamma * pk for pc, pk in zip(probs_ce, kw_probs)]

            if probs is None:
                probs = kw_probs

            dist = {opt: float(p) for opt, p in zip(q["options"], probs)}
            supports = ensure_100_supports(sel_ids, list(all_doc_ids), need=supports_exact)

            if not validate_row(q, dist, supports, all_doc_ids):
                # Repair: clip negatives, renorm, repad supports
                for k in dist:
                    dist[k] = max(0.0, float(dist[k]))
                s = sum(dist.values())
                if s == 0:
                    z = 1.0 / max(1, len(dist))
                    for k in dist:
                        dist[k] = z
                else:
                    for k in dist:
                        dist[k] /= s
                supports = ensure_100_supports(supports, list(all_doc_ids), need=supports_exact)

            out_rows.append({"question": q["question"], "distribution": dist, "supports": supports})

    # ------- Write outputs (JS + MAP) -------
    # Derive outputs if base is missing
    if output_csv is None:
        out_js  = paths.get("output_js_csv",  "artifacts/submission_js.csv")
        out_map = paths.get("output_map_csv", "artifacts/submission_map.csv")
    else:
        out_js  = paths.get("output_js_csv",  output_csv.replace(".csv", "_js.csv"))
        out_map = paths.get("output_map_csv", output_csv.replace(".csv", "_map.csv"))

    cal = cfg.get("aggregate_js", {})  # optional JS-specific calibration
    rows_js = apply_calibration_to_rows(
        out_rows,
        alpha=cal.get("dirichlet_alpha"),
        temperature=cal.get("temperature"),
    )
    write_csv(rows_js, out_js)
    print(f"[mas_survey.run] wrote {out_js}")

    write_csv(out_rows, out_map)
    print(f"[mas_survey.run] wrote {out_map}")


if __name__ == "__main__":
    main()
