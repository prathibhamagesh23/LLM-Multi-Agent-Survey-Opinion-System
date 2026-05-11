# CLI: python -m index.build --config <yaml> --api_key DUMMY
import argparse
import json
import gzip
import os
import pickle
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
from whoosh import index as windex
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import SimpleAnalyzer  # faster than stemming, good enough

ART_DIR = Path("artifacts/index")


def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                # skip malformed lines
                continue


def build_whoosh(docs_jsonl, limit_docs=None):
    """
    Build a Whoosh index with 4 stored text fields (title/description/post_content/content).
    Single-process writer (Windows/Py3.12 safe) + large RAM buffer.
    """
    ART_DIR.mkdir(parents=True, exist_ok=True)
    wdir = ART_DIR / "whoosh"
    wdir.mkdir(parents=True, exist_ok=True)

    schema = Schema(
        id=ID(stored=True, unique=True),
        title=TEXT(stored=True, analyzer=SimpleAnalyzer()),
        description=TEXT(stored=True, analyzer=SimpleAnalyzer()),
        post_content=TEXT(stored=True, analyzer=SimpleAnalyzer()),
        content=TEXT(stored=True, analyzer=SimpleAnalyzer()),
    )

    if not windex.exists_in(wdir):
        ix = windex.create_in(wdir, schema)
    else:
        ix = windex.open_dir(wdir)

    # Windows/Python 3.12: keep procs=1 (multiproc can crash). Use big buffer.
    writer = ix.writer(limitmb=1024, procs=1, multisegment=True)

    ids = []
    count = 0
    for rec in tqdm(iter_jsonl(docs_jsonl), desc="[Whoosh] indexing"):
        did = str(rec.get("id") or rec.get("_id") or "")
        if not did:
            continue
        fields_map = {
            "title":        str(rec.get("title", "")),
            "description":  str(rec.get("description", "")),
            "post_content": str(rec.get("post_content", "")),
            "content":      str(rec.get("content", "")),
        }
        writer.update_document(id=did, **fields_map)
        ids.append(did)

        count += 1
        if limit_docs and count >= int(limit_docs):
            break

    writer.commit()

    with gzip.open(ART_DIR / "doc_ids.pkl.gz", "wb") as f:
        pickle.dump(ids, f, protocol=pickle.HIGHEST_PROTOCOL)

    return ix, ids


def build_dense_table(id_to_embedding_npz):
    """
    Normalize and persist id->embedding table for MMR.
    """
    npz = np.load(id_to_embedding_npz, allow_pickle=True)
    ids = npz["ids"] if "ids" in npz.files else npz.get("arr_0")
    vecs = npz["embeddings"] if "embeddings" in npz.files else npz.get("arr_1")
    if ids is None or vecs is None:
        raise ValueError("id_to_embedding.npz must contain ids and embeddings (ids/arr_0, embeddings/arr_1).")

    ids = np.array(ids).ravel().tolist()
    ids = [str(x) for x in ids]

    vecs = np.asarray(vecs, dtype=np.float32)
    norms = (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12).astype(np.float32)
    vecs = (vecs / norms).astype(np.float32)

    pack = {"ids": ids, "vecs": vecs}
    with gzip.open(ART_DIR / "dense.pkl.gz", "wb") as f:
        pickle.dump(pack, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[index.build] dense table: {len(ids)} vectors saved")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--api_key", default="DUMMY")
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    docs_jsonl = cfg["paths"]["docs_jsonl"]
    id2npz = cfg["paths"]["id_to_embedding_npz"]

    # ✅ read debug cap AFTER cfg exists
    debug_limit = (cfg.get("debug") or {}).get("limit_docs")

    build_whoosh(docs_jsonl, limit_docs=debug_limit)
    try:
        build_dense_table(id2npz)
    except Exception as e:
        print("[WARN] Dense table not built:", e)

    print(f"[index.build] Done. Artifacts -> {ART_DIR}")


if __name__ == "__main__":
    main()
