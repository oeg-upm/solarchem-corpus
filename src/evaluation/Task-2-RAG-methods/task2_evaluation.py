import argparse
import json
import os
import pickle
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from thefuzz import fuzz

os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-xxx"

EMBED_MODEL = "qwen/qwen3-embedding-8b"
BASE_URL = "https://openrouter.ai/api/v1"

CONFIGS = ["Naive-Naive", "Naive-Hybrid", "Naive-Rerank",
           "Recursive-Naive", "Recursive-Hybrid", "Recursive-Rerank",
           "Semantic-Naive", "Semantic-Hybrid", "Semantic-Rerank"]

CATEGORIES = ["catalyst", "co_catalyst", "light_source", "lamp",
              "reactor_type", "reaction_medium", "operation_mode"]
DISPLAY = {"catalyst": "Catalyst", "co_catalyst": "Co-Catalyst",
           "light_source": "Light Source", "lamp": "Lamp",
           "reactor_type": "Reactor Type", "reaction_medium": "Reaction Medium",
           "operation_mode": "Operation Mode"}


def norm_cat(k):
    k = re.sub(r"[\s\-]+", "_", k.strip().lower())
    return k if k in CATEGORIES else None

def record_answers(rec):
    out = {}
    for k, v in rec.items():
        c = norm_cat(k)
        if c:
            out[c] = "" if v is None else str(v).strip()
    return out

def paper_id(fname):
    nums = re.findall(r"\d+", Path(fname).stem)
    return nums[-1] if nums else Path(fname).stem

def mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")

def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def load_vecs(cache_path, texts):
    vecs = pickle.load(open(cache_path, "rb")) if Path(cache_path).exists() else {}
    missing = sorted(t for t in texts if t and t not in vecs)
    print(f"cached vectors: {len(vecs)}   new strings to embed: {len(missing)}")

    if missing:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit("Set OPENROUTER_API_KEY in your environment before running.")
        from langchain_openai import OpenAIEmbeddings
        client = OpenAIEmbeddings(model=EMBED_MODEL, base_url=BASE_URL, api_key=api_key)
        for i in range(0, len(missing), 128):
            batch = missing[i:i + 128]
            for attempt in range(3):
                try:
                    out = client.embed_documents(batch); break
                except Exception as e:
                    if attempt == 2: raise
                    print(f"  retry after: {e}"); time.sleep(2 * (attempt + 1))
            vecs.update({t: np.asarray(v, float) for t, v in zip(batch, out)})
        pickle.dump(vecs, open(cache_path, "wb"), protocol=4)
        print(f"cache now holds {len(vecs)} vectors")
    return vecs


def score(preds, gt, vecs):
    cb, rb = defaultdict(list), defaultdict(list)
    for pid, pa in preds.items():
        ga = gt.get(pid)
        if not ga:
            continue
        for c in CATEGORIES:
            if c in pa and ga.get(c):
                p, g = pa[c], ga[c]
                vp, vg = vecs.get(p), vecs.get(g)
                cb[c].append(cosine(vp, vg) if vp is not None and vg is not None else 0.0)
                rb[c].append(fuzz.partial_ratio(p, g) / 100.0)
    cos = {c: mean(cb[c]) for c in CATEGORIES}
    rat = {c: mean(rb[c]) for c in CATEGORIES}
    return cos, rat


def get_parser():
    parser = argparse.ArgumentParser(description="SolarChemQA Task 2: RAG configuration evaluation")
    parser.add_argument("--results_dir", default="./result/", help="Folder with one sub-folder of predictions per RAG configuration", type=str)
    parser.add_argument("--ground_truth", default="../../../SolarChem-Dataset/domain_expert_annotation.json", help="Path of the domain expert annotation file", type=str)
    parser.add_argument("--vecs_cache", default="../vecs.pkl", help="Path of the embedding cache, shared with the Task 3 script", type=str)
    parser.add_argument("--json_out", default="./task2_evaluation.json", help="Path for the result dictionary", type=str)
    return parser


def main():
    args = get_parser().parse_args()

    gt = {str(k): record_answers(v) for k, v in json.load(open(args.ground_truth)).items()}
    print(f"ground truth: {len(gt)} papers")

    configs = {}
    for name in CONFIGS:
        preds = {}
        for fp in (Path(args.results_dir) / name).glob("*.json"):
            preds[paper_id(fp.name)] = record_answers(json.load(open(fp)))
        configs[name] = preds
        print(f"  {name}: {len(preds)} files")

    texts = {t for preds in configs.values() for ans in preds.values() for t in ans.values()}
    texts |= {t for ans in gt.values() for t in ans.values()}
    vecs = load_vecs(args.vecs_cache, texts)

    header = [DISPLAY[c] for c in CATEGORIES] + ["Average"]
    cos_rows, rat_rows, results = {}, {}, {}
    for name in CONFIGS:
        cos, rat = score(configs[name], gt, vecs)
        cos_rows[name] = [cos[c] for c in CATEGORIES] + [mean(list(cos.values()))]
        rat_rows[name] = [rat[c] for c in CATEGORIES] + [mean(list(rat.values()))]
        results[name] = {
            "cos_sim": {DISPLAY[c]: round(cos[c], 4) for c in CATEGORIES},
            "ratio":   {DISPLAY[c]: round(rat[c], 4) for c in CATEGORIES},
            "average": {"cos_sim": round(mean(list(cos.values())), 4),
                        "ratio":   round(mean(list(rat.values())), 4)},
        }

    cos_df = pd.DataFrame.from_dict(cos_rows, orient="index", columns=header)
    rat_df = pd.DataFrame.from_dict(rat_rows, orient="index", columns=header)

    print("Semantic similarity (cos_sim) by RAG configuration:")
    print(cos_df.round(4).to_string())
    print("Lexical matching (ratio) by RAG configuration:")
    print(rat_df.round(4).to_string())

    json.dump(results, open(args.json_out, "w"), indent=2)
    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
