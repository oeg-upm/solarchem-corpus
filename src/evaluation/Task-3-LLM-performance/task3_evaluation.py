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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from thefuzz import fuzz

os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-xxx"

EMBED_MODEL = "qwen/qwen3-embedding-8b"
BASE_URL = "https://openrouter.ai/api/v1"

CATEGORIES = ["catalyst", "co_catalyst", "light_source", "lamp",
              "reactor_type", "reaction_medium", "operation_mode"]
DISPLAY = {"catalyst": "Catalyst", "co_catalyst": "Co-Catalyst",
           "light_source": "Light Source", "lamp": "Lamp",
           "reactor_type": "Reactor Type", "reaction_medium": "Reaction Medium",
           "operation_mode": "Operation Mode"}

LOCAL_MODELS = {
    "google_gemma-4-26b-a4b-it", "google_gemma-4-31b-it",
    "qwen_qwen3.6-27b", "qwen_qwen3.6-35b-a3b",
}


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

def pretty(folder):
    return folder.split("_", 1)[1] if "_" in folder else folder

def is_local(folder):
    return folder in LOCAL_MODELS

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


def make_fig8(cos, rat, order):
    cp = [100 * mean(list(cos[m].values())) for m in order]
    rp = [100 * mean(list(rat[m].values())) for m in order]
    x, w = np.arange(len(order)), 0.38
    fig, ax = plt.subplots(figsize=(max(8, 1.25 * len(order)), 5.6))
    for i, m in enumerate(order):
        c1, c2 = ("#fdae6b", "#d94801") if is_local(m) else ("#9ecae1", "#3182bd")
        ax.bar(x[i] - w/2, cp[i], w, color=c1, ec="white", lw=.6)
        ax.bar(x[i] + w/2, rp[i], w, color=c2, ec="white", lw=.6)
        ax.text(x[i] - w/2, cp[i] + .8, f"{cp[i]:.1f}", ha="center", va="bottom", fontsize=8)
        ax.text(x[i] + w/2, rp[i] + .8, f"{rp[i]:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([pretty(m) for m in order], rotation=30, ha="right")
    ax.set_ylabel("Score"); ax.set_ylim(0, max(cp + rp) + 10)
    # legend lifted out of the plot, laid out horizontally above the axes
    ax.legend(handles=[Patch(facecolor="#9ecae1", label="API Models (cos_sim)"),
                       Patch(facecolor="#3182bd", label="API Models (ratio)"),
                       Patch(facecolor="#fdae6b", label="Local Models (cos_sim)"),
                       Patch(facecolor="#d94801", label="Local Models (ratio)")],
              loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4,
              frameon=False, fontsize=8, columnspacing=1.4, handletextpad=0.5)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", ls=":", alpha=.4); ax.set_axisbelow(True)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def get_parser():
    parser = argparse.ArgumentParser(description="SolarChemQA Task 3: LLM performance evaluation")
    parser.add_argument("--results_dir", default="./result/", help="Folder with one sub-folder of predictions per model", type=str)
    parser.add_argument("--ground_truth", default="../../../SolarChem-Dataset/domain_expert_annotation.json", help="Path of the domain expert annotation file", type=str)
    parser.add_argument("--output_dir", default="figures/", help="Folder for the metrics csv and the figures", type=str)
    parser.add_argument("--json_out", default="./task3_evaluation.json", help="Path for the result dictionary", type=str)
    parser.add_argument("--vecs_cache", default="../vecs.pkl", help="Path of the embedding cache, shared with the Task 2 script", type=str)
    return parser


def main():
    args = get_parser().parse_args()

    gt = {str(k): record_answers(v) for k, v in json.load(open(args.ground_truth)).items()}
    print(f"ground truth: {len(gt)} papers")

    models = {}
    for d in sorted(Path(args.results_dir).iterdir()):
        if not d.is_dir():
            continue
        preds = {}
        for fp in d.glob("*.json"):
            try:
                preds[paper_id(fp.name)] = record_answers(json.load(open(fp)))
            except Exception as e:
                print(f"  ! skip {fp.name}: {e}")
        if preds:
            models[d.name] = preds
            print(f"  {d.name}: {len(preds)} files")

    texts = {t for preds in models.values() for ans in preds.values() for t in ans.values()}
    texts |= {t for ans in gt.values() for t in ans.values()}
    vecs = load_vecs(args.vecs_cache, texts)

    cos, rat = {}, {}
    for name, preds in models.items():
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
        cos[name] = {c: mean(cb[c]) for c in CATEGORIES}
        rat[name] = {c: mean(rb[c]) for c in CATEGORIES}

    # order models by overall cos_sim
    model_order = sorted(cos, key=lambda m: mean(list(cos[m].values())), reverse=True)

    # per-category tables and per-model summary (Fig 8 data)
    cos_df = pd.DataFrame({pretty(m): cos[m] for m in model_order}).T[CATEGORIES].rename(columns=DISPLAY)
    rat_df = pd.DataFrame({pretty(m): rat[m] for m in model_order}).T[CATEGORIES].rename(columns=DISPLAY)
    summary = pd.DataFrame({
        "cos_sim (%)": [100 * mean(list(cos[m].values())) for m in model_order],
        "ratio (%)":   [100 * mean(list(rat[m].values())) for m in model_order],
    }, index=[pretty(m) for m in model_order])

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    summary.round(1).to_csv(Path(args.output_dir) / "task3_metrics.csv")

    results = {}
    for m in model_order:
        results[pretty(m)] = {
            "cos_sim": {DISPLAY[c]: round(cos[m][c], 4) for c in CATEGORIES},
            "ratio":   {DISPLAY[c]: round(rat[m][c], 4) for c in CATEGORIES},
            "average": {"cos_sim": round(mean(list(cos[m].values())), 4),
                        "ratio":   round(mean(list(rat[m].values())), 4)},
        }
    json.dump(results, open(args.json_out, "w"), indent=2)
    print(f"wrote {args.json_out}")

    print("Cosine similarity (model x category):")
    print(cos_df.round(4).to_string())
    print("Lexical ratio (model x category):")
    print(rat_df.round(4).to_string())
    print("Per-model averages:")
    print(summary.round(2).to_string())

    fig = make_fig8(cos, rat, model_order)
    fig.savefig(Path(args.output_dir) / "fig8_model_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(Path(args.output_dir) / "fig8_model_comparison.pdf", bbox_inches="tight")
    print(f"wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
