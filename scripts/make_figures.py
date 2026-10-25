"""Every report figure and table, regenerated from raw result files.

    python scripts/make_figures.py --results "results/*.jsonl" --out figures

Figures are grayscale (IEEE print): series differ by marker and line style,
never by colour alone; our method is the solid black line. Any figure whose
input rows are missing is skipped with a message, so this can run at any
stage of the project.

Outputs (PNG at 300 dpi + PDF):
  fig1_score_vs_budget      LongBench average vs KV budget, per method (RQ1)
  fig2_entropy_profiles     per-layer entropy for very different inputs (motivation)
  fig3_dynamic_vs_static    per-task gain of entropy over entropy_static and PyramidKV, 95% CI (RQ2)
  fig4_passkey_heatmaps     passkey accuracy, depth x length, per method
  fig5_ablations            tau / floor / random-allocation control
  fig6_memory_latency       KV size, decode latency, allocation overhead (RQ4; profile_*.jsonl)
  fig7_layer_starvation     accuracy change vs the most-starved layer's share (why allocations fail)
Tables:
  fig8_direction_test       passkey at 2% of 4K: entropy vs reversed vs SnapKV vs CAKE, both models
  table_main.csv / .md      task x method at the 20% budget (x100)
  table_llama.csv / .md     Llama-3.2-3B at 10% and 20%, plus table_llama_significance.csv
  table_significance.csv    ours minus each baseline, paired bootstrap CI, per budget
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from kvprune.stats import paired_bootstrap  # noqa: E402

OURS = "entropy"
STYLE = {  # method -> (gray, marker, linestyle, linewidth, display name)
    "full": ("0.0", "", ":", 1.2, "Full cache"),
    "entropy": ("0.0", "o", "-", 2.0, "EntropyKV (ours)"),
    "entropy_static": ("0.35", "s", "--", 1.4, "EntropyKV-static"),
    "cake_style": ("0.35", "^", "-.", 1.4, "CAKE-style"),
    "pyramidkv": ("0.55", "D", "--", 1.4, "PyramidKV"),
    "snapkv": ("0.55", "v", "-", 1.4, "SnapKV"),
    "streaming": ("0.7", "x", ":", 1.4, "StreamingLLM"),
    "random_alloc": ("0.7", "+", "--", 1.2, "Random allocation"),
    "random_tokens": ("0.8", ".", ":", 1.2, "Random tokens"),
}
ORDER = list(STYLE)
TASK_LABEL = {"qasper": "Qasper", "hotpotqa": "HotpotQA", "gov_report": "GovReport", "trec": "TREC",
              "passage_retrieval_en": "PassageRet.", "lcc": "LCC", "narrativeqa": "NarrativeQA",
              "multifieldqa_en": "MultiFieldQA", "2wikimqa": "2WikiMQA", "multi_news": "MultiNews",
              "samsum": "SAMSum", "passage_count": "PassageCount", "repobench-p": "RepoBench-P"}


def setup():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "0.88", "grid.linewidth": 0.6,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)
    print("  wrote", name)


def name_of(m):
    return STYLE.get(m, ("0.5", "", "-", 1, m))[4]


def load(pattern):
    rows = []
    for p in sorted(glob.glob(pattern)):
        with open(p, encoding="utf-8") as f:
            rows += [json.loads(l) for l in f if l.strip()]
    return pd.DataFrame(rows)


def lb(df):
    return df[df["task"] != "passkey"] if len(df) else df


# ----------------------------------------------------------------- figure 1
def fig1(df, out):
    d = lb(df)
    d = d[d["method"].isin(STYLE)]
    if d.empty:
        return print("  skip fig1: no LongBench rows")
    # Average over tasks of per-task means (each task weighted equally, as LongBench reports).
    per_task = d.groupby(["method", "budget", "task"])["score"].mean().reset_index()
    avg = per_task.groupby(["method", "budget"])["score"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    full = avg[avg["method"] == "full"]["score"]
    for m in ORDER:
        if m == "full" or m not in set(avg["method"]):
            continue
        s = avg[avg["method"] == m].sort_values("budget")
        g, mk, ls, lw, lab = STYLE[m]
        ax.plot(100 * s["budget"], 100 * s["score"], color=g, marker=mk, ls=ls, lw=lw, ms=4, label=lab)
    if len(full):
        ax.axhline(100 * full.iloc[0], color="0", ls=":", lw=1.2, label="Full cache")
    ax.set_xlabel("KV cache budget (% of full cache)")
    ax.set_ylabel("LongBench average score")
    ax.set_title("Accuracy vs. memory budget")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    save(fig, out, "fig1_score_vs_budget")


# ----------------------------------------------------------------- figure 2
def fig2(df, out):
    d = lb(df)
    if d.empty or "entropies" not in d:
        return print("  skip fig2: no entropy rows")
    d = d[d["method"] == "full"].drop_duplicates(["task", "idx"])
    picks = d.groupby("task").head(1).head(4)
    if picks.empty:
        return print("  skip fig2: no full-cache rows")
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    styles = [("0.0", "-"), ("0.35", "--"), ("0.55", "-."), ("0.7", ":")]
    for (g, ls), (_, r) in zip(styles, picks.iterrows()):
        e = r["entropies"]
        ax.plot(range(len(e)), e, color=g, ls=ls, lw=1.4, label=f"{TASK_LABEL.get(r['task'], r['task'])} #{r['idx']}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalised attention entropy")
    ax.set_title("Per-layer entropy differs across inputs")
    ax.legend(frameon=False, loc="best")
    save(fig, out, "fig2_entropy_profiles")


# ----------------------------------------------------------------- figure 3
def _paired(d, a, b, budget):
    x = d[(d["method"] == a) & (d["budget"] == budget)].set_index(["task", "idx"])["score"]
    y = d[(d["method"] == b) & (d["budget"] == budget)].set_index(["task", "idx"])["score"]
    j = x.to_frame("a").join(y.to_frame("b"), how="inner")
    return j


def fig3(df, out, budget=0.2):
    d = lb(df)
    comps = [c for c in ("entropy_static", "pyramidkv") if c in set(d.get("method", []))]
    if d.empty or OURS not in set(d["method"]) or not comps:
        return print("  skip fig3: need entropy and entropy_static/pyramidkv rows")
    tasks = sorted(d["task"].unique())
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    width = 0.8 / len(comps)
    for k, c in enumerate(comps):
        means, los, his = [], [], []
        for t in tasks:
            j = _paired(d[d["task"] == t], OURS, c, budget)
            if len(j) < 2:
                means.append(np.nan); los.append(0); his.append(0); continue
            mdiff, lo, hi, _ = paired_bootstrap(j["a"], j["b"])
            means.append(100 * mdiff); los.append(100 * (mdiff - lo)); his.append(100 * (hi - mdiff))
        x = np.arange(len(tasks)) + (k - (len(comps) - 1) / 2) * width
        ax.bar(x, means, width * 0.92, color=["0.35", "0.75"][k], edgecolor="0", lw=0.6,
               hatch=["", "///"][k], yerr=[los, his], capsize=2, error_kw={"lw": 0.8},
               label=f"vs {name_of(c)}")
    ax.axhline(0, color="0", lw=0.8)
    ax.set_xticks(range(len(tasks)), [TASK_LABEL.get(t, t) for t in tasks], rotation=30, ha="right")
    ax.set_ylabel("Score gain (points)")
    ax.set_title(f"EntropyKV gain at {int(budget * 100)}% budget (95% CI)")
    ax.legend(frameon=False)
    save(fig, out, "fig3_dynamic_vs_static")


# ----------------------------------------------------------------- figure 4
def fig4(df, out, budget=None):
    d = df[df["task"] == "passkey"] if len(df) else df
    if len(d) and "experiment" in d:  # Qwen heatmap runs only (confirm runs are Fig 8)
        d = d[d["experiment"].astype(str).isin(["passkey_qwen3b", "passkey_extreme_qwen3b"])]
    if d.empty:
        return print("  skip fig4: no passkey rows")
    if budget is None:  # hardest budget available: the one that separates methods most
        budget = float(d[d["method"] != "full"]["budget"].min())
    d = d[(d["budget"] == budget) | (d["method"] == "full")]
    methods = [m for m in ["full", "snapkv", "pyramidkv", "random_alloc", "cake_style", OURS]
               if m in set(d["method"])]
    fig, axes = plt.subplots(1, len(methods), figsize=(1.6 * len(methods) + 0.6, 2.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, methods):
        p = d[d["method"] == m].pivot_table(index="depth", columns="length", values="score", aggfunc="mean")
        im = ax.imshow(100 * p.values, cmap="Greys_r", vmin=0, vmax=100, aspect="auto", origin="lower")
        ax.set_xticks(range(len(p.columns)), [f"{c // 1024}K" for c in p.columns])
        ax.set_yticks(range(len(p.index)), [f"{int(100 * v)}%" for v in p.index])
        ax.set_title(name_of(m), fontsize=8)
        ax.grid(False)
        for (i, j), v in np.ndenumerate(p.values):
            ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=6.5,
                    color="0" if v > 0.5 else "1")
        ax.set_xlabel("Context length")
    axes[0].set_ylabel("Needle depth")
    fig.colorbar(im, ax=axes, shrink=0.8, label="Accuracy (%)")
    fig.suptitle(f"Passkey retrieval at {budget * 100:g}% budget", fontsize=9.5, y=1.08)
    save(fig, out, "fig4_passkey_heatmaps")


# ----------------------------------------------------------------- figure 5
ABL_LABEL = {
    "entropy": "Entropy, τ = 1 (ours)", "entropy:tau=0.5": "Entropy, τ = 0.5", "entropy:tau=2": "Entropy, τ = 2",
    "entropy:tau=-1": "Entropy reversed, τ = −1", "entropy:floor=16": "Entropy, floor 16",
    "entropy:floor=64": "Entropy, floor 64", "random_alloc": "Random split, seed 0",
    "random_alloc:seed=1": "Random split, seed 1", "random_alloc:seed=2": "Random split, seed 2",
}


def fig5(df, out, base="snapkv"):
    """Each ablation variant minus SnapKV on the same samples, with paired 95% bootstrap CI."""
    d = lb(df)
    if d.empty or "experiment" not in d:
        return print("  skip fig5: no rows")
    d = d[d["experiment"].astype(str).str.startswith("ablation")]
    if d.empty:
        return print("  skip fig5: no ablation rows")
    pts = []
    for (exp, m), g in d.groupby(["experiment", "method"]):
        if m == base:
            continue
        ref = d[(d["experiment"] == exp) & (d["method"] == base)].set_index(["task", "idx"])["score"]
        j = g.set_index(["task", "idx"])["score"].to_frame("a").join(ref.to_frame("b"), how="inner")
        if len(j) < 2:
            continue
        diff, lo, hi, _ = paired_bootstrap(j["a"], j["b"])
        lab = ABL_LABEL.get(m, m)
        if exp != "ablation_qwen3b":
            lab += f" (window {exp.split('window')[-1].split('_')[0]})"
        pts.append((lab, 100 * diff, 100 * lo, 100 * hi, m.startswith("random")))
    pts.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(3.5, 0.27 * len(pts) + 0.9))
    for y, (lab, mdiff, lo, hi, rnd) in enumerate(pts):
        ax.errorbar(mdiff, y, xerr=[[mdiff - lo], [hi - mdiff]], fmt="s" if rnd else "o",
                    color="0.6" if rnd else "0", ms=4, capsize=2, lw=0.9)
    ax.axvline(0, color="0", lw=0.8)
    ax.set_yticks(range(len(pts)), [p[0] for p in pts], fontsize=7)
    b = sorted(d["budget"].unique())
    bl = f" at {b[0] * 100:g}% budget" if len(b) == 1 else ""
    ax.set_xlabel(f"Score change vs SnapKV (points){bl}")
    ax.set_title("Ablations (95% CI): entropy variants vs random splits")
    ax.grid(axis="y", visible=False)
    save(fig, out, "fig5_ablations")


# ----------------------------------------------------------------- figure 6
def fig6(prof, out):
    if prof is None or prof.empty:
        return print("  skip fig6: no profile_*.jsonl rows (run scripts/profile_memory.py)")
    L = int(prof["length"].max())
    p = prof[prof["length"] == L]
    g = p.groupby(["method", "budget"]).agg(kv=("kv_mb", "mean"), dec=("decode_ms_tok", "mean"),
                                            ovh=("overhead_ms", "mean")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    # Every pruned method keeps exactly the same total, so memory is one line by construction.
    pruned = g[g["method"] != "full"].groupby("budget")["kv"].mean()
    axes[0].plot(100 * pruned.index, pruned.values, color="0", marker="o", lw=1.6, ms=4,
                 label="Any pruned method (equal by design)")
    for m in ORDER:
        s = g[g["method"] == m].sort_values("budget")
        if s.empty:
            continue
        gr, mk, ls, lw, lab = STYLE[m]
        if m == "full":
            axes[0].axhline(s["kv"].iloc[0], color="0", ls=":", lw=1.2, label=lab)
            axes[1].axhline(s["dec"].iloc[0], color="0", ls=":", lw=1.2, label=lab)
            continue
        axes[1].plot(100 * s["budget"], s["dec"], color=gr, marker=mk, ls=ls, lw=lw, ms=4, label=lab)
    axes[0].legend(frameon=False, fontsize=7, loc="center right")
    axes[0].set_ylabel("KV cache size (MB)")
    axes[1].set_ylabel("Decode latency (ms / token)")
    for ax in axes:
        ax.set_xlabel("KV cache budget (%)")
    axes[0].set_title(f"Memory at {round(L / 1024)}K context")
    axes[1].set_title(f"Decode latency at {round(L / 1024)}K context")
    axes[1].legend(frameon=False, fontsize=7)
    ovh = g[g["method"] == OURS]["ovh"].mean() if OURS in set(g["method"]) else float("nan")
    fig.text(0.5, -0.12, f"EntropyKV allocation + selection overhead: {ovh:.1f} ms per prompt "
                         f"(one-time, after prefill)", ha="center", fontsize=8)
    save(fig, out, "fig6_memory_latency")


# ----------------------------------------------------------------- figure 7
def fig7(df, out, budget=0.1, base="snapkv"):
    """Accuracy change vs. how much the most-starved layer gets (share relative to uniform)."""
    if df.empty or "layer_budgets" not in df:
        return print("  skip fig7: no rows")
    d = df[(df["task"] != "passkey") & (df["budget"] == budget) & (df["method"] != "streaming")]
    if d.empty:
        return print(f"  skip fig7: no rows at budget {budget}")
    pts = []
    for (exp, m), g in d.groupby(["experiment", "method"]):
        if m == base:
            continue
        ref = d[(d["experiment"] == exp) & (d["method"] == base)].set_index(["task", "idx"])["score"]
        j = g.set_index(["task", "idx"])["score"].to_frame("a").join(ref.to_frame("b"), how="inner")
        if len(j) < 20:
            continue
        diff, lo, hi, _ = paired_bootstrap(j["a"], j["b"])
        share = g["layer_budgets"].apply(lambda b: min(b) / (sum(b) / len(b))).mean()
        pts.append((exp, m, share, 100 * diff, 100 * lo, 100 * hi))
    if not pts:
        return print("  skip fig7: nothing to compare")
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    # One label per method, taken from the main experiment (ablation repeats stay unlabelled),
    # except the reversed-entropy variant, which only exists in the ablation run.
    offs = {"entropy:tau=-1": (-22, 14), "entropy": (8, -14), "pyramidkv": (-4, -16),
            "cake_style": (6, 4), "random_alloc": (8, -2)}
    labelled = set()
    for exp, m, x, y, lo, hi in pts:
        base_m = m.split(":")[0]
        g_, mk, _, _, _ = STYLE.get(base_m, ("0.5", "o", "-", 1, m))
        ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt=mk or "o", color=g_, ms=5, capsize=2, lw=0.8,
                    mec="0", mew=0.5)
        main_only = m in ("entropy", "random_alloc", "pyramidkv", "cake_style")
        if m in offs and m not in labelled and (not main_only or exp == "main_qwen3b"):
            labelled.add(m)
            lab = {"entropy:tau=-1": "Reversed (τ=−1)", "random_alloc": "Random"}.get(m, name_of(base_m))
            ax.annotate(lab, (x, y), xytext=offs[m], textcoords="offset points", fontsize=6.5)
    ax.axhline(0, color="0", lw=0.8)
    ax.set_xlabel("Smallest layer budget / uniform share")
    ax.set_ylabel(f"Score change vs SnapKV (points)")
    ax.set_title(f"Starving a layer drives the loss ({budget * 100:g}% budget)")
    save(fig, out, "fig7_layer_starvation")


# ----------------------------------------------------------------- figure 8
DIR_METHODS = [("snapkv", "SnapKV\n(equal)"), ("entropy", "EntropyKV\n(τ = 1)"),
               ("entropy:tau=-1", "Reversed\n(τ = −1)"), ("cake_style", "CAKE-style")]


def fig8(df, out, budget=0.02):
    """Direction test: passkey at 2% of a 4K prompt, both models, 95% bootstrap CI."""
    d = df[df["experiment"].astype(str).str.startswith("passkey_confirm")] if len(df) and "experiment" in df else df
    d = d[d["budget"] == budget] if len(d) else d
    if d.empty:
        return print("  skip fig8: no passkey_confirm rows")
    models = [e for e in ["passkey_confirm_qwen3b", "passkey_confirm_llama3b"] if e in set(d["experiment"])]
    mlabel = {"passkey_confirm_qwen3b": "Qwen2.5-3B", "passkey_confirm_llama3b": "Llama-3.2-3B"}
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    w = 0.8 / len(models)
    rng = np.random.default_rng(0)
    for k, e in enumerate(models):
        for i, (m, _) in enumerate(DIR_METHODS):
            s = d[(d["experiment"] == e) & (d["method"] == m)]["score"].to_numpy()
            if not len(s):
                continue
            boots = rng.choice(s, (2000, len(s))).mean(1)
            mu, lo, hi = 100 * s.mean(), 100 * np.percentile(boots, 2.5), 100 * np.percentile(boots, 97.5)
            ax.bar(i + (k - (len(models) - 1) / 2) * w, mu, w * 0.92, color=["0.25", "0.7"][k],
                   yerr=[[mu - lo], [hi - mu]], capsize=2, error_kw={"lw": 0.8},
                   label=mlabel[e] if i == 0 else None)
    ax.set_xticks(range(len(DIR_METHODS)), [l for _, l in DIR_METHODS], fontsize=7)
    ax.set_ylim(0, 122)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("Passkey accuracy (%)")
    ax.set_title(f"Which layers deserve memory? ({budget * 100:g}% of a 4K prompt)")
    ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=2)
    ax.grid(axis="x", visible=False)
    save(fig, out, "fig8_direction_test")


# ------------------------------------------------------------- Llama table
def table_llama(df, out, exp="llama3b_generalisation"):
    d = lb(df)
    d = d[d["experiment"].astype(str) == exp] if len(d) and "experiment" in d else d.iloc[0:0]
    if d.empty:
        return print("  skip table_llama: no Llama rows")
    rows, sig = [], []
    for b in sorted(x for x in d["budget"].unique() if x < 1.0):
        sub = d[(d["budget"] == b) | (d["method"] == "full")]
        t = sub.groupby(["method", "task"])["score"].mean().unstack("task") * 100
        t.columns = [TASK_LABEL.get(c, c) for c in t.columns]
        t["Average"] = t.mean(axis=1)
        order = [m for m in ORDER if m in t.index] + [m for m in t.index if m not in ORDER]
        t = t.reindex(order)
        t.index = [f"{name_of(m) if m != 'entropy:tau=-1' else 'EntropyKV reversed (τ=−1)'}" for m in t.index]
        t.insert(0, "Budget", f"{b * 100:g}%")
        rows.append(t)
        for c in [m for m in set(d["method"]) if m not in ("full", OURS)]:
            j = _paired(d, OURS, c, b)
            if len(j) < 2:
                continue
            m_, lo, hi, p = paired_bootstrap(j["a"], j["b"])
            sig.append({"budget": b, "baseline": name_of(c) if c != "entropy:tau=-1" else "EntropyKV reversed",
                        "n_pairs": len(j), "diff_x100": 100 * m_, "ci_low": 100 * lo, "ci_high": 100 * hi,
                        "p_boot": p, "significant": bool(lo > 0 or hi < 0)})
    t = pd.concat(rows)
    t.round(2).to_csv(os.path.join(out, "table_llama.csv"))
    with open(os.path.join(out, "table_llama.md"), "w", encoding="utf-8") as f:
        f.write("Llama-3.2-3B-Instruct, LongBench scores (x100), 50 samples per task.\n\n")
        f.write(t.round(2).to_markdown() if hasattr(t, "to_markdown") else t.round(2).to_string())
        f.write("\n")
    if sig:
        pd.DataFrame(sig).round(3).to_csv(os.path.join(out, "table_llama_significance.csv"), index=False)
    print("  wrote table_llama")


# ------------------------------------------------------------------ tables
def tables(df, out, budget=0.2):
    d = lb(df)
    if d.empty:
        return print("  skip tables: no LongBench rows")
    d = d[(d["budget"] == budget) | (d["method"] == "full")]
    t = d.groupby(["method", "task"])["score"].mean().unstack("task") * 100
    t.columns = [TASK_LABEL.get(c, c) for c in t.columns]
    t["Average"] = t.mean(axis=1)
    t = t.reindex([m for m in ORDER if m in t.index] + [m for m in t.index if m not in ORDER])
    t.index = [name_of(m) for m in t.index]
    t.round(2).to_csv(os.path.join(out, "table_main.csv"))
    with open(os.path.join(out, "table_main.md"), "w", encoding="utf-8") as f:
        f.write(f"LongBench scores (x100) at {int(budget * 100)}% KV budget; full cache is the upper bound.\n\n")
        f.write(t.round(2).to_markdown() if hasattr(t, "to_markdown") else t.round(2).to_string())
        f.write("\n")
    print("  wrote table_main")

    sig = []
    dd = lb(df)
    for b in sorted(dd["budget"].unique()):
        if b >= 1.0:
            continue
        for c in [m for m in ORDER if m not in ("full", OURS) and m in set(dd["method"])]:
            j = _paired(dd, OURS, c, b)
            if len(j) < 2:
                continue
            m, lo, hi, p = paired_bootstrap(j["a"], j["b"])
            sig.append({"budget": b, "baseline": name_of(c), "n_pairs": len(j), "diff_x100": 100 * m,
                        "ci_low": 100 * lo, "ci_high": 100 * hi, "p_boot": p,
                        "significant": bool(lo > 0 or hi < 0)})
    if sig:
        pd.DataFrame(sig).round(3).to_csv(os.path.join(out, "table_significance.csv"), index=False)
        print("  wrote table_significance")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/*.jsonl")
    ap.add_argument("--profile", default="results/profile_*.jsonl")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--budget", type=float, default=0.2, help="budget for fig3 and the main table")
    ap.add_argument("--main-experiment", default="main_qwen3b", help="experiment used for Fig 1-3 and the tables")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    setup()
    files = [f for f in glob.glob(a.results) if "profile_" not in os.path.basename(f)]
    df = pd.concat([load(f) for f in files], ignore_index=True) if files else pd.DataFrame()
    if len(df):
        keys = [c for c in ("experiment", "model", "task", "idx", "method", "budget", "length", "depth") if c in df]
        before = len(df)
        df = df.drop_duplicates(subset=keys, keep="first").reset_index(drop=True)
        if len(df) < before:
            print(f"dropped {before - len(df)} duplicate rows")
    prof = load(a.profile) if glob.glob(a.profile) else None
    print(f"{len(df)} result rows, {0 if prof is None else len(prof)} profile rows")
    # Main figures and tables use only the main experiment of the chosen model, so smoke
    # tests (other models, short contexts) and ablations never mix into them.
    main = df
    if len(df) and "experiment" in df:
        main = df[df["experiment"].astype(str) == a.main_experiment]
        if main.empty:
            print(f"no rows for experiment {a.main_experiment!r}; main figures skipped")
    fig1(main, a.out)
    fig2(main, a.out)
    fig3(main, a.out, a.budget)
    fig4(df, a.out)
    fig5(df, a.out)
    fig6(prof, a.out)
    fig7(df[df["experiment"].astype(str).isin([a.main_experiment, "ablation_qwen3b"])] if len(df) else df, a.out)
    fig8(df, a.out)
    tables(main, a.out, a.budget)
    table_llama(df, a.out)


if __name__ == "__main__":
    main()
