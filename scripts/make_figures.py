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
Tables:
  table_main.csv / .md      task x method at the 20% budget (x100)
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
        ax.plot(range(len(e)), e, color=g, ls=ls, lw=1.4, label=f"{r['task']} #{r['idx']}")
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
    ax.set_xticks(range(len(tasks)), tasks, rotation=30, ha="right")
    ax.set_ylabel("Score gain (points)")
    ax.set_title(f"EntropyKV gain at {int(budget * 100)}% budget (95% CI)")
    ax.legend(frameon=False)
    save(fig, out, "fig3_dynamic_vs_static")


# ----------------------------------------------------------------- figure 4
def fig4(df, out, budget=0.1):
    d = df[df["task"] == "passkey"] if len(df) else df
    if d.empty:
        return print("  skip fig4: no passkey rows")
    d = d[(d["budget"] == budget) | (d["method"] == "full")]
    methods = [m for m in ["full", "snapkv", "pyramidkv", "cake_style", OURS] if m in set(d["method"])]
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
    fig.suptitle(f"Passkey retrieval at {int(budget * 100)}% budget", fontsize=9.5, y=1.08)
    save(fig, out, "fig4_passkey_heatmaps")


# ----------------------------------------------------------------- figure 5
def fig5(df, out):
    d = lb(df)
    if d.empty:
        return print("  skip fig5: no rows")
    d = d[d["experiment"].astype(str).str.startswith("ablation")] if "experiment" in d else d
    if d.empty:
        return print("  skip fig5: no ablation rows")
    per = d.groupby(["experiment", "method", "task"])["score"].mean().reset_index()
    avg = per.groupby(["experiment", "method"])["score"].mean().reset_index()
    avg["label"] = avg.apply(lambda r: r["method"] + ("" if r["experiment"] == "ablation_qwen3b"
                                                      else f" ({r['experiment'].split('_')[1]})"), axis=1)
    avg = avg.sort_values("score")
    fig, ax = plt.subplots(figsize=(3.5, 0.28 * len(avg) + 0.8))
    cols = ["0.0" if l.startswith("entropy") else ("0.75" if "random" in l else "0.5") for l in avg["label"]]
    ax.barh(avg["label"], 100 * avg["score"], color=cols, edgecolor="0", lw=0.5)
    for y, v in enumerate(avg["score"]):
        ax.text(100 * v + 0.2, y, f"{100 * v:.1f}", va="center", fontsize=7)
    ax.set_xlabel("LongBench average score at 20% budget")
    ax.set_title("Ablations (black = entropy variants, light = random control)")
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
    axes[0].set_title(f"Memory at {L // 1024}K context")
    axes[1].set_title(f"Decode latency at {L // 1024}K context")
    axes[1].legend(frameon=False, fontsize=7)
    ovh = g[g["method"] == OURS]["ovh"].mean() if OURS in set(g["method"]) else float("nan")
    fig.text(0.5, -0.12, f"EntropyKV allocation + selection overhead: {ovh:.1f} ms per prompt "
                         f"(one-time, after prefill)", ha="center", fontsize=8)
    save(fig, out, "fig6_memory_latency")


# ------------------------------------------------------------------ tables
def tables(df, out, budget=0.2):
    d = lb(df)
    if d.empty:
        return print("  skip tables: no LongBench rows")
    d = d[(d["budget"] == budget) | (d["method"] == "full")]
    t = d.groupby(["method", "task"])["score"].mean().unstack("task") * 100
    t["Average"] = t.mean(axis=1)
    t = t.reindex([m for m in ORDER if m in t.index] + [m for m in t.index if m not in ORDER])
    t.index = [name_of(m) for m in t.index]
    t.round(2).to_csv(os.path.join(out, "table_main.csv"))
    with open(os.path.join(out, "table_main.md"), "w") as f:
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
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    setup()
    files = [f for f in glob.glob(a.results) if "profile_" not in os.path.basename(f)]
    df = pd.concat([load(f) for f in files], ignore_index=True) if files else pd.DataFrame()
    prof = load(a.profile) if glob.glob(a.profile) else None
    print(f"{len(df)} result rows, {0 if prof is None else len(prof)} profile rows")
    fig1(df, a.out)
    fig2(df, a.out)
    fig3(df, a.out, a.budget)
    fig4(df, a.out)
    fig5(df, a.out)
    fig6(prof, a.out)
    tables(df, a.out, a.budget)


if __name__ == "__main__":
    main()
