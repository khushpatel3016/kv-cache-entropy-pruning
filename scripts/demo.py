"""Live viva demo (runs offline once the model is cached; ~1-2 min on a 6 GB GPU).

1. Hides a passkey in a long document and asks for it with: the full cache,
   SnapKV, PyramidKV and EntropyKV at the same small budget.
2. Prints the answer, correctness and KV size for each.
3. Saves a figure of the per-layer budgets EntropyKV chose for two different
   inputs, to show the allocation changes shape with the input while
   PyramidKV's does not.

    python scripts/demo.py --model Qwen/Qwen2.5-1.5B-Instruct --length 4096 --budget 0.1
    python scripts/demo.py --doc my_notes.txt --question "Who wrote the report?"
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kvprune.data.passkey import make_passkey_prompt, passkey_correct  # noqa: E402
from kvprune.engine import PrunedGenerator, load_model  # noqa: E402

METHODS = ["full", "snapkv", "pyramidkv", "entropy"]


def doc_prompt(tok, path, question, max_len):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    ids = tok(text, add_special_tokens=False).input_ids[: max_len - 200]
    msg = tok.decode(ids) + f"\n\nAnswer briefly based on the document above.\nQuestion: {question}"
    if getattr(tok, "chat_template", None):
        out = tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True, tokenize=True)
        return list(out["input_ids"] if hasattr(out, "keys") else out)
    return tok(msg).input_ids


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--length", type=int, default=4096)
    ap.add_argument("--depth", type=float, default=0.5)
    ap.add_argument("--budget", type=float, default=0.1)
    ap.add_argument("--doc", help="optional text file to ask a question about")
    ap.add_argument("--question", default="What is this document about?")
    ap.add_argument("--out", default="figures/demo_budgets.png")
    a = ap.parse_args(argv)

    model, tok = load_model(a.model)
    gen = PrunedGenerator(model, tok)

    ids, key = make_passkey_prompt(tok, a.length, a.depth, seed=42)
    st = gen.prefill(torch.tensor(ids))
    print(f"\nPasskey {key} hidden at {int(a.depth * 100)}% depth of a {st.n}-token prompt; budget {int(a.budget * 100)}%\n")
    print(f"{'method':<12}{'answer':<22}{'correct':<9}{'KV cache':>10}")
    budgets = {}
    for m in METHODS:
        r = gen.run_method(st, m, ratio=None if m == "full" else a.budget, max_new_tokens=12)
        budgets[m] = r.budgets
        ok = "yes" if passkey_correct(r.text, key) else "no"
        print(f"{m:<12}{r.text.strip()[:20]!r:<22}{ok:<9}{r.kv_bytes / 2**20:>8.1f} MB")

    # A second, structurally different input to show the allocation adapts.
    if a.doc:
        ids2 = doc_prompt(tok, a.doc, a.question, a.length)
        label2 = f"your document ({os.path.basename(a.doc)})"
    else:
        ids2, _ = make_passkey_prompt(tok, a.length // 2, 0.05, seed=7)
        label2 = "short prompt, needle near start"
    st2 = gen.prefill(torch.tensor(ids2))
    r2 = gen.run_method(st2, "entropy", ratio=a.budget, max_new_tokens=48)
    if a.doc:
        print(f"\nEntropyKV answer about {label2}: {r2.text.strip()}")

    L = gen.num_layers
    fig, ax = plt.subplots(figsize=(7, 2.6))
    x = torch.arange(L).numpy()
    tot1, tot2 = sum(budgets["entropy"]), sum(r2.budgets)
    ax.plot(x, [b / tot1 * 100 for b in budgets["pyramidkv"]], color="0.6", ls="--", marker="D", ms=3, label="PyramidKV (same for every input)")
    ax.plot(x, [b / tot1 * 100 for b in budgets["entropy"]], color="0", marker="o", ms=3, label="EntropyKV: passkey document")
    ax.plot(x, [b / tot2 * 100 for b in r2.budgets], color="0.3", ls="-.", marker="s", ms=3, label=f"EntropyKV: {label2}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Share of KV budget (%)")
    ax.set_title("Per-layer budget chosen for each input")
    ax.grid(color="0.9")
    ax.legend(frameon=False, fontsize=7)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print(f"\nSaved per-layer budget chart to {a.out}")


if __name__ == "__main__":
    main()
