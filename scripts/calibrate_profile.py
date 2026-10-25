"""Build the frozen per-layer entropy profile used by the `entropy_static` ablation.

The profile is the mean normalised attention entropy of every layer over a
calibration set. By default the calibration prompts come from LongBench tasks
that are NOT in the evaluation set, so the static profile never sees test data.

    python scripts/calibrate_profile.py --model Qwen/Qwen2.5-3B-Instruct \
        --out profiles/qwen2.5-3b.json --per-task 9 --max-len 8192
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kvprune.data.longbench import build_longbench_prompt, load_longbench  # noqa: E402
from kvprune.data.passkey import make_passkey_prompt  # noqa: E402
from kvprune.engine import PrunedGenerator, load_model  # noqa: E402

HELD_OUT_TASKS = ["narrativeqa", "multifieldqa_en", "2wikimqa", "multi_news", "samsum", "passage_count"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", nargs="+", default=HELD_OUT_TASKS)
    ap.add_argument("--per-task", type=int, default=9)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--source", choices=["longbench", "passkey"], default="longbench",
                    help="passkey = offline smoke test only; use longbench for real results")
    ap.add_argument("--load-in-4bit", action="store_true")
    a = ap.parse_args(argv)

    model, tok = load_model(a.model, load_in_4bit=a.load_in_4bit)
    gen = PrunedGenerator(model, tok, window=a.window)
    prompts = []
    if a.source == "longbench":
        for t in a.tasks:
            for s in load_longbench(t, limit=a.per_task):
                prompts.append((t, build_longbench_prompt(s, tok, a.max_len)))
    else:
        for i in range(a.per_task):
            ids, _ = make_passkey_prompt(tok, min(a.max_len, 1024), depth=(i % 5) / 4, seed=i)
            prompts.append(("passkey", ids))

    rows = []
    for i, (task, ids) in enumerate(prompts):
        st = gen.prefill(torch.tensor(ids))
        rows.append(st.entropies)
        print(f"[{i + 1}/{len(prompts)}] {task} n={st.n} mean H={np.mean(st.entropies):.3f}", flush=True)
        del st
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    H = np.array(rows)
    out = {
        "model": a.model,
        "entropies": H.mean(axis=0).tolist(),
        "std": H.std(axis=0).tolist(),
        "n_prompts": len(rows),
        "tasks": sorted({t for t, _ in prompts}),
        "max_len": a.max_len,
        "window": a.window,
        "source": a.source,
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("saved", a.out)


if __name__ == "__main__":
    main()
