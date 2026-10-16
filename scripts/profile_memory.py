"""RQ4: memory and latency of each method, measured in isolation.

For every (context length, method, budget) this does a fresh prefill, builds
the pruned cache, frees the full prefill cache, then decodes a fixed number of
tokens. It records

* kv_mb            size of the retained KV cache;
* resident_mb      GPU memory held after the full cache is freed (weights + pruned cache);
* decode_peak_mb   peak GPU memory during decoding;
* decode_ms_tok    mean decode latency per token;
* overhead_ms      allocation + selection time (includes the entropy-based budget);
* prefill_ms       prefill time (identical for all methods: statistics are computed in it).

    python scripts/profile_memory.py --model Qwen/Qwen2.5-3B-Instruct \
        --lengths 4096 8192 16384 --out results/profile_qwen3b.jsonl
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kvprune.data.passkey import make_passkey_prompt  # noqa: E402
from kvprune.engine import PrunedGenerator, kv_bytes, load_model  # noqa: E402
from kvprune.methods import get_method  # noqa: E402

MB = 1024 ** 2


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192])
    ap.add_argument("--methods", nargs="+", default=["full", "snapkv", "pyramidkv", "entropy", "cake_style"])
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--load-in-4bit", action="store_true")
    a = ap.parse_args(argv)

    cuda = torch.cuda.is_available()
    model, tok = load_model(a.model, load_in_4bit=a.load_in_4bit)
    gen = PrunedGenerator(model, tok)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    with open(a.out, "a") as f:
        for length in a.lengths:
            ids, _ = make_passkey_prompt(tok, length, 0.5, seed=0)
            for name in a.methods:
                m = get_method(name)
                for b in ([1.0] if m.allocation == "none" else a.budgets):
                    for rep in range(a.repeats):
                        gc.collect()
                        if cuda:
                            torch.cuda.empty_cache()
                        st = gen.prefill(torch.tensor(ids))
                        cache, budgets, sel_ms = gen.compress(st, m, ratio=None if m.allocation == "none" else b)
                        logits, n, prefill_ms = st.last_logits, st.n, st.prefill_ms
                        del st
                        gc.collect()
                        resident = peak = None
                        if cuda:
                            torch.cuda.empty_cache()
                            resident = torch.cuda.memory_allocated() / MB
                            torch.cuda.reset_peak_memory_stats()
                        kvb = kv_bytes(cache, gen.num_layers)
                        # Force exactly decode_tokens steps (no EOS stop) for comparable latency.
                        toks, dec_ms = gen.decode(cache, logits, n, a.decode_tokens, eos_ids=[])
                        if cuda:
                            peak = torch.cuda.max_memory_allocated() / MB
                        row = {"model": a.model, "length": n, "method": m.label, "budget": b, "repeat": rep,
                               "kv_mb": kvb / MB, "resident_mb": resident, "decode_peak_mb": peak,
                               "decode_ms_tok": dec_ms / max(1, len(toks) - 1), "overhead_ms": sel_ms,
                               "prefill_ms": prefill_ms, "layer_budgets": budgets,
                               "gpu": torch.cuda.get_device_name(0) if cuda else "cpu"}
                        f.write(json.dumps(row) + "\n")
                        f.flush()
                        print(f"n={n} {m.label}@{b} rep{rep}: kv={row['kv_mb']:.1f}MB "
                              f"dec={row['decode_ms_tok']:.1f}ms/tok overhead={sel_ms:.1f}ms", flush=True)
                        del cache
    print("wrote", a.out)


if __name__ == "__main__":
    main()
