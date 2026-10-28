"""Experiment runner.

    python -m kvprune.run --config configs/main_qwen3b.yaml

One prefill per sample, then every (method, budget) pair decodes from its own
pruned copy. Rows are appended to a JSONL file as they finish, so a run that
dies (Kaggle timeout, OOM) resumes where it stopped when relaunched.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import time
from typing import Iterator

import torch
import yaml

from .data.longbench import CATEGORY, MAX_NEW_TOKENS, build_longbench_prompt, load_longbench
from .data.passkey import make_passkey_prompt, passkey_correct
from .engine import PrunedGenerator, device_name, load_model
from .methods import get_method
from .metrics import score_longbench


def _key(d: dict) -> tuple:
    return (d.get("task"), d.get("idx"), d.get("method"), d.get("budget"), d.get("length"), d.get("depth"))


def _done_keys(path: str, experiment: str | None = None, model: str | None = None) -> set:
    """Keys of rows already finished: this output file, plus any other JSONL file in
    the same folder holding rows of the same experiment and model (so re-splitting
    tasks across GPUs never repeats work)."""
    keys = set()
    folder = os.path.dirname(path) or "."
    paths = {os.path.abspath(path)}
    if experiment:
        paths |= {os.path.abspath(p) for p in glob.glob(os.path.join(folder, "*.jsonl"))}
    for p in sorted(paths):
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a line cut off by a killed process
                if p == os.path.abspath(path) or (row.get("experiment") == experiment and row.get("model") == model):
                    keys.add(_key(row))
    return keys


def _jobs(cfg: dict, methods, budgets):
    """Yield (method_spec, budget) pairs; 'full' appears once with budget 1.0."""
    for m in methods:
        if m.allocation == "none":
            yield m, 1.0
        else:
            for b in budgets:
                yield m, float(b)


def parse_task_slice(spec: str) -> tuple[str, int | None, int | None]:
    """'gov_report' -> ('gov_report', None, None); 'gov_report[0:68]' -> ('gov_report', 0, 68).

    The slice applies after the config's limit/offset, so a task can be split
    across GPUs without changing which samples are evaluated overall.
    """
    spec = spec.strip()
    if "[" not in spec:
        return spec, None, None
    name, _, rng = spec.partition("[")
    lo, _, hi = rng.rstrip("]").partition(":")
    return name, (int(lo) if lo else None), (int(hi) if hi else None)


def _samples(cfg: dict, tok) -> Iterator[dict]:
    lb = cfg.get("longbench")
    if lb:
        for spec in lb["tasks"]:
            task, lo, hi = parse_task_slice(spec)
            samples = load_longbench(task, limit=lb.get("limit"), offset=lb.get("offset", 0))
            for s in samples[lo:hi]:
                yield {
                    "kind": "longbench", "task": task, "idx": s.idx, "category": CATEGORY[task],
                    "ids": None, "sample": s, "max_new": MAX_NEW_TOKENS[task],
                    "length": None, "depth": None,
                }
    pk = cfg.get("passkey")
    if pk:
        for L in pk["lengths"]:
            for d in pk["depths"]:
                for t in range(pk.get("trials", 10)):
                    yield {
                        "kind": "passkey", "task": "passkey", "idx": t, "category": "passkey",
                        "ids": None, "sample": None, "max_new": 12, "length": int(L), "depth": float(d),
                    }


def run(cfg: dict) -> str:
    out_path = cfg["output"]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    done = _done_keys(out_path, cfg.get("name"), cfg.get("model"))
    methods = [get_method(m) for m in cfg["methods"]]
    budgets = cfg.get("budgets", [0.2])
    jobs = list(_jobs(cfg, methods, budgets))

    model, tok = load_model(cfg["model"], device=cfg.get("device"), dtype=cfg.get("dtype", "auto"),
                            load_in_4bit=cfg.get("load_in_4bit", False))
    gen = PrunedGenerator(model, tok, window=cfg.get("window", 32), kernel=cfg.get("kernel", 7),
                          pooling=cfg.get("pooling", "avg"))
    if any(m.allocation == "entropy_static" for m in methods):
        prof = cfg.get("static_profile")
        if not prof or not os.path.exists(prof):
            raise FileNotFoundError(f"entropy_static needs static_profile; missing {prof!r}. "
                                    "Run scripts/calibrate_profile.py first.")
        gen.set_static_profile(prof)

    env = {"torch": torch.__version__, "python": platform.python_version(),
           "gpu": device_name(model.device.type)}
    max_len = int(cfg.get("max_len", 8192))
    n_written = 0
    t_start = time.time()
    max_hours = cfg.get("max_hours")

    with open(out_path, "a", encoding="utf-8") as fout:
        for s in _samples(cfg, tok):
            base = {"task": s["task"], "idx": s["idx"], "length": s["length"], "depth": s["depth"]}
            todo = [(m, b) for (m, b) in jobs if _key({**base, "method": m.label, "budget": b}) not in done]
            if not todo:
                continue
            if max_hours is not None and (time.time() - t_start) / 3600 >= float(max_hours):
                print(f"time limit of {float(max_hours):.2f} h reached; stopping cleanly. "
                      f"Rerun the same command to continue from here.", flush=True)
                break
            key = None
            if s["kind"] == "longbench":
                ids = build_longbench_prompt(s["sample"], tok, max_len)
            else:
                ids, key = make_passkey_prompt(tok, s["length"], s["depth"], seed=1000 * s["idx"] + int(100 * s["depth"]))
            try:
                state = gen.prefill(torch.tensor(ids))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"[OOM] prefill {s['task']}#{s['idx']} n={len(ids)}; skipped")
                continue
            for m, b in todo:
                try:
                    r = gen.run_method(state, m, ratio=None if m.allocation == "none" else b,
                                       max_new_tokens=s["max_new"])
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"[OOM] {m.label}@{b} {s['task']}#{s['idx']}; skipped")
                    continue
                if s["kind"] == "longbench":
                    smp = s["sample"]
                    score = score_longbench(s["task"], r.text, smp.answers, smp.all_classes)
                    answers = smp.answers
                else:
                    score = passkey_correct(r.text, key)
                    answers = [key]
                row = {
                    "experiment": cfg.get("name", ""), "model": cfg["model"], **base,
                    "category": s["category"], "n_tokens": state.n, "method": m.label,
                    "method_base": m.name, "budget": b, "score": score, "pred": r.text,
                    "answers": answers, "kv_bytes": r.kv_bytes, "kv_bytes_full": r.kv_bytes_full,
                    "kv_ratio": r.kv_ratio, "prefill_ms": state.prefill_ms, "select_ms": r.select_ms,
                    "decode_ms_per_token": r.decode_ms_per_token, "new_tokens": r.new_tokens,
                    "layer_budgets": r.budgets, "entropies": state.entropies, "env": env,
                }
                fout.write(json.dumps(row) + "\n")
                fout.flush()
                n_written += 1
            del state
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            el = time.time() - t_start
            print(f"{s['task']}#{s['idx']} n={len(ids)} rows={n_written} elapsed={el/60:.1f} min", flush=True)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, help="override longbench.limit (quick runs)")
    ap.add_argument("--tasks", nargs="+", help="override longbench.tasks")
    ap.add_argument("--output", help="override output path")
    ap.add_argument("--budgets", nargs="+", type=float, help="override budgets, e.g. --budgets 0.2")
    ap.add_argument("--methods", nargs="+", help="override methods")
    ap.add_argument("--max-hours", type=float, help="stop cleanly (before the next sample) after this many hours")
    a = ap.parse_args(argv)
    with open(a.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if a.budgets:
        cfg["budgets"] = a.budgets
    if a.methods:
        cfg["methods"] = a.methods
    if a.max_hours is not None:
        cfg["max_hours"] = a.max_hours
    if a.limit is not None and cfg.get("longbench"):
        cfg["longbench"]["limit"] = a.limit
    if a.tasks and cfg.get("longbench"):
        cfg["longbench"]["tasks"] = a.tasks
    if a.output:
        cfg["output"] = a.output
    print("wrote", run(cfg))


if __name__ == "__main__":
    main()
