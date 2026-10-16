"""Statistics: paired bootstrap confidence intervals and result loading."""

from __future__ import annotations

import glob
import json
from typing import Iterable

import numpy as np


def paired_bootstrap(a: Iterable[float], b: Iterable[float], n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Mean of (a - b) with a percentile bootstrap CI over paired samples.

    Returns (mean_diff, low, high, p_value) where p_value is the two-sided
    bootstrap probability that the difference has the opposite sign.
    """
    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("a and b must be non-empty and paired")
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(min(1.0, p))


def load_results(pattern: str):
    """Load every JSONL result row matching a glob into a pandas DataFrame."""
    import pandas as pd

    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    if not rows:
        raise FileNotFoundError(f"no result rows match {pattern}")
    return pd.DataFrame(rows)
