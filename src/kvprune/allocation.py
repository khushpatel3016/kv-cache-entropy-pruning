"""Per-layer KV budget allocation rules.

Every allocator returns a list of non-negative integers ``b_l`` (tokens kept
in layer ``l``) such that

* ``sum(b_l) == total`` exactly (all methods spend the same memory);
* ``floor <= b_l <= n`` for every layer (no layer starves, none exceeds the
  context length).

Keeping the total identical across methods is what makes the comparison
fair: methods differ only in *where* the memory goes, never in how much.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _largest_remainder(shares: np.ndarray, total: int) -> np.ndarray:
    """Round non-negative real shares to integers summing exactly to ``total``."""
    base = np.floor(shares).astype(np.int64)
    rem = int(total - base.sum())
    if rem > 0:
        order = np.argsort(-(shares - base), kind="stable")
        base[order[:rem]] += 1
    elif rem < 0:
        order = np.argsort(shares - base, kind="stable")
        for i in order:
            if rem == 0:
                break
            if base[i] > 0:
                base[i] -= 1
                rem += 1
    return base


def allocate_proportional(
    weights: Sequence[float],
    total: int,
    n: int,
    floor: int = 0,
) -> list[int]:
    """Split ``total`` tokens across layers in proportion to ``weights``.

    ``b_l = floor + (total - L*floor) * w_l / sum(w)``, then capped at ``n`` with
    the overflow redistributed among uncapped layers, then rounded so the sum
    is exact.
    """
    w = np.asarray(weights, dtype=np.float64)
    L = len(w)
    if L == 0:
        return []
    if np.any(~np.isfinite(w)) or np.any(w < 0):
        raise ValueError(f"weights must be finite and non-negative, got {w}")
    total = int(total)
    floor = int(max(0, floor))
    if total > L * n:
        raise ValueError(f"total budget {total} exceeds L*n = {L * n}")
    floor = min(floor, total // L, n)

    if w.sum() <= 0:
        w = np.ones(L)
    shares = np.full(L, float(floor))
    free = float(total - L * floor)
    capped = np.zeros(L, dtype=bool)
    # Water-filling: distribute, cap at n, repeat with the leftover.
    for _ in range(L + 1):
        active = ~capped
        if free <= 1e-9 or not active.any():
            break
        wa = np.where(active, w, 0.0)
        if wa.sum() <= 0:
            wa = active.astype(np.float64)
        add = free * wa / wa.sum()
        shares = shares + add
        over = shares > n
        free = float((shares[over] - n).sum())
        shares[over] = n
        capped |= over
    b = _largest_remainder(shares, total)
    b = np.clip(b, 0, n)
    # Clipping can only break the sum if rounding pushed a capped layer over n.
    diff = total - int(b.sum())
    i = 0
    while diff != 0 and i < 10 * L:
        j = i % L
        if diff > 0 and b[j] < n:
            b[j] += 1
            diff -= 1
        elif diff < 0 and b[j] > floor:
            b[j] -= 1
            diff += 1
        i += 1
    return [int(x) for x in b]


def uniform_budgets(L: int, total: int, n: int) -> list[int]:
    """Same budget for every layer (SnapKV / H2O / StreamingLLM default)."""
    return allocate_proportional(np.ones(L), total, n, floor=0)


def pyramid_budgets(L: int, total: int, n: int, beta: float = 20.0) -> list[int]:
    """PyramidKV's fixed linear schedule: most cache in layer 0, least in layer L-1.

    Follows the PyramidKV paper: with average budget k = total / L,
    ``min = k / beta``, ``max = 2k - min``, and layer l receives
    ``max - (max - min) * l / (L - 1)``.
    """
    if L == 1:
        return [min(total, n)]
    k = total / L
    b_min = k / beta
    b_max = 2 * k - b_min
    raw = np.array([b_max - (b_max - b_min) * l / (L - 1) for l in range(L)])
    return allocate_proportional(raw, total, n, floor=0)


def random_budgets(L: int, total: int, n: int, floor: int = 0, seed: int = 0) -> list[int]:
    """Control: random split of the same total (Dirichlet(1) weights)."""
    rng = np.random.default_rng(seed)
    return allocate_proportional(rng.dirichlet(np.ones(L)), total, n, floor=floor)


def entropy_weights(entropies: Sequence[float], tau: float = 1.0, eps: float = 1e-6) -> np.ndarray:
    """Preference weights from normalised entropies: ``H_l ** tau``.

    ``tau = 0`` reduces to uniform allocation; larger ``tau`` sharpens it.
    """
    h = np.clip(np.asarray(entropies, dtype=np.float64), eps, 1.0)
    return h ** float(tau)


def cake_weights(
    entropies: Sequence[float],
    variances: Sequence[float],
    tau1: float = 1.0,
    tau2: float = 1.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """CAKE-style preference ``P_l = H_l^(1/tau1) * V_l^(1/tau2)`` (Qin et al., ICLR 2025).

    This reproduces CAKE's *allocation rule* (spatial dispersion x temporal
    shift) inside our framework; it does not include CAKE's cascading memory
    management or its variance-aware eviction indicator.
    """
    h = np.clip(np.asarray(entropies, dtype=np.float64), eps, None)
    v = np.clip(np.asarray(variances, dtype=np.float64), eps, None)
    return h ** (1.0 / tau1) * v ** (1.0 / tau2)


def total_budget(n: int, L: int, ratio: float | None = None, per_layer: int | None = None) -> int:
    """Total tokens kept across all layers, from a ratio of the full cache or a per-layer average."""
    if (ratio is None) == (per_layer is None):
        raise ValueError("give exactly one of ratio or per_layer")
    if ratio is not None:
        if not 0 < ratio <= 1:
            raise ValueError("ratio must be in (0, 1]")
        return int(math.floor(ratio * n * L))
    return int(min(per_layer, n) * L)
