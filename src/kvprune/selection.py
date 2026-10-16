"""Token selection inside one layer, once that layer's budget is known.

All allocation methods share the same selection rule (SnapKV-style window
scoring), so experiments isolate the effect of *how budget is split across
layers*. StreamingLLM keeps its own sink + recent rule as a reference point.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pool_scores(scores: torch.Tensor, kernel: int = 7, mode: str = "avg") -> torch.Tensor:
    """Smooth token scores so neighbouring tokens are kept together (SnapKV)."""
    if kernel <= 1 or scores.numel() < kernel:
        return scores
    x = scores.view(1, 1, -1)
    if mode == "max":
        y = F.max_pool1d(x, kernel, stride=1, padding=kernel // 2)
    else:
        y = F.avg_pool1d(x, kernel, stride=1, padding=kernel // 2, count_include_pad=False)
    return y.view(-1)[: scores.numel()]


def select_snapkv(scores: torch.Tensor, budget: int, window: int, kernel: int = 7, pooling: str = "avg") -> torch.Tensor:
    """Keep the observation window plus the top-scoring prefix tokens."""
    n = scores.numel()
    device = scores.device
    if budget >= n:
        return torch.arange(n, device=device)
    if budget <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    w = min(window, n)
    if budget <= w:
        return torch.arange(n - budget, n, device=device)
    prefix = pool_scores(scores[: n - w].float(), kernel, pooling)
    k = budget - w
    top = torch.topk(prefix, k, largest=True, sorted=False).indices
    keep = torch.cat([top, torch.arange(n - w, n, device=device)])
    return torch.sort(keep).values


def select_streaming(n: int, budget: int, sinks: int = 4, device=None) -> torch.Tensor:
    """StreamingLLM: first ``sinks`` tokens + the most recent ones."""
    if budget >= n:
        return torch.arange(n, device=device)
    if budget <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    s = min(sinks, budget)
    recent = budget - s
    head = torch.arange(s, device=device)
    tail = torch.arange(n - recent, n, device=device) if recent > 0 else torch.empty(0, dtype=torch.long, device=device)
    return torch.cat([head, tail])


def select_random(n: int, budget: int, window: int, generator: torch.Generator | None = None, device=None) -> torch.Tensor:
    """Control: keep the window, fill the rest uniformly at random."""
    if budget >= n:
        return torch.arange(n, device=device)
    w = min(window, n, budget)
    rest = budget - w
    perm = torch.randperm(n - w, generator=generator)[:rest].to(device) if rest > 0 else torch.empty(0, dtype=torch.long, device=device)
    keep = torch.cat([perm, torch.arange(n - w, n, device=device)])
    return torch.sort(keep).values
