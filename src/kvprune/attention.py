"""Custom attention implementation that measures per-layer attention statistics.

We register an attention backend called ``"kvprune"`` with Hugging Face
transformers. It behaves exactly like PyTorch SDPA, but during the prefill
pass it also computes, from an *observation window* of the last ``w`` query
tokens (the SnapKV trick), three statistics per layer:

* ``scores``   [n]   window attention mass received by every key position,
                     summed over heads and window rows (token importance);
* ``entropy``  float length-normalised attention entropy in [0, 1]
                     (how spread out the layer's attention is);
* ``variance`` float attention variance across window rows (CAKE-style
                     "temporal shift" signal, used only by the CAKE-style
                     baseline).

The full ``seq_len x seq_len`` attention matrix is never materialised: we only
form the ``w x seq_len`` block for the window queries, which is what makes
this feasible at 8K-16K tokens on a 16 GB T4 (and 4K-8K on a 6 GB laptop GPU).

During decoding (query length 1) the backend ignores the attention mask so a
layer can attend to however many cached tokens it has kept. This is what
allows each layer to keep a *different* number of tokens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

ATTN_NAME = "kvprune"


@dataclass
class LayerStats:
    scores: torch.Tensor  # [n] float32, on the model device
    entropy: float
    variance: float


@dataclass
class StatsCollector:
    """Holds per-layer statistics captured during one prefill pass."""

    window: int = 32
    active: bool = False
    compute_variance: bool = False
    layers: dict = field(default_factory=dict)

    def reset(self) -> None:
        self.layers = {}

    def ordered(self, num_layers: int) -> list[LayerStats]:
        missing = [i for i in range(num_layers) if i not in self.layers]
        if missing:
            raise RuntimeError(
                f"No attention statistics captured for layers {missing}. Was the model "
                f"loaded with attn_implementation='{ATTN_NAME}'?"
            )
        return [self.layers[i] for i in range(num_layers)]


# A single global collector: one model runs one prompt at a time (batch size 1).
COLLECTOR = StatsCollector()


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


@torch.no_grad()
def window_statistics(
    query: torch.Tensor,
    key: torch.Tensor,
    scaling: float,
    window: int,
    compute_variance: bool = False,
) -> LayerStats:
    """Compute token scores, normalised entropy and variance for one layer.

    Args:
        query: [1, H, n, d] query states after RoPE.
        key:   [1, H_kv, n, d] key states after RoPE (not yet repeated for GQA).
        scaling: softmax scale (1/sqrt(d) normally).
        window: number of trailing query positions to observe.
    """
    if query.shape[0] != 1:
        raise ValueError("kvprune supports batch size 1 only")
    n = key.shape[2]
    w = min(window, n)
    n_heads = query.shape[1]
    k = _repeat_kv(key, n_heads // key.shape[1]).float()  # [1, H, n, d]
    q = query[:, :, -w:, :].float()  # [1, H, w, d]

    logits = torch.matmul(q, k.transpose(-1, -2)) * scaling  # [1, H, w, n]
    # Causal mask inside the window: row i sits at absolute position n - w + i.
    rows = torch.arange(w, device=q.device).unsqueeze(1) + (n - w)
    cols = torch.arange(n, device=q.device).unsqueeze(0)
    future = cols > rows  # [w, n]
    logits = logits.masked_fill(future, float("-inf"))
    attn = torch.softmax(logits, dim=-1)  # [1, H, w, n]

    # Token importance: total attention mass from the window, all heads.
    scores = attn.sum(dim=(0, 1, 2))  # [n]

    # Entropy per (head, row), normalised by log(number of visible keys).
    plogp = torch.where(attn > 0, attn * torch.log(attn.clamp_min(1e-30)), torch.zeros_like(attn))
    ent = -plogp.sum(dim=-1)  # [1, H, w]
    visible = (rows.squeeze(1) + 1).float()  # [w]
    norm = torch.log(visible.clamp_min(2.0))
    ent_norm = (ent / norm).clamp(0.0, 1.0)
    entropy = float(ent_norm.mean().item())

    variance = 0.0
    if compute_variance and n > w:
        # Over the prefix (keys visible to every window row).
        prefix = attn[..., : n - w]  # [1, H, w, n-w]
        variance = float(prefix.var(dim=2, unbiased=False).sum(dim=-1).mean().item())

    return LayerStats(scores=scores, entropy=entropy, variance=variance)


def kvprune_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
):
    """Drop-in replacement for transformers' SDPA attention."""
    q_len = query.shape[2]
    if scaling is None:
        scaling = 1.0 / math.sqrt(query.shape[-1])

    if COLLECTOR.active and q_len > 1:
        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is None:
            raise RuntimeError("attention module has no layer_idx")
        COLLECTOR.layers[layer_idx] = window_statistics(
            query, key, scaling, COLLECTOR.window, COLLECTOR.compute_variance
        )

    n_rep = query.shape[1] // key.shape[1]
    k = _repeat_kv(key, n_rep)
    v = _repeat_kv(value, n_rep)

    if q_len == 1:
        # Decoding: attend to every token this layer kept. Layers may hold
        # different numbers of tokens, so a shared mask would be wrong.
        out = F.scaled_dot_product_attention(query, k, v, attn_mask=None, dropout_p=0.0, scale=scaling)
    else:
        mask = attention_mask
        if mask is not None and mask.dim() == 4:
            mask = mask[:, :, :, : k.shape[-2]]
        use_causal = mask is None and q_len == k.shape[-2]
        if mask is None and not use_causal:
            # Prefill continuing an existing cache: build an explicit causal mask.
            offset = k.shape[-2] - q_len
            r = torch.arange(q_len, device=query.device).unsqueeze(1) + offset
            c = torch.arange(k.shape[-2], device=query.device).unsqueeze(0)
            mask = c <= r
        out = F.scaled_dot_product_attention(
            query, k, v, attn_mask=mask, dropout_p=0.0, is_causal=use_causal, scale=scaling
        )
    return out.transpose(1, 2).contiguous(), None


_REGISTERED = False


def register_kvprune_attention() -> str:
    """Register the backend with transformers (idempotent). Returns its name."""
    global _REGISTERED
    if _REGISTERED:
        return ATTN_NAME
    from transformers import AttentionInterface

    AttentionInterface.register(ATTN_NAME, kvprune_attention_forward)
    try:
        from transformers import AttentionMaskInterface
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

        AttentionMaskInterface.register(ATTN_NAME, ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])
    except Exception:  # older transformers: mask falls back to default
        pass
    _REGISTERED = True
    return ATTN_NAME
