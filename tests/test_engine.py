"""Correctness of statistics capture, pruning and uneven-length decoding."""

import copy

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import make_tiny
from kvprune.attention import _repeat_kv
from kvprune.engine import PrunedGenerator, _cache_kv
from kvprune.methods import get_method
from kvprune.selection import select_snapkv

N = 400
W = 16


def _ids(seed=0, n=N):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(4, 128, (1, n), generator=g)


def _forced_logits(model, cache, first_pos, tokens):
    out = []
    pos = first_pos
    for t in tokens:
        p = torch.tensor([[pos]])
        o = model(input_ids=torch.tensor([[t]]), past_key_values=cache, position_ids=p, cache_position=p[0], use_cache=True)
        cache = o.past_key_values
        out.append(o.logits[0, -1].float())
        pos += 1
    return torch.stack(out)


@pytest.mark.parametrize("arch", ["qwen2", "llama"])
def test_full_cache_matches_reference(arch):
    m, cfg = make_tiny(arch)
    ref = make_tiny(arch, attn="sdpa")[0]
    ref.load_state_dict(m.state_dict())
    ids = _ids()
    forced = [5, 9, 17, 33, 65]
    g = PrunedGenerator(m, window=W)
    st = g.prefill(ids)
    cache, budgets, _ = g.compress(st, "full")
    ours = torch.cat([st.last_logits[None], _forced_logits(m, cache, N, forced)])[:-1]
    full_seq = torch.cat([ids, torch.tensor([forced])], dim=1)
    with torch.no_grad():
        theirs = ref(input_ids=full_seq).logits[0, N - 1: N - 1 + len(forced)].float()
    assert budgets == [N] * cfg.num_hidden_layers
    assert torch.allclose(ours, theirs, atol=1e-4, rtol=1e-4)


def test_window_statistics_match_eager():
    m, _ = make_tiny("qwen2")
    eager = make_tiny("qwen2", attn="eager")[0]
    eager.load_state_dict(m.state_dict())
    ids = _ids(1)
    g = PrunedGenerator(m, window=W)
    st = g.prefill(ids)
    with torch.no_grad():
        att = eager(input_ids=ids, output_attentions=True).attentions
    for l, a in enumerate(att):
        win = a[0, :, -W:, :].float()  # [H, W, n]
        exp_scores = win.sum(dim=(0, 1))
        assert torch.allclose(st.stats[l].scores, exp_scores, atol=1e-4)
        vis = torch.arange(N - W, N).float() + 1
        ent = -(win * torch.log(win.clamp_min(1e-30))).sum(-1) / torch.log(vis)
        assert abs(st.stats[l].entropy - ent.clamp(0, 1).mean().item()) < 1e-4
        assert 0.0 <= st.stats[l].entropy <= 1.0


def test_layers_have_different_entropy():
    m, _ = make_tiny("qwen2", layers=6)
    st = PrunedGenerator(m, window=W).prefill(_ids(2))
    assert np.std(st.entropies) > 0.01


@pytest.mark.parametrize("method", ["snapkv", "pyramidkv", "entropy", "cake_style", "streaming", "random_alloc", "random_tokens"])
def test_compress_respects_budgets(method):
    m, cfg = make_tiny("qwen2", layers=6)
    g = PrunedGenerator(m, window=W)
    st = g.prefill(_ids(3))
    cache, budgets, _ = g.compress(st, method, ratio=0.2)
    assert sum(budgets) == int(0.2 * N * cfg.num_hidden_layers)
    for l, b in enumerate(budgets):
        k, v = _cache_kv(cache, l)
        assert k.shape[2] == b and v.shape[2] == b
    toks, _ = g.decode(cache, st.last_logits, st.n, 8)
    assert len(toks) == 8


def test_ratio_one_equals_full():
    m, _ = make_tiny("qwen2")
    g = PrunedGenerator(m, window=W)
    st = g.prefill(_ids(4))
    full = g.run_method(st, "full", max_new_tokens=12)
    snap = g.run_method(st, "snapkv", ratio=1.0, max_new_tokens=12)
    ent = g.run_method(st, "entropy", ratio=1.0, max_new_tokens=12)
    assert full.token_ids == snap.token_ids == ent.token_ids


def test_equal_entropies_reduce_to_snapkv():
    m, cfg = make_tiny("qwen2")
    g = PrunedGenerator(m, window=W)
    st = g.prefill(_ids(5))
    for s in st.stats:
        s.entropy = 0.5
    a = g.run_method(st, "entropy:floor=0", ratio=0.25, max_new_tokens=10)
    b = g.run_method(st, "snapkv", ratio=0.25, max_new_tokens=10)
    assert a.budgets == b.budgets
    assert a.token_ids == b.token_ids


def test_uneven_decode_equals_masked_full_cache():
    """Decoding from per-layer pruned caches of different lengths must equal
    decoding from the full cache with the dropped positions masked out."""
    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

    KEEP = {}

    def masked(module, query, key, value, attention_mask, scaling=None, **kw):
        n_rep = query.shape[1] // key.shape[1]
        k, v = _repeat_kv(key, n_rep), _repeat_kv(value, n_rep)
        if query.shape[2] == 1:
            keep = KEEP[module.layer_idx]
            extra = k.shape[2] - keep.numel()
            mask = torch.cat([keep, torch.ones(extra, dtype=torch.bool)])[None, None, None, :]
            out = F.scaled_dot_product_attention(query, k, v, attn_mask=mask, scale=scaling)
        else:
            out = F.scaled_dot_product_attention(query, k, v, is_causal=True, scale=scaling)
        return out.transpose(1, 2).contiguous(), None

    AttentionInterface.register("masktest", masked)
    AttentionMaskInterface.register("masktest", ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])

    m, cfg = make_tiny("qwen2", layers=5)
    ref = make_tiny("qwen2", layers=5, attn="masktest")[0]
    ref.load_state_dict(m.state_dict())
    ids = _ids(6)
    g = PrunedGenerator(m, window=W)
    st = g.prefill(ids)
    method = get_method("entropy:tau=3")
    pruned, budgets, _ = g.compress(st, method, ratio=0.15)
    assert len(set(budgets)) > 1, "test needs uneven budgets"
    for l, b in enumerate(budgets):
        idx = select_snapkv(st.stats[l].scores, b, W)
        mask = torch.zeros(N, dtype=torch.bool)
        mask[idx] = True
        KEEP[l] = mask
    forced = [7, 11, 13, 19, 23, 29]
    ours = _forced_logits(m, pruned, N, forced)
    with torch.no_grad():
        full_cache = ref(input_ids=ids, use_cache=True).past_key_values
    theirs = _forced_logits(ref, full_cache, N, forced)
    assert torch.allclose(ours, theirs, atol=1e-4, rtol=1e-4)


def test_static_profile_validation():
    m, cfg = make_tiny("qwen2")
    g = PrunedGenerator(m, window=W)
    with pytest.raises(ValueError):
        g.set_static_profile([0.5] * (cfg.num_hidden_layers + 1))
    st = g.prefill(_ids(7))
    with pytest.raises(RuntimeError):
        g.run_method(st, "entropy_static", ratio=0.2)
    g.set_static_profile([0.2, 0.4, 0.6, 0.8])
    r = g.run_method(st, "entropy_static", ratio=0.2, max_new_tokens=4)
    assert r.budgets[0] < r.budgets[-1]
