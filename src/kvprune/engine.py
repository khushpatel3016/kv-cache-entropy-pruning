"""Prefill -> allocate -> prune -> decode.

The key design choice for fair, cheap experiments: one prefill per prompt,
shared by every method. Each method then builds its own pruned copy of the
prefilled cache and decodes greedily. All methods therefore see bit-identical
attention statistics and differ only in what they keep.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch

from .allocation import (
    allocate_proportional,
    cake_weights,
    entropy_weights,
    pyramid_budgets,
    random_budgets,
    total_budget,
    uniform_budgets,
)
from .attention import ATTN_NAME, COLLECTOR, LayerStats, register_kvprune_attention
from .methods import MethodSpec, get_method
from .selection import select_random, select_snapkv, select_streaming


# ---------------------------------------------------------------------------
# Cache helpers (work across transformers 4.4x and 5.x)
# ---------------------------------------------------------------------------

def _cache_kv(cache, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(cache, "layers"):
        lyr = cache.layers[layer]
        return lyr.keys, lyr.values
    return cache.key_cache[layer], cache.value_cache[layer]


def _new_cache(config):
    from transformers import DynamicCache

    try:
        return DynamicCache(config=config)
    except TypeError:
        return DynamicCache()


def kv_bytes(cache, num_layers: int) -> int:
    total = 0
    for l in range(num_layers):
        k, v = _cache_kv(cache, l)
        total += k.numel() * k.element_size() + v.numel() * v.element_size()
    return total


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(name_or_path: str, device: Optional[str] = None, dtype: str = "auto", load_in_4bit: bool = False):
    """Load a causal LM with the kvprune attention backend and its tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    register_kvprune_attention()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype == "auto":
        torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    else:
        torch_dtype = getattr(torch, dtype)
    kwargs = dict(attn_implementation=ATTN_NAME)
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
        )
        kwargs["device_map"] = device
    try:
        model = AutoModelForCausalLM.from_pretrained(name_or_path, dtype=torch_dtype, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(name_or_path, torch_dtype=torch_dtype, **kwargs)
    if not load_in_4bit:
        model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Results containers
# ---------------------------------------------------------------------------

@dataclass
class PrefillState:
    cache: object
    last_logits: torch.Tensor  # [vocab]
    stats: list[LayerStats]
    n: int
    prefill_ms: float

    @property
    def entropies(self) -> list[float]:
        return [s.entropy for s in self.stats]

    @property
    def variances(self) -> list[float]:
        return [s.variance for s in self.stats]


@dataclass
class GenerationResult:
    method: str
    text: str
    token_ids: list[int]
    budgets: list[int]
    kv_bytes: int
    kv_bytes_full: int
    select_ms: float
    decode_ms: float
    new_tokens: int
    peak_mem_bytes: Optional[int] = None
    extra: dict = field(default_factory=dict)

    @property
    def decode_ms_per_token(self) -> float:
        return self.decode_ms / max(1, self.new_tokens)

    @property
    def kv_ratio(self) -> float:
        return self.kv_bytes / max(1, self.kv_bytes_full)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class PrunedGenerator:
    def __init__(
        self,
        model,
        tokenizer=None,
        window: int = 32,
        kernel: int = 7,
        pooling: str = "avg",
        static_profile: Optional[Sequence[float] | str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.window = window
        self.kernel = kernel
        self.pooling = pooling
        cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
        self.num_layers = cfg.num_hidden_layers
        self.device = next(model.parameters()).device
        self.static_profile = None
        if static_profile is not None:
            self.set_static_profile(static_profile)

    # -- static profile -----------------------------------------------------
    def set_static_profile(self, profile: Sequence[float] | str) -> None:
        if isinstance(profile, str):
            with open(profile) as f:
                profile = json.load(f)["entropies"]
        profile = [float(x) for x in profile]
        if len(profile) != self.num_layers:
            raise ValueError(f"profile has {len(profile)} layers, model has {self.num_layers}")
        self.static_profile = profile

    # -- prefill ------------------------------------------------------------
    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor) -> PrefillState:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.shape[0] != 1:
            raise ValueError("batch size 1 only")
        input_ids = input_ids.to(self.device)
        n = input_ids.shape[1]
        COLLECTOR.window = self.window
        COLLECTOR.compute_variance = True
        COLLECTOR.reset()
        COLLECTOR.active = True
        _sync(self.device)
        t0 = time.perf_counter()
        try:
            out = self.model(input_ids=input_ids, use_cache=True)
        finally:
            COLLECTOR.active = False
        _sync(self.device)
        ms = (time.perf_counter() - t0) * 1000
        stats = COLLECTOR.ordered(self.num_layers)
        COLLECTOR.reset()
        last = out.logits[0, -1].float()
        if not torch.isfinite(last).all():
            raise FloatingPointError(
                "non-finite logits after prefill (fp16 overflow?). Load with dtype='bfloat16' "
                "on Ampere+ GPUs or dtype='float32' for small models."
            )
        return PrefillState(out.past_key_values, last, stats, n, ms)

    # -- allocation ---------------------------------------------------------
    def budgets_for(self, state: PrefillState, method: MethodSpec, total: int) -> list[int]:
        L, n, p = self.num_layers, state.n, method.params
        a = method.allocation
        if a == "none":
            return [n] * L
        if a == "uniform":
            return uniform_budgets(L, total, n)
        if a == "pyramid":
            return pyramid_budgets(L, total, n, beta=float(p.get("beta", 20.0)))
        if a == "random":
            return random_budgets(L, total, n, floor=int(p.get("floor", 0)), seed=int(p.get("seed", 0)) + n)
        if a == "entropy":
            w = entropy_weights(state.entropies, tau=float(p.get("tau", 1.0)))
            return allocate_proportional(w, total, n, floor=int(p.get("floor", 0)))
        if a == "entropy_static":
            if self.static_profile is None:
                raise RuntimeError("entropy_static needs a profile: run scripts/calibrate_profile.py")
            w = entropy_weights(self.static_profile, tau=float(p.get("tau", 1.0)))
            return allocate_proportional(w, total, n, floor=int(p.get("floor", 0)))
        if a == "cake":
            w = cake_weights(state.entropies, state.variances, float(p.get("tau1", 1.0)), float(p.get("tau2", 1.0)))
            return allocate_proportional(w, total, n, floor=int(p.get("floor", 0)))
        raise ValueError(f"unknown allocation {a}")

    # -- compression --------------------------------------------------------
    @torch.no_grad()
    def compress(self, state: PrefillState, method: MethodSpec | str, ratio: float | None = None,
                 per_layer: int | None = None):
        if isinstance(method, str):
            method = get_method(method)
        L, n = self.num_layers, state.n
        _sync(self.device)
        t0 = time.perf_counter()
        if method.allocation == "none":
            total = n * L
        else:
            total = total_budget(n, L, ratio=ratio, per_layer=per_layer)
        budgets = self.budgets_for(state, method, total)
        gen = torch.Generator().manual_seed(int(method.params.get("seed", 0)) + n)
        new = _new_cache(self.model.config)
        for l in range(L):
            k, v = _cache_kv(state.cache, l)
            b = budgets[l]
            if b >= n:
                idx = None
            elif method.selection == "streaming":
                idx = select_streaming(n, b, int(method.params.get("sinks", 4)), device=k.device)
            elif method.selection == "random":
                idx = select_random(n, b, self.window, gen, device=k.device)
            else:
                idx = select_snapkv(state.stats[l].scores, b, self.window, self.kernel, self.pooling).to(k.device)
            if idx is None:
                ks, vs = k.clone(), v.clone()
            else:
                ks, vs = k.index_select(2, idx), v.index_select(2, idx)
            new.update(ks, vs, l)
        _sync(self.device)
        select_ms = (time.perf_counter() - t0) * 1000
        return new, budgets, select_ms

    # -- decoding -----------------------------------------------------------
    @torch.no_grad()
    def decode(self, cache, first_logits: torch.Tensor, start_pos: int, max_new_tokens: int,
               eos_ids: Optional[Sequence[int]] = None):
        eos = set(eos_ids or [])
        tokens: list[int] = []
        nxt = int(torch.argmax(first_logits).item())
        _sync(self.device)
        t0 = time.perf_counter()
        pos = start_pos
        for _ in range(max_new_tokens):
            tokens.append(nxt)
            if nxt in eos or len(tokens) >= max_new_tokens:
                break
            ids = torch.tensor([[nxt]], device=self.device)
            p = torch.tensor([[pos]], device=self.device)
            out = self.model(input_ids=ids, past_key_values=cache, position_ids=p,
                             cache_position=p[0], use_cache=True)
            cache = out.past_key_values
            nxt = int(torch.argmax(out.logits[0, -1]).item())
            pos += 1
        _sync(self.device)
        return tokens, (time.perf_counter() - t0) * 1000

    def _eos_ids(self) -> list[int]:
        ids = set()
        gc = getattr(self.model, "generation_config", None)
        if gc is not None and gc.eos_token_id is not None:
            e = gc.eos_token_id
            ids.update(e if isinstance(e, (list, tuple)) else [e])
        if self.tokenizer is not None and self.tokenizer.eos_token_id is not None:
            ids.add(self.tokenizer.eos_token_id)
        return sorted(ids)

    def run_method(self, state: PrefillState, method: MethodSpec | str, ratio: float | None = None,
                   per_layer: int | None = None, max_new_tokens: int = 64, measure_peak: bool = False) -> GenerationResult:
        if isinstance(method, str):
            method = get_method(method)
        if measure_peak and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        cache, budgets, sel_ms = self.compress(state, method, ratio=ratio, per_layer=per_layer)
        kb = kv_bytes(cache, self.num_layers)
        full = kv_bytes(state.cache, self.num_layers)
        toks, dec_ms = self.decode(cache, state.last_logits, state.n, max_new_tokens, self._eos_ids())
        peak = torch.cuda.max_memory_allocated(self.device) if (measure_peak and self.device.type == "cuda") else None
        text = ""
        if self.tokenizer is not None:
            eos = set(self._eos_ids())
            text = self.tokenizer.decode([t for t in toks if t not in eos], skip_special_tokens=True)
        del cache
        return GenerationResult(method.label, text, toks, budgets, kb, full, sel_ms, dec_ms, len(toks), peak)

    def generate(self, input_ids: torch.Tensor, method: MethodSpec | str = "entropy", ratio: float | None = 0.2,
                 per_layer: int | None = None, max_new_tokens: int = 64) -> GenerationResult:
        state = self.prefill(input_ids)
        res = self.run_method(state, method, ratio=ratio if per_layer is None else None,
                              per_layer=per_layer, max_new_tokens=max_new_tokens)
        res.extra["prefill_ms"] = state.prefill_ms
        res.extra["entropies"] = state.entropies
        return res
