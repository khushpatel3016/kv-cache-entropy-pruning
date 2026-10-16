"""kvprune: per-layer KV-cache budget allocation driven by attention entropy.

Cross-Layer KV-Cache Pruning for Low-Latency LLM Inference on
Resource-Constrained Hardware (BCSE306L, VIT Chennai).
"""

from .attention import register_kvprune_attention, StatsCollector
from .allocation import (
    allocate_proportional,
    uniform_budgets,
    pyramid_budgets,
    random_budgets,
)
from .methods import METHODS, MethodSpec, get_method
from .engine import PrunedGenerator, load_model

__all__ = [
    "register_kvprune_attention",
    "StatsCollector",
    "allocate_proportional",
    "uniform_budgets",
    "pyramid_budgets",
    "random_budgets",
    "METHODS",
    "MethodSpec",
    "get_method",
    "PrunedGenerator",
    "load_model",
]

__version__ = "1.0.0"
