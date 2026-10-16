"""Registry of compression methods compared in the study.

A method = an allocation rule (how many tokens each layer keeps) + a
selection rule (which tokens). Methods can be parameterised from a string,
e.g. ``"entropy:tau=2,floor=16"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class MethodSpec:
    name: str
    allocation: str  # none | uniform | pyramid | random | entropy | entropy_static | cake
    selection: str = "snapkv"  # snapkv | streaming | random
    params: dict = field(default_factory=dict)
    description: str = ""

    @property
    def label(self) -> str:
        """Name plus only the parameters that differ from the registered defaults."""
        defaults = METHODS[self.name].params if self.name in METHODS else {}
        diff = {k: v for k, v in self.params.items() if defaults.get(k) != v}
        if not diff:
            return self.name
        return self.name + ":" + ",".join(f"{k}={v}" for k, v in sorted(diff.items()))


METHODS: dict[str, MethodSpec] = {
    "full": MethodSpec("full", "none", description="Full KV cache (upper bound)"),
    "streaming": MethodSpec("streaming", "uniform", "streaming", {"sinks": 4},
                            "StreamingLLM: attention sinks + recent window, uniform budget"),
    "snapkv": MethodSpec("snapkv", "uniform", "snapkv", {},
                         "SnapKV: window scoring, same budget in every layer"),
    "pyramidkv": MethodSpec("pyramidkv", "pyramid", "snapkv", {"beta": 20.0},
                            "PyramidKV: fixed linear (pyramid) per-layer schedule"),
    "random_alloc": MethodSpec("random_alloc", "random", "snapkv", {"floor": 32, "seed": 0},
                               "Control: random per-layer split of the same budget"),
    "random_tokens": MethodSpec("random_tokens", "uniform", "random", {"seed": 0},
                                "Control: uniform budget, random tokens (floor)"),
    "entropy": MethodSpec("entropy", "entropy", "snapkv", {"tau": 1.0, "floor": 32},
                          "Ours: per-input allocation from normalised attention entropy"),
    "entropy_static": MethodSpec("entropy_static", "entropy_static", "snapkv", {"tau": 1.0, "floor": 32},
                                 "Ablation: entropy profile averaged offline, frozen"),
    "cake_style": MethodSpec("cake_style", "cake", "snapkv", {"tau1": 1.0, "tau2": 1.0, "floor": 32},
                             "CAKE allocation rule (entropy x variance) in our framework"),
}


def _coerce(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def get_method(spec: str) -> MethodSpec:
    """Parse ``"name"`` or ``"name:key=val,key=val"`` into a MethodSpec."""
    name, _, rest = spec.partition(":")
    name = name.strip()
    if name not in METHODS:
        raise KeyError(f"unknown method {name!r}; choose from {sorted(METHODS)}")
    base = METHODS[name]
    if not rest:
        return base
    params = dict(base.params)
    for kv in rest.split(","):
        k, _, v = kv.partition("=")
        params[k.strip()] = _coerce(v.strip())
    return replace(base, params=params)
