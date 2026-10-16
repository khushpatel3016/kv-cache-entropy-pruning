import numpy as np
import pytest

from kvprune.allocation import (
    allocate_proportional,
    cake_weights,
    entropy_weights,
    pyramid_budgets,
    random_budgets,
    total_budget,
    uniform_budgets,
)


@pytest.mark.parametrize("seed", range(25))
def test_proportional_invariants(seed):
    rng = np.random.default_rng(seed)
    L = int(rng.integers(1, 40))
    n = int(rng.integers(50, 5000))
    total = int(rng.integers(L, L * n + 1))
    floor = int(rng.integers(0, 64))
    w = rng.random(L) ** 3
    b = allocate_proportional(w, total, n, floor)
    assert sum(b) == total
    assert all(0 <= x <= n for x in b)
    assert min(b) >= min(floor, total // L, n)


def test_equal_weights_give_equal_budgets():
    assert allocate_proportional([0.7] * 8, 800, 1000) == [100] * 8
    assert uniform_budgets(8, 800, 1000) == [100] * 8


def test_tau_zero_is_uniform():
    w = entropy_weights([0.1, 0.5, 0.9, 0.3], tau=0.0)
    assert allocate_proportional(w, 400, 1000) == [100] * 4


def test_higher_entropy_gets_more():
    w = entropy_weights([0.2, 0.4, 0.8], tau=1.0)
    b = allocate_proportional(w, 700, 1000, floor=0)
    assert b[0] < b[1] < b[2]
    assert b == [100, 200, 400]


def test_cap_redistributes_overflow():
    b = allocate_proportional([1, 0, 0, 0], 300, 100, floor=0)
    assert sum(b) == 300 and max(b) == 100


def test_pyramid_monotone_and_exact():
    b = pyramid_budgets(28, 28 * 512, 8192)
    assert sum(b) == 28 * 512
    assert all(b[i] >= b[i + 1] for i in range(27))
    assert b[0] > b[-1]


def test_random_reproducible():
    assert random_budgets(10, 1000, 500, seed=3) == random_budgets(10, 1000, 500, seed=3)
    assert sum(random_budgets(10, 1000, 500, floor=32, seed=4)) == 1000


def test_cake_weights_positive():
    w = cake_weights([0.5, 0.6], [0.0, 1.0])
    assert np.all(w > 0)


def test_total_budget():
    assert total_budget(1000, 10, ratio=0.2) == 2000
    assert total_budget(1000, 10, per_layer=128) == 1280
    with pytest.raises(ValueError):
        total_budget(1000, 10)
