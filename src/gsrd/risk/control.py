from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import beta


@dataclass(frozen=True)
class ThresholdStats:
    threshold: float
    num_kept: int
    num_errors: int
    risk: float
    coverage: float
    ucb_risk: float


def hoeffding_ucb(error_rate: float, n: int, delta: float) -> float:
    if n <= 0:
        return 1.0
    return min(1.0, error_rate + math.sqrt(math.log(1.0 / delta) / (2.0 * n)))


def clopper_pearson_ucb(num_errors: int, n: int, delta: float) -> float:
    if n <= 0:
        return 1.0
    if num_errors >= n:
        return 1.0
    # One-sided upper confidence bound for Bernoulli risk.
    return float(beta.ppf(1.0 - delta, num_errors + 1, n - num_errors))


def enforce_monotone_nonincreasing(risks: list[float]) -> list[float]:
    arr = np.asarray(risks, dtype=float)
    fixed = np.maximum.accumulate(arr[::-1])[::-1]
    return [float(x) for x in fixed]


def select_threshold(
    stats: list[ThresholdStats],
    alpha: float,
) -> ThresholdStats:
    feasible = [item for item in stats if item.ucb_risk <= alpha]
    if not feasible:
        return max(stats, key=lambda x: x.threshold)
    return min(feasible, key=lambda x: x.threshold)
