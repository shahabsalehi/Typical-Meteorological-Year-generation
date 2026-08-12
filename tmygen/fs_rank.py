from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass

__all__ = ["fs_rank_month", "FSResult"]


@dataclass(slots=True)
class FSResult:
    """Holds FS statistics for one (month, year) candidate."""
    month: int
    year: int
    fs_score: float      # weighted Σ FS across variables
    daily_mean: pd.DataFrame  # (n_days × vars) daily means


def _fs_distance(sample: np.ndarray, reference: np.ndarray) -> float:
    """
    Finkelstein–Schafer distance between two empirical CDFs.

    The long-term CDF and candidate-month CDF are evaluated at each sorted
    candidate value.  Their input lengths intentionally differ: ``sample``
    contains one month while ``reference`` contains that month from all years.
    """
    sample = np.sort(np.asarray(sample, dtype=float))
    reference = np.sort(np.asarray(reference, dtype=float))
    if sample.ndim != 1 or reference.ndim != 1:
        raise ValueError("FS inputs must be one-dimensional")
    if not len(sample) or not len(reference):
        raise ValueError("FS inputs must not be empty")
    if not np.isfinite(sample).all() or not np.isfinite(reference).all():
        raise ValueError("FS inputs must contain only finite values")

    # ``side='right'`` gives P(X <= x), including tied observations.
    sample_cdf = np.searchsorted(sample, sample, side="right") / len(sample)
    reference_cdf = (
        np.searchsorted(reference, sample, side="right") / len(reference)
    )
    return float(np.mean(np.abs(sample_cdf - reference_cdf)))


def fs_rank_month(df_hourly: pd.DataFrame,
                  month: int,
                  weights: dict[str, float],
                  group_col: str = "year") -> list[FSResult]:
    """
    Computes FS score for *every* year of a given calendar month.

    Parameters
    ----------
    df_hourly: DataFrame with 'year', 'month', 'day' columns plus variables.
    month:     1–12 calendar month to process.
    weights:   mapping {variable: weight}. Must sum to 1.0.
    group_col: column that identifies individual years (default 'year').

    Returns
    -------
    list[FSResult], sorted by ascending fs_score (best first).
    """
    if month not in range(1, 13):
        raise ValueError("month must be between 1 and 12")
    if not weights:
        raise ValueError("weights must not be empty")
    # 1. daily means -------------------------------------------
    vars_ = list(weights)
    daily = (df_hourly.query("month == @month")
                        .groupby([group_col, "day"])[vars_]
                        .mean()
                        .reset_index())

    # 2. reference CDF built from *all* years’ daily means -----
    ref_sorted = {v: np.sort(daily[v].values) for v in vars_}

    if daily.empty:
        raise ValueError(f"no observations available for month {month}")

    results: list[FSResult] = []
    for yr, grp in daily.groupby(group_col, sort=True):
        # grp is n_day × vars
        score = 0.0
        for v, w in weights.items():
            dist = _fs_distance(np.sort(grp[v].values), ref_sorted[v])
            score += w * dist
        results.append(FSResult(month, int(yr), score, grp.set_index("day")))

    return sorted(results, key=lambda r: r.fs_score)
