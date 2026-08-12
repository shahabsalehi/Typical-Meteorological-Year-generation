from __future__ import annotations

import calendar
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .fs_rank import FSResult, fs_rank_month
from .smoother import smooth_month_edges

__all__ = ["TMYResult", "generate_tmy"]


@dataclass(slots=True)
class TMYResult:
    """Generated hourly weather and the source-month selection manifest."""

    weather: pd.DataFrame
    manifest: pd.DataFrame


def _load_weights(path: Path | str) -> dict[int, dict[str, float]]:
    """Read and validate a 12 x N monthly weight-factor CSV."""
    frame = pd.read_csv(path, index_col=0)
    try:
        frame.index = frame.index.astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError("weight CSV index must contain month numbers 1-12") from exc

    expected = set(range(1, 13))
    if set(frame.index) != expected or frame.index.has_duplicates:
        raise ValueError("weight CSV must contain each month 1-12 exactly once")
    if frame.empty or not len(frame.columns):
        raise ValueError("weight CSV must contain at least one weather variable")

    numeric = frame.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("weights must be finite numeric values with no missing data")
    if (values < 0).any():
        raise ValueError("weights must be non-negative")
    sums = numeric.sum(axis=1)
    bad = sums.index[~np.isclose(sums, 1.0, rtol=0, atol=1e-6)].tolist()
    if bad:
        raise ValueError(f"weights must sum to 1.0 for every month (invalid: {bad})")
    return {month: numeric.loc[month].to_dict() for month in range(1, 13)}


def _load_weather(path: Path | str, variables: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "dt" not in frame:
        raise ValueError("weather CSV must contain a 'dt' timestamp column")
    missing_columns = variables.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"weather CSV is missing variables: {sorted(missing_columns)}")

    try:
        frame["dt"] = pd.to_datetime(frame["dt"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("weather CSV contains invalid timestamps") from exc
    if frame.empty:
        raise ValueError("weather CSV is empty")
    if frame["dt"].isna().any() or frame["dt"].duplicated().any():
        raise ValueError("weather timestamps must be present and unique")
    if not frame["dt"].is_monotonic_increasing:
        raise ValueError("weather timestamps must be in increasing order")
    if len(frame) > 1 and not (frame["dt"].diff().iloc[1:] == pd.Timedelta(hours=1)).all():
        raise ValueError("weather timestamps must form a continuous hourly series")
    if frame.isna().any().any():
        columns = frame.columns[frame.isna().any()].tolist()
        raise ValueError(f"weather CSV contains missing data in columns: {columns}")

    for variable in variables:
        converted = pd.to_numeric(frame[variable], errors="coerce")
        if not np.isfinite(converted.to_numpy(dtype=float)).all():
            raise ValueError(f"weather variable '{variable}' must be finite and numeric")
        frame[variable] = converted

    frame["year"] = frame["dt"].dt.year
    frame["month"] = frame["dt"].dt.month
    frame["day"] = frame["dt"].dt.day
    frame["hour"] = frame["dt"].dt.hour
    return frame


def generate_tmy(
    weather_csv: Path | str,
    weight_csv: Path | str,
    top_k: int = 1,
    tie_break_var: str | None = "windspeed",
    reference_year: int = 2001,
    smooth_hours: int = 8,
) -> TMYResult:
    """Select, stitch, and smooth twelve typical meteorological months.

    FS ranking and the supplied simulation-response-optimized weights determine
    each selected month. ``top_k`` retains the existing optional standard-
    deviation tie-break behavior. Source timestamps are mapped to a non-leap
    reference year so the result is a monotonic 8760-hour series.
    """
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if not isinstance(reference_year, int) or calendar.isleap(reference_year):
        raise ValueError("reference_year must be a non-leap calendar year")

    weights = _load_weights(weight_csv)
    weighted_variables = set().union(*(month_weights for month_weights in weights.values()))
    if top_k > 1 and tie_break_var and tie_break_var not in weighted_variables:
        raise ValueError("tie_break_var must also be present in the weight table")
    weather = _load_weather(weather_csv, weighted_variables)

    # A leap day has no position in the fixed 8760-hour reference calendar.
    weather = weather.loc[~((weather["month"] == 2) & (weather["day"] == 29))].copy()

    chosen: list[FSResult] = []
    manifest_rows: list[dict[str, int | float]] = []
    for month in range(1, 13):
        expected_hours = calendar.monthrange(reference_year, month)[1] * 24
        month_rows = weather.loc[weather["month"] == month]
        complete_years = month_rows.groupby("year").size()
        complete_years = complete_years.index[complete_years == expected_hours]
        candidates = weather.loc[
            (weather["month"] == month) & weather["year"].isin(complete_years)
        ]
        if candidates.empty:
            raise ValueError(f"no complete hourly candidate available for month {month}")
        ranked_all = fs_rank_month(candidates, month, weights[month])
        ranked = ranked_all[:top_k]
        if not ranked:
            raise ValueError(f"no candidate weather months available for month {month}")
        if len(ranked) > 1 and tie_break_var:
            ranked.sort(key=lambda result: result.daily_mean[tie_break_var].std())
        selected = ranked[0]
        chosen.append(selected)
        manifest_rows.append({
            "month": month,
            "source_year": selected.year,
            "fs_score": selected.fs_score,
            "fs_rank": ranked_all.index(selected) + 1,
        })

    parts: list[pd.DataFrame] = []
    for selected in chosen:
        part = weather.loc[
            (weather["year"] == selected.year) & (weather["month"] == selected.month)
        ].copy()
        expected_hours = calendar.monthrange(reference_year, selected.month)[1] * 24
        if len(part) != expected_hours:
            raise ValueError(
                f"selected {selected.year}-{selected.month:02d} has {len(part)} hourly "
                f"rows; expected {expected_hours}"
            )
        start = pd.Timestamp(reference_year, selected.month, 1)
        part["dt"] = pd.date_range(start, periods=expected_hours, freq="h")
        part["year"] = reference_year
        part["month"] = selected.month
        part["day"] = part["dt"].dt.day
        part["hour"] = part["dt"].dt.hour
        parts.append(part)

    output = pd.concat(parts, ignore_index=True)
    if len(output) != 8760 or not output["dt"].is_monotonic_increasing:
        raise RuntimeError("generated TMY is not a monotonic 8760-hour reference year")
    output = smooth_month_edges(output, hours=smooth_hours)
    return TMYResult(weather=output, manifest=pd.DataFrame(manifest_rows))
