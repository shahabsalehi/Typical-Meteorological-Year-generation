import pandas as pd
import numpy as np

__all__ = ["smooth_month_edges"]


def smooth_month_edges(df: pd.DataFrame,
                       hours: int = 8,
                       var_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Linear ramp across the last/first ``hours`` at each month interface.

    Only the 11 internal TMY month seams are modified. The returned frame is a
    copy, leaving the caller's source data unchanged.
    """
    if hours < 1:
        raise ValueError("hours must be at least 1")
    if "dt" not in df:
        raise ValueError("weather data must contain a 'dt' column")

    df = df.copy()
    if var_cols is None:
        var_cols = [c for c in df.columns
                    if c not in ("dt", "year", "month", "day", "hour")
                    and pd.api.types.is_numeric_dtype(df[c])]
    unknown = set(var_cols).difference(df.columns)
    if unknown:
        raise ValueError(f"unknown smoothing columns: {sorted(unknown)}")
    nonnumeric = [c for c in var_cols
                  if not pd.api.types.is_numeric_dtype(df[c])]
    if nonnumeric:
        raise ValueError(f"smoothing columns must be numeric: {nonnumeric}")

    months = df["dt"].dt.month.to_numpy()
    seam_positions = np.flatnonzero(months[1:] != months[:-1]) + 1
    for seam in seam_positions:
        start, stop = seam - hours, seam + hours
        if start < 0 or stop > len(df):
            raise ValueError("not enough rows around a month seam to smooth")
        for var in var_cols:
            df.loc[df.index[start:stop], var] = np.linspace(
                df.iloc[start][var], df.iloc[stop - 1][var], 2 * hours
            )
    return df
