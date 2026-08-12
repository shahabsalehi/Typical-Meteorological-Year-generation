import numpy as np
import pandas as pd

from tmygen.smoother import smooth_month_edges


def test_smoothing_changes_only_month_interface_rows():
    dt = pd.date_range("2001-01-01", "2001-03-01", freq="h", inclusive="left")
    original = pd.DataFrame({"dt": dt, "value": np.arange(len(dt), dtype=float)})
    # Make the month discontinuity visible.
    original.loc[original.dt.dt.month == 2, "value"] += 1000

    smoothed = smooth_month_edges(original, hours=2)
    seam = original.index[original.dt == pd.Timestamp("2001-02-01 00:00")][0]
    changed = smoothed.index[smoothed.value != original.value].tolist()

    assert set(changed).issubset(set(range(seam - 2, seam + 2)))
    assert smoothed.loc[24, "value"] == original.loc[24, "value"]
