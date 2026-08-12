import numpy as np
import pandas as pd

from tmygen.fs_rank import _fs_distance, fs_rank_month


def test_fs_evaluates_different_length_cdfs_at_sample_values():
    sample = np.array([1.0, 2.0])
    reference = np.array([0.0, 1.0, 2.0, 3.0])

    assert _fs_distance(sample, reference) == 0.125


def test_fs_simple():
    dates = (pd.date_range("2000-01-01", periods=3 * 24, freq="h").tolist()
             + pd.date_range("2001-01-01", periods=3 * 24, freq="h").tolist())
    df = pd.DataFrame({"dt": dates})
    df["temp"] = range(len(df))
    df["dni"] = range(len(df))[::-1]
    df["year"] = df.dt.dt.year
    df["month"] = 1
    df["day"] = df.dt.dt.day

    result = fs_rank_month(df, month=1, weights={"temp": 0.5, "dni": 0.5})
    assert len(result) == 2
    assert result[0].fs_score <= result[1].fs_score
