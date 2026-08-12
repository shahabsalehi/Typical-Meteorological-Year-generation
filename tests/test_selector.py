import pandas as pd

from tmygen import TMYResult, generate_tmy


def test_generate_tmy_returns_8760_hours_and_manifest(tmp_path):
    dt = pd.date_range("1999-01-01", "2001-01-01", freq="h", inclusive="left")
    weather = pd.DataFrame({"dt": dt, "temp": dt.month + dt.hour / 24})
    weather_path = tmp_path / "weather.csv"
    weather.to_csv(weather_path, index=False)

    weights = pd.DataFrame({"month": range(1, 13), "temp": 1.0})
    weights_path = tmp_path / "weights.csv"
    weights.to_csv(weights_path, index=False)

    result = generate_tmy(weather_path, weights_path)

    assert isinstance(result, TMYResult)
    assert len(result.weather) == 8760
    assert result.weather.dt.iloc[0] == pd.Timestamp("2001-01-01 00:00")
    assert result.weather.dt.iloc[-1] == pd.Timestamp("2001-12-31 23:00")
    assert result.weather.dt.is_monotonic_increasing
    assert list(result.manifest.columns) == ["month", "source_year", "fs_score", "fs_rank"]
    assert len(result.manifest) == 12
