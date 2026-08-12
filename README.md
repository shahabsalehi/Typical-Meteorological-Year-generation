# tmygen

`tmygen` generates an 8760-hour Typical Meteorological Year (TMY) from a
continuous multi-year hourly weather record. It implements Shahab Salehi's
research method for selecting typical months with
**simulation-response-optimized weather-variable weights** [Seyed Salehi,
2024]. The supplied weights are used unchanged.

## Method

For each calendar month, the tool:

1. computes daily means for every weighted weather variable;
2. compares each candidate year's empirical CDF with the long-term empirical
   CDF using the Finkelstein-Schafer (FS) statistic;
3. combines the variable FS values using that month's supplied,
   simulation-response-optimized weights;
4. selects the lowest-scoring candidate (or applies the optional standard
   deviation tie-break among the best `top_k` candidates);
5. stitches the 12 source months and smooths only the 11 month interfaces.

The output timestamps are mapped to a coherent non-leap reference year. A
selected leap-year February has February 29 removed, producing exactly 8760
monotonic hourly records. A selection manifest records the source year and FS
score for every month.

This implementation draws on concepts from ISO 15927-4 but contains no
copyrighted content from the standard. The research method and academic
citation remain the basis of the package.

## Input files

The weather CSV must contain:

- `dt`: unique, increasing, uninterrupted hourly timestamps;
- one numeric, complete column for every variable named in the weight file.

The weight CSV uses months `1` through `12` as its first column and weather
variables as the remaining headers. Each row must contain finite, non-negative
weights summing to `1.0`.

```csv
month,temp,windspeed
1,0.8,0.2
2,0.75,0.25
...
12,0.8,0.2
```

## Installation and CLI

```bash
python -m pip install -e .
tmygen --weather multi_year.csv --weights weights.csv --out tmy.csv
```

This writes `tmy.csv` and, by default, `tmy_manifest.csv`. Run `tmygen --help`
for options including a custom manifest path, `top_k`, tie-break variable, and
reference year.

## Python API

```python
from tmygen import generate_tmy

result = generate_tmy("multi_year.csv", "weights.csv")
result.weather.to_csv("tmy.csv", index=False)
print(result.manifest)
```

`result.weather` contains the hourly TMY; `result.manifest` contains one row per
selected month.

## License

MIT. See [LICENSE](LICENSE).
