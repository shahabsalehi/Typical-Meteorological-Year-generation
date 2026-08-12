import click
from pathlib import Path
import sys
from .selector import generate_tmy

@click.command()
@click.option("--weather", "-w", type=click.Path(exists=True, dir_okay=False),
              required=True, help="CSV with multi-year hourly weather.")
@click.option("--weights", "-f", type=click.Path(exists=True, dir_okay=False),
              required=True, help="CSV of monthly weight factors.")
@click.option("--out", "-o", type=click.Path(dir_okay=False),
              default="tmy.csv", show_default=True,
              help="Output CSV for the generated TMY.")
@click.option("--manifest", type=click.Path(dir_okay=False), default=None,
              help="Selection manifest CSV (default: <out stem>_manifest.csv).")
@click.option("--top-k", type=click.IntRange(min=1), default=1, show_default=True)
@click.option("--tie-break-var", default="windspeed", show_default=True)
@click.option("--reference-year", type=int, default=2001, show_default=True)
def main(weather, weights, out, manifest, top_k, tie_break_var, reference_year):
    """Generate a Typical Meteorological Year using FS ranking."""
    try:
        result = generate_tmy(
            weather, weights, top_k=top_k, tie_break_var=tie_break_var,
            reference_year=reference_year,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc

    output_path = Path(out)
    manifest_path = Path(manifest) if manifest else output_path.with_name(
        f"{output_path.stem}_manifest.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    result.weather.to_csv(output_path, index=False)
    result.manifest.to_csv(manifest_path, index=False)
    click.echo(f"Wrote {output_path} with {len(result.weather):,} hourly rows.")
    click.echo(f"Wrote selection manifest to {manifest_path}.")


if __name__ == "__main__":
    sys.exit(main())
