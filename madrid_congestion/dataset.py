from pathlib import Path
from typing import Annotated
import zipfile

from loguru import logger
import polars as pl
import requests
from tqdm import tqdm
import typer

from madrid_congestion.config import INTERIM_DATA_DIR, MADRID_BASE_URL, MONTHLY_FILES, RAW_DATA_DIR

app = typer.Typer()
CHUNK = 1 << 20  # 1 MiB
RAW_SCHEMA = {
    "id": pl.Int32,
    "fecha": pl.String,
    "tipo_elem": pl.String,
    "intensidad": pl.Int32,
    "ocupacion": pl.Int16,
    "carga": pl.Int16,
    "vmed": pl.Int16,
    "error": pl.String,
    "periodo_integracion": pl.Int8,
}


@app.callback()
def cli() -> None:
    """Madrid congestion data pipeline."""


def url_for(month: str) -> str:
    """Download URL for one month, e.g. '2022-01'. Raises KeyError if unknown."""
    rid = MONTHLY_FILES[month]
    return f"{MADRID_BASE_URL}/{rid}/download/{rid}.zip"


def read_month_csv(zip_path: Path) -> pl.DataFrame:
    """Read the single CSV inside a monthly zip with RAW_SCHEMA; 'NaN' and '' become null."""
    with zipfile.ZipFile(zip_path) as archive:
        csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"Expected exactly one CSV in {zip_path}, found {len(csv_names)}")
        csv_bytes = archive.read(csv_names[0])
    return pl.read_csv(csv_bytes, separator=";", schema=RAW_SCHEMA, null_values=["NaN", ""])


def resolve_months(months: list[str] | None) -> list[str]:
    """Return the requested months, or all months if none are given. Raises typer.BadParameter for unknown months."""
    if months is None:
        return list(MONTHLY_FILES.keys())
    if unknown := [m for m in months if m not in MONTHLY_FILES]:
        raise typer.BadParameter(f"Unknown month(s): {', '.join(unknown)}")
    return months


@app.command()
def download(
    months: Annotated[list[str] | None, typer.Argument(help="Months as YYYY-MM.")] = None,
) -> None:
    """Download monthly loop-detector zips into data/raw/."""
    months = resolve_months(months)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for month in months:
        zip_path = RAW_DATA_DIR / f"{month}.zip"
        if zip_path.exists():
            logger.info(f"{month}: already downloaded, skipping.")
            continue
        part_path = zip_path.with_suffix(".zip.part")

        with requests.get(url_for(month), stream=True, timeout=60) as response:
            response.raise_for_status()
            expected_bytes = int(response.headers.get("Content-Length", 0)) or None
            with (
                open(part_path, "wb") as part_file,
                tqdm(total=expected_bytes, unit="B", unit_scale=True, desc=month) as progress,
            ):
                for chunk in response.iter_content(chunk_size=CHUNK):
                    part_file.write(chunk)
                    progress.update(len(chunk))
        if expected_bytes is not None and progress.n != expected_bytes:
            raise ValueError(
                f"Incomplete download for {month}: expected {expected_bytes} bytes, got {progress.n} bytes."
            )
        part_path.replace(zip_path)
        logger.success(f"{month}: download complete, saved to {zip_path}.")


@app.command()
def convert(
    months: Annotated[list[str] | None, typer.Argument(help="Months as YYYY-MM.")] = None,
) -> None:
    """Convert monthly raw zips to typed Parquet files in data/interim/."""
    months = resolve_months(months)
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for month in months:
        zip_path = RAW_DATA_DIR / f"{month}.zip"
        parquet_path = INTERIM_DATA_DIR / f"{month}.parquet"
        if parquet_path.exists():
            logger.info(f"{month}: already converted, skipping.")
            continue
        if not zip_path.exists():
            raise FileNotFoundError(f"{zip_path} not found. Run `download {month}` first.")

        df = read_month_csv(zip_path)
        df = df.with_columns(pl.col("fecha").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"))
        part_path = parquet_path.with_suffix(".parquet.part")
        df.write_parquet(part_path)
        part_path.replace(parquet_path)
        logger.success(f"{month}: {df.height:,} rows -> {parquet_path.name}")


if __name__ == "__main__":
    app()
