from pathlib import Path
from typing import Annotated
import zipfile

from loguru import logger
import polars as pl
import requests
from tqdm import tqdm
import typer

from madrid_congestion.config import (
    EXTERNAL_DATA_DIR,
    INTERIM_DATA_DIR,
    LOCATION_FILES,
    LOCATIONS_BASE_URL,
    MADRID_BASE_URL,
    MONTHLY_FILES,
    RAW_DATA_DIR,
)

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
    # Madrid congestion data pipeline.
    pass


def url_for(month: str) -> str:
    # Download URL for one month, e.g. '2022-01'. Raises KeyError if unknown.
    rid = MONTHLY_FILES[month]
    return f"{MADRID_BASE_URL}/{rid}/download/{rid}.zip"


def read_month_csv(zip_path: Path) -> pl.DataFrame:
    # Read the single CSV inside a monthly zip with RAW_SCHEMA; 'NaN' and '' become null.
    with zipfile.ZipFile(zip_path) as archive:
        csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"Expected exactly one CSV in {zip_path}, found {len(csv_names)}")
        csv_bytes = archive.read(csv_names[0])
    return pl.read_csv(csv_bytes, separator=";", schema=RAW_SCHEMA, null_values=["NaN", ""])


def resolve_months(months: list[str] | None) -> list[str]:
    # Return the requested months, or all months if none are given. Raises typer.BadParameter for unknown months.
    if months is None:
        return list(MONTHLY_FILES.keys())
    if unknown := [m for m in months if m not in MONTHLY_FILES]:
        raise typer.BadParameter(f"Unknown month(s): {', '.join(unknown)}")
    return months


@app.command()
def download(
    months: Annotated[list[str] | None, typer.Argument(help="Months as YYYY-MM.")] = None,
) -> None:
    # Download monthly loop-detector zips into data/raw/.
    months = resolve_months(months)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for month in months:
        zip_path = RAW_DATA_DIR / f"{month}.zip"  # where this month's zip goes
        if zip_path.exists():
            logger.info(f"{month}: already downloaded, skipping.")
            continue  # go to the next month
        download_file(url_for(month), zip_path, month)  # the safe download, shared with download-locations


def download_file(url: str, target: Path, label: str) -> None:
    # Download url to target via a .part file, raise ValueError if the byte count is wrong.
    part_path = target.with_name(target.name + ".part")  # temporary name while downloading
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()  # stop on an HTTP error (e.g. 404)
        expected_bytes = int(response.headers.get("Content-Length", 0)) or None  # size the server announces
        with (
            open(part_path, "wb") as part_file,
            tqdm(total=expected_bytes, unit="B", unit_scale=True, desc=label) as progress,
        ):
            for chunk in response.iter_content(chunk_size=CHUNK):  # 1 MiB at a time
                part_file.write(chunk)
                progress.update(len(chunk))
    if expected_bytes is not None and progress.n != expected_bytes:  # bytes received vs announced
        raise ValueError(
            f"Incomplete download for {label}: expected {expected_bytes} bytes, got {progress.n} bytes."
        )
    part_path.replace(target)  # only a complete file gets its real name
    logger.success(f"{label}: download complete, saved to {target}.")


@app.command()
def download_locations() -> None:
    # Download the two location snapshots (dataset 202468) into data/external/.
    EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)  # create the folder if it is missing
    for month, rid in LOCATION_FILES.items():
        csv_path = EXTERNAL_DATA_DIR / f"locations-{month}.csv"  # e.g. locations-2022-12.csv
        if csv_path.exists():
            logger.info(f"locations {month}: already downloaded, skipping.")
            continue  # go to the next snapshot
        url = f"{LOCATIONS_BASE_URL}/{rid}/download/{rid}.csv"
        download_file(url, csv_path, f"locations {month}")  # the same safe download as the monthly zips


@app.command()
def convert(
    months: Annotated[list[str] | None, typer.Argument(help="Months as YYYY-MM.")] = None,
) -> None:
    # Convert monthly raw zips to typed Parquet files in data/interim/.
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
