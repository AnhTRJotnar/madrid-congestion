from typing import Annotated

from loguru import logger
import requests
from tqdm import tqdm
import typer

from madrid_congestion.config import MADRID_BASE_URL, MONTHLY_FILES, RAW_DATA_DIR

app = typer.Typer()
CHUNK = 1 << 20  # 1 MiB


@app.callback()
def cli() -> None:
    """Madrid congestion data pipeline."""


def url_for(month: str) -> str:
    rid = MONTHLY_FILES[month]
    return f"{MADRID_BASE_URL}/{rid}/download/{rid}.zip"


@app.command()
def download(
    months: Annotated[list[str] | None, typer.Argument(help="Months as YYYY-MM.")] = None,
) -> None:
    """Download monthly loop-detector zips into data/raw/."""
    if months is None:
        months = list(MONTHLY_FILES.keys())

    if unknown := [m for m in months if m not in MONTHLY_FILES]:
        raise typer.BadParameter(f"Unknown month(s): {', '.join(unknown)}")

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for month in months:
        out = RAW_DATA_DIR / f"{month}.zip"
        if out.exists():
            logger.info(f"{month}: already downloaded, skipping.")
            continue
        part = out.with_suffix(".zip.part")

        with requests.get(url_for(month), stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0)) or None
            with (
                open(part, "wb") as f,
                tqdm(total=total, unit="B", unit_scale=True, desc=month) as bar,
            ):
                for chunk in r.iter_content(chunk_size=CHUNK):
                    f.write(chunk)
                    bar.update(len(chunk))
        if total is not None and bar.n != total:
            raise ValueError(
                f"Incomplete download for {month}: expected {total} bytes, got {bar.n} bytes."
            )
        part.replace(out)
        logger.success(f"{month}: download complete, saved to {out}.")


if __name__ == "__main__":
    app()
