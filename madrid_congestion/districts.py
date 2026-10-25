from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

from loguru import logger
import polars as pl
import typer

from madrid_congestion.config import (
    DISTRICT_MINIMUM,
    DST_ROWS,
    EDA_YEARS,
    EXTERNAL_DATA_DIR,
    INTERIM_DATA_DIR,
    LOCATION_FILES,
    MAX_FLOW,
    MIN_SAMPLES,
    PANEL_CUTOFFS,
    PANEL_YEAR,
    PROCESSED_DATA_DIR,
    SAMPLES_FULL,
    SLOT,
    URB,
    VALID_SLOT_PANEL_SHARE,
)

app = typer.Typer()
PANEL_PATH = PROCESSED_DATA_DIR / "panel.parquet"
READING_COLUMNS = ["id", "fecha", "intensidad", "ocupacion", "periodo_integracion"]

# A valid reading: k to 15 samples (more marks a clock-change artefact), both values present, plausible flow.
SAMPLES_OK = pl.col("periodo_integracion").is_between(MIN_SAMPLES, SAMPLES_FULL)
VALUES_PRESENT = pl.col("ocupacion").is_not_null() & pl.col("intensidad").is_not_null()
FLOW_PLAUSIBLE = pl.col("intensidad") <= MAX_FLOW
VALID_READING = SAMPLES_OK & VALUES_PRESENT & FLOW_PLAUSIBLE


def months_of(year: int) -> list[str]:
    # The twelve months of a year as YYYY-MM.
    return [f"{year}-{month:02d}" for month in range(1, 13)]


def urb_rows(rows: pl.LazyFrame) -> pl.LazyFrame:
    # Urban sensors only, without the clock-change rows.
    return rows.filter(pl.col("tipo_elem") == URB, ~pl.col("fecha").is_in(DST_ROWS))


def month_urb_rows(month: str, columns: list[str]) -> pl.DataFrame:
    # One month's urban rows from data/interim/, with only the given columns.
    return urb_rows(pl.scan_parquet(INTERIM_DATA_DIR / f"{month}.parquet")).select(columns).collect()


def by_slot_start(rows: pl.DataFrame, year: int) -> pl.DataFrame:
    # Key rows by slot start (fecha marks the slot's end) and keep the slots that start in `year`.
    # A January file's first row is the year before's last slot; a year's own last slot sits in the
    # next January file, which is never read, so each year misses that one slot (2023 never touches 2024).
    return rows.with_columns(slot_start=pl.col("fecha") - SLOT).filter(pl.col("slot_start").dt.year() == year)


def slot_rows(month: str) -> pl.DataFrame:
    # One month's urban readings keyed by slot start, inside the month's own year.
    return by_slot_start(month_urb_rows(month, READING_COLUMNS), int(month[:4]))


def read_locations(month: str) -> pl.DataFrame:
    # One location snapshot (id, distrito) from data/external/.
    return pl.read_csv(EXTERNAL_DATA_DIR / f"locations-{month}.csv", separator=";", columns=["id", "distrito"])


def combine_locations(latest: pl.DataFrame, earlier: pl.DataFrame) -> pl.DataFrame:
    # Union of two snapshots; where both list an id, the latest snapshot's district wins.
    return pl.concat([latest, earlier.join(latest, on="id", how="anti")]).with_columns(pl.col("id").cast(pl.Int32))


def eligible_districts(shares: pl.DataFrame, c: float, minimum: int) -> set[int]:
    # Districts with at least `minimum` sensors whose availability is at least c.
    return set(
        shares.filter(pl.col("valid_share") >= c)
        .group_by("distrito")
        .agg(sensors=pl.len())
        .filter(pl.col("sensors") >= minimum)["distrito"]
        .to_list()
    )


def panel_cutoff(shares: pl.DataFrame, minimum: int = DISTRICT_MINIMUM, cutoffs: Sequence[float] = PANEL_CUTOFFS) -> float:
    # The highest cut-off c at which every district eligible without a cut-off stays eligible.
    without_cutoff = eligible_districts(shares, 0.0, minimum)
    if not without_cutoff:
        raise ValueError(f"no district has {minimum} sensors even without a cut-off")
    passing = [c for c in cutoffs if eligible_districts(shares, c, minimum) == without_cutoff]
    if not passing:
        raise ValueError(f"no cut-off from {min(cutoffs)} keeps every eligible district")
    return max(passing)


def select_panel(shares: pl.DataFrame, c: float, minimum: int = DISTRICT_MINIMUM) -> pl.DataFrame:
    # The fixed sensor panel: sensors with availability at least c, in districts that keep `minimum` of them.
    kept = shares.filter(pl.col("valid_share") >= c)
    eligible = kept.group_by("distrito").agg(sensors=pl.len()).filter(pl.col("sensors") >= minimum)
    return kept.join(eligible, on="distrito", how="semi").select("id", "distrito", "valid_share").sort("id")


def district_series(
    readings: pl.DataFrame,
    panel: pl.DataFrame,
    slots: pl.DataFrame,
    minimum: int = DISTRICT_MINIMUM,
    panel_share: float = VALID_SLOT_PANEL_SHARE,
) -> pl.DataFrame:
    # Per district and slot start: mean occupancy and flow over the panel sensors with a valid reading.
    # Every slot in `slots` is kept for every district; a slot without readings has 0 sensors and is never filled in.
    panel_sizes = panel.group_by("distrito").agg(panel_sensors=pl.len())
    per_slot = (
        readings.join(panel.select("id", "distrito"), on="id", how="inner")  # panel sensors only, with their district
        .filter(VALID_READING)
        .group_by("distrito", "slot_start")
        .agg(occupancy=pl.col("ocupacion").mean(), flow=pl.col("intensidad").mean(), sensors=pl.len())
    )
    return (
        slots.join(panel_sizes, how="cross")  # every slot x every district
        .join(per_slot, on=["distrito", "slot_start"], how="left")
        .with_columns(pl.col("sensors").fill_null(0))
        .with_columns(
            valid_slot=(pl.col("sensors") >= minimum) & (pl.col("sensors") >= panel_share * pl.col("panel_sensors"))
        )
        .select("distrito", "slot_start", "occupancy", "flow", "sensors", "panel_sensors", "valid_slot")
        .sort("distrito", "slot_start")
    )


def write_parquet(df: pl.DataFrame, path: Path) -> None:
    # Write via a .part file, so only a complete file gets its real name.
    part_path = path.with_suffix(".parquet.part")
    df.write_parquet(part_path)
    part_path.replace(path)


@app.command()
def build_panel() -> None:
    # Choose the fixed sensor panel on PANEL_YEAR and write it to data/processed/panel.parquet.
    counts, year_slots = [], 0
    for month in months_of(PANEL_YEAR):
        rows = slot_rows(month)
        counts.append(rows.group_by("id").agg(valid_readings=VALID_READING.sum()))
        year_slots += rows["slot_start"].n_unique()  # slots with urban data, clock-change rows excluded
        logger.info(f"{month}: done")

    snapshots = sorted(LOCATION_FILES)  # e.g. ["2021-12", "2022-12"]
    districts = combine_locations(read_locations(snapshots[-1]), read_locations(snapshots[0]))
    shares = (
        pl.concat(counts)
        .group_by("id")
        .agg(pl.col("valid_readings").sum())
        .with_columns(valid_share=pl.col("valid_readings") / year_slots)
        .join(districts, on="id", how="inner")
        .filter(pl.col("distrito").is_not_null())  # sensors without a district are dropped
    )
    c = panel_cutoff(shares)
    panel = select_panel(shares, c)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_parquet(panel, PANEL_PATH)
    logger.success(
        f"panel: c = {c}, {panel.height:,} sensors in {panel['distrito'].n_unique()} districts, "
        f"{year_slots:,} slots -> {PANEL_PATH.name}"
    )


@app.command()
def build_district_series(
    years: Annotated[list[int] | None, typer.Argument(help="Years to build (default: 2022 2023).")] = None,
) -> None:
    # Build one district series per year from the fixed panel, into data/processed/district-series-YYYY.parquet.
    years = years or list(EDA_YEARS)
    if locked := [year for year in years if year not in EDA_YEARS]:
        raise typer.BadParameter(f"not allowed before the design freeze: {locked}")  # 2024 is the test year
    if not PANEL_PATH.exists():
        raise FileNotFoundError(f"{PANEL_PATH} not found. Run `build-panel` first.")
    panel = pl.read_parquet(PANEL_PATH)

    for year in years:
        months = []
        for month in months_of(year):
            rows = slot_rows(month)
            months.append(district_series(rows, panel, rows.select("slot_start").unique()))
            logger.info(f"{month}: done")
        series = pl.concat(months).sort("distrito", "slot_start")  # a month's first row may start in the month before
        path = PROCESSED_DATA_DIR / f"district-series-{year}.parquet"
        write_parquet(series, path)
        valid_share = series.select(pl.col("valid_slot").mean()).item()
        logger.success(f"{year}: {series.height:,} district slots, {valid_share:.3f} valid -> {path.name}")


if __name__ == "__main__":
    app()
