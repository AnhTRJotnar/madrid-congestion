from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Source: datos.madrid.es dataset 208627 (Histórico de datos del tráfico), resource ids from
# https://datos.madrid.es/api/3/action/package_show?id=208627-0-transporte-ptomedida-historico
MADRID_BASE_URL = (
    "https://datos.madrid.es/dataset/208627-0-transporte-ptomedida-historico/resource"
)


MONTHLY_FILES: dict[str, str] = {
    "2022-01": "208627-82-transporte-ptomedida-historico-zip",
    "2022-02": "208627-68-transporte-ptomedida-historico-zip",
    "2022-03": "208627-124-transporte-ptomedida-historico-zip",
    "2022-04": "208627-83-transporte-ptomedida-historico-zip",
    "2022-05": "208627-50-transporte-ptomedida-historico-zip",
    "2022-06": "208627-137-transporte-ptomedida-historico-zip",
    "2022-07": "208627-114-transporte-ptomedida-historico-zip",
    "2022-08": "208627-102-transporte-ptomedida-historico-zip",
    "2022-09": "208627-24-transporte-ptomedida-historico-zip",
    "2022-10": "208627-101-transporte-ptomedida-historico-zip",
    "2022-11": "208627-10-transporte-ptomedida-historico-zip",
    "2022-12": "208627-0-transporte-ptomedida-historico-zip",
    "2023-01": "208627-74-transporte-ptomedida-historico-zip",
    "2023-02": "208627-138-transporte-ptomedida-historico-zip",
    "2023-03": "208627-61-transporte-ptomedida-historico-zip",
    "2023-04": "208627-55-transporte-ptomedida-historico-zip",
    "2023-05": "208627-139-transporte-ptomedida-historico-zip",
    "2023-06": "208627-140-transporte-ptomedida-historico-zip",
    "2023-07": "208627-37-transporte-ptomedida-historico-zip",
    "2023-08": "208627-30-transporte-ptomedida-historico-zip",
    "2023-09": "208627-23-transporte-ptomedida-historico-zip",
    "2023-10": "208627-84-transporte-ptomedida-historico-zip",
    "2023-11": "208627-125-transporte-ptomedida-historico-zip",
    "2023-12": "208627-2-transporte-ptomedida-historico-zip",
    "2024-01": "208627-141-transporte-ptomedida-historico-zip",
    "2024-02": "208627-67-transporte-ptomedida-historico-zip",
    "2024-03": "208627-60-transporte-ptomedida-historico-zip",
    "2024-04": "208627-54-transporte-ptomedida-historico-zip",
    "2024-05": "208627-49-transporte-ptomedida-historico-zip",
    "2024-06": "208627-43-transporte-ptomedida-historico-zip",
    "2024-07": "208627-36-transporte-ptomedida-historico-zip",
    "2024-08": "208627-29-transporte-ptomedida-historico-zip",
    "2024-09": "208627-85-transporte-ptomedida-historico-zip",
    "2024-10": "208627-16-transporte-ptomedida-historico-zip",
    "2024-11": "208627-9-transporte-ptomedida-historico-zip",
    "2024-12": "208627-1-transporte-ptomedida-historico-zip",
}

# Source: datos.madrid.es dataset 202468
LOCATIONS_BASE_URL = "https://datos.madrid.es/dataset/202468-0-intensidad-trafico/resource"  # base of the download links

LOCATION_FILES: dict[str, str] = {  # snapshot month -> resource id (Location rules confirmed: union of both)
    "2021-12": "202468-3-intensidad-trafico-csv",  # December 2021 snapshot
    "2022-12": "202468-4-intensidad-trafico-csv",
}

# Data rules (EDA and cleaning)
EDA_YEARS = (2022, 2023)  # 2024 is used to test
URB = "URB"  # tipo_elem value for urban points (URB only)
SLOT = timedelta(minutes=15)  # one slot, fecha marks its END, so slot_start = fecha - SLOT
SAMPLES_FULL = 15  # one-minute samples in a complete URB slot

# fecha values removed by date and time (Slot start confirmed):
# spring = the 03:00 row (1 sample); autumn = the four rows that merge two real slots.
DST_ROWS: tuple[datetime, ...] = (
    # 2022 spring
    datetime(2022, 3, 27, 3, 0),
    # 2022 autumn
    datetime(2022, 10, 30, 2, 15),
    datetime(2022, 10, 30, 2, 30),
    datetime(2022, 10, 30, 2, 45),
    datetime(2022, 10, 30, 3, 0),
    # 2023 spring
    datetime(2023, 3, 26, 3, 0),
    # 2023 autumn
    datetime(2023, 10, 29, 2, 15),
    datetime(2023, 10, 29, 2, 30),
    datetime(2023, 10, 29, 2, 45),
    datetime(2023, 10, 29, 3, 0),
    # 2024 spring
    datetime(2024, 3, 31, 3, 0),
    # 2024 autumn
    datetime(2024, 10, 27, 2, 15),
    datetime(2024, 10, 27, 2, 30),
    datetime(2024, 10, 27, 2, 45),
    datetime(2024, 10, 27, 3, 0),
)

MIN_SAMPLES = 15  # k: a valid reading has MIN_SAMPLES to SAMPLES_FULL one-minute samples
MAX_FLOW = 5000  # value rule: a reading with flow above this (vehicles per hour) is invalid
DISTRICT_MINIMUM = 30  # panel sensors a district needs; evidence: subsampling test, notebook 1.0 Section 6
PANEL_YEAR = 2022  # the fixed sensor panel is chosen on the training year only
PANEL_CUTOFFS: tuple[float, ...] = tuple(round(0.50 + 0.01 * step, 2) for step in range(51))  # candidate c, 0.50 to 1.00
VALID_SLOT_PANEL_SHARE = 0.5  # valid slot: at least DISTRICT_MINIMUM sensors AND at least this share of the panel report

# Label rules: the operational congestion proxy (persistent high district occupancy), fixed on PANEL_YEAR
SUPPORT_SHARE = 0.5  # a district is studied only if at least this share of its 2022 slots is valid
HIGH_OCC_QUANTILE = 0.95  # high-occupancy threshold and flow reference: the district's 2022 95th percentile
MIN_EPISODE_SLOTS = 2  # a persistent high-occupancy episode lasts at least 2 slots (30 minutes)
CALM_SLOTS = 4  # calm origin: none of t-3 ... t is a high-occupancy slot (one hour)
HORIZONS = {30: 2, 60: 4}  # warning horizon in minutes -> slots in the target window t+1 ... t+H
WEEKEND_FROM = 6  # polars weekday: Monday = 1 ... Saturday = 6, Sunday = 7; Saturday and Sunday are the weekend
# Calendar baseline: the 2022 positive share of usable calm origins per district x weekday/weekend x slot of day


# If tqdm is installed, configure loguru with tqdm.write

try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
