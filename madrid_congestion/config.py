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
# Download URL = {MADRID_BASE_URL}/{resource_id}/download/{resource_id}.zip
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


# If tqdm is installed, configure loguru with tqdm.write

try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
