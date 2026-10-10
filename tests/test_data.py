import zipfile

import polars as pl
import pytest
import typer

from madrid_congestion.config import MONTHLY_FILES
from madrid_congestion.dataset import read_month_csv, resolve_months, url_for

HEADER = '"id";"fecha";"tipo_elem";"intensidad";"ocupacion";"carga";"vmed";"error";"periodo_integracion"'
ROW = '1001;"2022-01-01 00:00:00";"M30";"408";"NaN";"0";"61";"";5'


def test_url_for_known_month():
    assert (
        url_for("2022-01")
        == "https://datos.madrid.es/dataset/208627-0-transporte-ptomedida-historico/resource/208627-82-transporte-ptomedida-historico-zip/download/208627-82-transporte-ptomedida-historico-zip.zip"
    )


def test_url_for_unknown_month():
    with pytest.raises(KeyError):
        url_for("1999-01")


def test_all_36_months_present():
    assert len(MONTHLY_FILES) == 36


def test_resolve_months_none():
    assert resolve_months(None) == list(MONTHLY_FILES.keys())


def test_resolve_months_known():
    assert resolve_months(["2022-01"]) == ["2022-01"]


def test_resolve_months_unknown():
    with pytest.raises(typer.BadParameter):
        resolve_months(["1999-01"])


def make_zip(path, files: dict[str, str]):
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)


def test_read_month_csv_types_and_nulls(tmp_path):
    zip_path = tmp_path / "test.zip"
    make_zip(zip_path, {"01-2022.csv": HEADER + "\r\n" + ROW + "\r\n"})
    df = read_month_csv(zip_path)
    assert df.schema["intensidad"] == pl.Int32
    assert df["ocupacion"][0] is None
    assert df["error"][0] is None
