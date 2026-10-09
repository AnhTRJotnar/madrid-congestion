import pytest

from madrid_congestion.config import MONTHLY_FILES
from madrid_congestion.dataset import url_for


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
