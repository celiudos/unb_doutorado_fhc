from pathlib import Path

import pytest

from twin_core.calibrate import calibrate_from_responses
from twin_core.io_smartpls import load_item_labels, load_responses

DATA_DIR = Path(__file__).resolve().parents[1] / "dados" / "onda1"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA_DIR


@pytest.fixture(scope="session")
def responses(data_dir):
    return load_responses(data_dir / "respostas_full.csv")


@pytest.fixture(scope="session")
def calibrated(responses, data_dir):
    labels = load_item_labels(data_dir / "cross_loadings.csv")
    model, battery = calibrate_from_responses(responses, name="onda1", item_labels=labels)
    return model, battery
