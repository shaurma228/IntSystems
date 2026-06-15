import pickle
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
MODEL_PATH = ROOT / "models" / "model.pkl"
CONFIG_PATH = ROOT / "config" / "config.yaml"


def model_available() -> bool:
    return MODEL_PATH.exists()


@pytest.fixture(scope="session")
def model_data():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@pytest.fixture
def small_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Glucose": [100, 0, 120, 110, 0, 130],
        "BloodPressure": [70, 80, 0, 75, 85, 0],
        "Outcome": [0, 1, 0, 0, 1, 0],
        "BMI": [25, 30, 0, 26, 31, 0],
        "Insulin": [0, 100, 0, 0, 110, 0],
        "SkinThickness": [20, 0, 0, 21, 0, 0],
        "Pregnancies": [1, 5, 0, 2, 6, 0],
        "DiabetesPedigreeFunction": [0.5, 0.1, 0.4, 0.6, 0.2, 0.5],
        "Age": [30, 40, 50, 31, 41, 51],
    })
