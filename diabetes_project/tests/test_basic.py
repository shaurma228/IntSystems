import os
import pickle

import numpy as np
import pandas as pd
import yaml

from src.data import prepare_data

MODEL_PATH = "model.pkl"
CONFIG_PATH = "config.yaml"


# 1. Проверка конфигурации
def test_config_exists():
    assert os.path.exists(CONFIG_PATH)
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    assert "dataset" in config
    assert "model" in config


# 2. Проверка загрузки модели и метаданных
def test_model_structure():
    assert os.path.exists(MODEL_PATH)
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)

    assert isinstance(data, dict)
    assert "model" in data
    assert "feature_names" in data
    assert "model_name" in data


# 3. Проверка логики очистки данных (замена 0 на NaN)
def test_data_cleaning():
    test_df = pd.DataFrame({
        "Glucose": [100, 0, 120, 110, 0, 130],
        "BloodPressure": [70, 80, 0, 75, 85, 0],
        "Outcome": [0, 1, 0, 0, 1, 0],
        "BMI": [25, 30, 0, 26, 31, 0],
        "Insulin": [0, 100, 0, 0, 110, 0],
        "SkinThickness": [20, 0, 0, 21, 0, 0],
        "Pregnancies": [1, 5, 0, 2, 6, 0],
        "DiabetesPedigreeFunction": [0.5, 0.1, 0.4, 0.6, 0.2, 0.5],
        "Age": [30, 40, 50, 31, 41, 51]
    })

    zero_cols = ["Glucose", "BloodPressure", "BMI", "Insulin", "SkinThickness"]
    X_train, _, _, _ = prepare_data(test_df, zero_cols, test_size=0.3)

    # В X_train нули должны стать NaN
    assert X_train["Glucose"].isnull().any()
    assert X_train["BloodPressure"].isnull().any()
    # Pregnancies не в списке zero_cols, 0 должен остаться 0 (или не быть NaN)
    assert not X_train["Pregnancies"].isnull().any()


# 4. Проверка предсказания на экстремальных значениях
def test_prediction_edge_cases():
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    model = data["model"]
    features = data["feature_names"]

    # Очень высокие значения
    sample_high = pd.DataFrame([[20, 300, 200, 100, 900, 70, 3.0, 120]], columns=features)
    pred_high = model.predict(sample_high)
    assert pred_high[0] in [0, 1]

    # Значения с NaN (которые должен обработать Imputer в пайплайне)
    sample_nan = pd.DataFrame([[1, 100, np.nan, 20, np.nan, 25, 0.5, 30]], columns=features)
    pred_nan = model.predict(sample_nan)
    assert pred_nan[0] in [0, 1]
