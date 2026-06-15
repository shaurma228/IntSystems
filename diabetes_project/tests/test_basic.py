from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import prepare_data, validate_columns
from src.model import create_pipelines

ROOT = Path(__file__).parents[1]
MODEL_PATH = ROOT / "models" / "model.pkl"
CONFIG_PATH = ROOT / "config" / "config.yaml"

requires_model = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="models/model.pkl не найден — запустите train.py"
)


# 1. Проверка конфигурации
def test_config_exists():
    import yaml
    assert CONFIG_PATH.exists()
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    assert "dataset" in config
    assert "model" in config


# 2. Проверка структуры сохранённой модели
@requires_model
def test_model_structure(model_data):
    assert isinstance(model_data, dict)
    assert "model" in model_data
    assert "feature_names" in model_data
    assert "model_name" in model_data


# 3. Проверка логики очистки данных (замена 0 на NaN)
def test_data_cleaning(small_df):
    zero_cols = ["Glucose", "BloodPressure", "BMI", "Insulin", "SkinThickness"]
    X_train, _, _, _ = prepare_data(small_df, zero_cols, test_size=0.3)

    assert X_train["Glucose"].isnull().any()
    assert X_train["BloodPressure"].isnull().any()
    assert not X_train["Pregnancies"].isnull().any()


# 4. Предсказание на экстремальных и NaN-значениях
@requires_model
def test_prediction_edge_cases(model_data):
    model = model_data["model"]
    features = model_data["feature_names"]

    sample_high = pd.DataFrame([[20, 300, 200, 100, 900, 70, 3.0, 120]], columns=features)
    assert model.predict(sample_high)[0] in [0, 1]

    sample_nan = pd.DataFrame([[1, 100, np.nan, 20, np.nan, 25, 0.5, 30]], columns=features)
    assert model.predict(sample_nan)[0] in [0, 1]


# 5. Структура пайплайнов
def test_pipeline_structure():
    pipelines = create_pipelines()
    assert set(pipelines.keys()) == {"logistic_regression", "random_forest", "gradient_boosting"}
    for name, pipe in pipelines.items():
        assert "imputer" in pipe.named_steps
        assert "model" in pipe.named_steps


# 6. Стратификация при разделении
def test_prepare_data_stratification(small_df):
    X_train, X_test, y_train, y_test = prepare_data(small_df, [], test_size=0.3, random_state=0)
    total = len(small_df)
    assert len(X_train) + len(X_test) == total
    assert len(y_train) + len(y_test) == total


# 7. Валидация колонок
def test_validate_columns_raises():
    df = pd.DataFrame({"A": [1], "B": [2]})
    with pytest.raises(ValueError, match="отсутствуют обязательные колонки"):
        validate_columns(df, ["A", "B", "C"])


def test_validate_columns_ok():
    df = pd.DataFrame({"A": [1], "B": [2], "Outcome": [0]})
    validate_columns(df, ["A", "B", "Outcome"])  # не должно бросать исключение
