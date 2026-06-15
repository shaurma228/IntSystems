import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]


def load_data(url: str, local_cache: str | None = None) -> pd.DataFrame:
    """Загрузка данных по URL с опциональным кешированием на диск."""
    cache_path = Path(local_cache) if local_cache else None

    if cache_path and cache_path.exists():
        logger.info("Загрузка из кеша: %s", cache_path)
        return pd.read_csv(cache_path)

    try:
        logger.info("Загрузка из сети: %s", url)
        df = pd.read_csv(url)
    except Exception as exc:
        raise RuntimeError(
            f"Не удалось загрузить датасет по URL '{url}'. "
            "Проверьте подключение к интернету."
        ) from exc

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        logger.info("Датасет сохранён в кеш: %s", cache_path)

    return df


def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Проверка наличия обязательных колонок в датасете."""
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"В датасете отсутствуют обязательные колонки: {sorted(missing)}. "
            f"Доступные колонки: {list(df.columns)}"
        )


def prepare_data(
    df: pd.DataFrame,
    zero_columns: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Предобработка: замена нулей на NaN и разделение на train/test."""
    df_clean = df.copy()

    for col in zero_columns:
        df_clean[col] = df_clean[col].replace(0, np.nan)

    X = df_clean.drop("Outcome", axis=1)
    y = df_clean["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test
