import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(url):
    """Загрузка данных по URL."""
    return pd.read_csv(url)


def prepare_data(df, zero_columns, test_size=0.2, random_state=42):
    """
    Предобработка данных: замена нулей на NaN в указанных колонках
    и разделение на тренировочную и тестовую выборки.
    """
    df_clean = df.copy()

    # Заменяем нули на NaN там, где они физиологически невозможны
    for col in zero_columns:
        df_clean[col] = df_clean[col].replace(0, np.nan)

    X = df_clean.drop("Outcome", axis=1)
    y = df_clean["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test
