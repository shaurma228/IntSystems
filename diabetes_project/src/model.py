import logging

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def create_pipelines(random_state: int = 42, max_iter: int = 1000) -> dict[str, Pipeline]:
    """Создание пайплайнов для трёх моделей-кандидатов."""
    return {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=max_iter, random_state=random_state)),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(random_state=random_state)),
        ]),
        "gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(random_state=random_state)),
        ]),
    }


def tune_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    pipelines: dict[str, Pipeline],
    param_grids: dict[str, dict],
    cv: int = 5,
    scoring: str = "f1",
) -> dict[str, dict]:
    """Подбор гиперпараметров для всех моделей через GridSearchCV."""
    best_models: dict[str, dict] = {}

    for name, pipeline in pipelines.items():
        logger.info("Тюнинг модели: %s", name)
        grid_search = GridSearchCV(
            pipeline,
            param_grids[name],
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)
        best_models[name] = {
            "model": grid_search.best_estimator_,
            "score": grid_search.best_score_,
        }
        logger.info("Лучший %s для %s: %.4f", scoring, name, grid_search.best_score_)

    return best_models
