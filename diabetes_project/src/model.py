from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_pipelines():
    """Создание словаря пайплайнов для разных моделей."""
    pipelines = {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(random_state=42))
        ]),
        "gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(random_state=42))
        ])
    }
    return pipelines


def tune_models(X_train, y_train, pipelines, param_grids, cv=5, scoring="f1"):
    """Подбор гиперпараметров для всех моделей."""
    best_models = {}

    for name, pipeline in pipelines.items():
        print(f"Тюнинг модели: {name}...")
        grid_search = GridSearchCV(
            pipeline,
            param_grids[name],
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_models[name] = {
            "model": grid_search.best_estimator_,
            "score": grid_search.best_score_
        }
        print(f"Лучший {scoring} для {name}: {grid_search.best_score_:.4f}")

    return best_models
