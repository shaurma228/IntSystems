# -*- coding: utf-8 -*-
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError as error:
    print("Не установлена библиотека imbalanced-learn.")
    print("Установите ее командой: pip install imbalanced-learn")
    raise SystemExit(error)


RANDOM_STATE = 42
TEST_SIZE = 0.35
RARE_CLASS_TRAIN_LIMIT = 8

# Те же базовые настройки RandomForest, которые зафиксированы в 3MO.py.
BASE_RF_PARAMS = {
    "n_estimators": 30,
    "max_depth": 2,
    "min_samples_leaf": 4,
    "random_state": RANDOM_STATE,
}


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def show_class_distribution(y: np.ndarray, target_names: list[str]) -> int:
    counts = np.bincount(y, minlength=len(target_names))
    for class_id, count in enumerate(counts):
        share = count / len(y)
        print(f"{target_names[class_id]}: {count} объектов ({share:.2%})")
    return int(np.argmin(counts))


def make_educational_imbalanced_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    rare_class: int,
    rare_limit: int = RARE_CLASS_TRAIN_LIMIT,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_STATE)
    rare_indices = np.where(y_train == rare_class)[0]
    other_indices = np.where(y_train != rare_class)[0]

    selected_rare_indices = rng.choice(
        rare_indices,
        size=min(rare_limit, len(rare_indices)),
        replace=False,
    )
    selected_indices = np.concatenate([other_indices, selected_rare_indices])
    rng.shuffle(selected_indices)

    return X_train[selected_indices], y_train[selected_indices]


def evaluate_model(
    model_name: str,
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    target_names: list[str],
) -> float:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred, average="macro")

    print_section(model_name)
    print(f"Macro-F1 на test: {score:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=target_names,
            digits=4,
            zero_division=0,
        )
    )

    return score


def build_baseline_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    **BASE_RF_PARAMS,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def build_smote_pipeline() -> ImbPipeline:
    return ImbPipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=1)),
            (
                "classifier",
                RandomForestClassifier(
                    **BASE_RF_PARAMS,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def plot_learning_curves(model, X_train, y_train) -> None:
    print_section("Кривые обучения")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator=model,
        X=X_train,
        y=y_train,
        cv=cv,
        scoring="f1_macro",
        train_sizes=np.linspace(0.5, 1.0, 5),
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    validation_mean = validation_scores.mean(axis=1)
    validation_std = validation_scores.std(axis=1)

    for size, train_score, validation_score in zip(
        train_sizes,
        train_mean,
        validation_mean,
    ):
        print(
            f"Размер train={size:>3}: "
            f"train macro-F1={train_score:.4f}, "
            f"validation macro-F1={validation_score:.4f}"
        )

    gap = train_mean[-1] - validation_mean[-1]
    validation_growth = validation_mean[-1] - validation_mean[0]

    print("\nАнализ графика:")
    if gap > 0.10:
        print(
            "Есть признаки переобучения: train заметно выше validation. "
            "Можно ограничить глубину деревьев, увеличить min_samples_leaf или собрать больше данных."
        )
    else:
        print("Сильного переобучения не видно: train и validation близки.")

    if validation_growth > 0.03:
        print(
            "Validation-кривая растет при увеличении объема train, значит дополнительные данные "
            "могут улучшить модель."
        )
    else:
        print(
            "Validation-кривая почти не растет, значит модель близка к своему текущему потолку "
            "на этих признаках."
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_mean, marker="o", label="Train macro-F1")
    ax.plot(train_sizes, validation_mean, marker="o", label="Validation macro-F1")
    ax.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15,
    )
    ax.fill_between(
        train_sizes,
        validation_mean - validation_std,
        validation_mean + validation_std,
        alpha=0.15,
    )
    ax.set_title("Learning curve для лучшей модели")
    ax.set_xlabel("Количество объектов в обучении")
    ax.set_ylabel("Macro-F1")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()


def run_grid_search(X_train, y_train):
    print_section("GridSearchCV")

    pipeline = ImbPipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=1)),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    param_grid = {
        "classifier__n_estimators": [50, 100],
        "classifier__max_depth": [5, 10, None],
        "classifier__min_samples_leaf": [1, 2],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    print(f"Лучшие параметры: {grid_search.best_params_}")
    print(f"Лучший macro-F1 на CV: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def print_results_table(results: list[tuple[str, float]]) -> None:
    print_section("Итоговая таблица")
    print(f"{'Шаг':<36} {'Макро-F1 (test)':>16} {'Прирост':>12}")

    baseline = results[0][1]
    for name, score in results:
        growth = score - baseline
        growth_text = "-" if np.isclose(growth, 0.0) else f"{growth:+.4f}"
        print(f"{name:<36} {score:>16.4f} {growth_text:>12}")

    best_name, best_score = max(results, key=lambda item: item[1])
    growth = best_score - baseline
    relative_growth = growth / baseline * 100 if baseline > 0 else 0.0

    print("\nВывод по цели 5-10%:")
    if 0.05 <= growth <= 0.10:
        print(
            f"Цель достигнута: {best_name} улучшил macro-F1 на {growth:.4f} "
            f"({relative_growth:.2f}% относительно базовой модели)."
        )
    elif growth > 0.10:
        print(
            f"Цель перевыполнена: {best_name} улучшил macro-F1 на {growth:.4f} "
            f"({relative_growth:.2f}%)."
        )
    elif growth > 0:
        print(
            f"Качество улучшилось на {growth:.4f} ({relative_growth:.2f}%), "
            "но до целевых 5% не дотянуло."
        )
    else:
        print(
            "Улучшения на test нет. Для роста качества нужно расширить данные, "
            "изменить признаки или подобрать более широкую сетку параметров."
        )


def main() -> None:
    print_section("Загрузка и подготовка данных")

    wine = datasets.load_wine()
    X = wine.data
    y = wine.target
    target_names = [str(name) for name in wine.target_names]

    print(f"Объектов: {X.shape[0]}")
    print(f"Признаков: {X.shape[1]}")
    print(f"Классов: {len(target_names)}")
    print("\nРаспределение классов во всем датасете:")
    rarest_class = show_class_distribution(y, target_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train_model, y_train_model = make_educational_imbalanced_train(
        X_train,
        y_train,
        rarest_class,
    )

    print("\nTrain до учебного дисбаланса:")
    show_class_distribution(y_train, target_names)
    print("\nTrain после учебного дисбаланса из 3MO.py:")
    show_class_distribution(y_train_model, target_names)
    print("\nTest остается без изменений:")
    show_class_distribution(y_test, target_names)

    baseline_model = build_baseline_pipeline()
    baseline_f1 = evaluate_model(
        "Базовая модель из практики 3: RandomForest class_weight='balanced'",
        baseline_model,
        X_train_model,
        X_test,
        y_train_model,
        y_test,
        target_names,
    )

    smote_model = build_smote_pipeline()
    smote_f1 = evaluate_model(
        "+ SMOTE только на тренировочных данных",
        smote_model,
        X_train_model,
        X_test,
        y_train_model,
        y_test,
        target_names,
    )

    if smote_f1 >= baseline_f1:
        best_for_curve = build_smote_pipeline()
        print("\nДля learning_curve выбрана модель после SMOTE.")
    else:
        best_for_curve = build_baseline_pipeline()
        print("\nДля learning_curve выбрана модель с class_weight без SMOTE.")

    plot_learning_curves(best_for_curve, X_train_model, y_train_model)

    best_grid_model = run_grid_search(X_train_model, y_train_model)
    grid_f1 = evaluate_model(
        "+ GridSearchCV",
        best_grid_model,
        X_train_model,
        X_test,
        y_train_model,
        y_test,
        target_names,
    )

    print_results_table(
        [
            ("Базовая модель (практика 3)", baseline_f1),
            ("+ SMOTE / class_weight", smote_f1),
            ("+ GridSearchCV", grid_f1),
        ]
    )


if __name__ == "__main__":
    main()
