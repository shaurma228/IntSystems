# -*- coding: utf-8 -*-
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.35

# Настройки специально оставлены не максимальными, чтобы в практике 4 было
# пространство для улучшения через SMOTE и GridSearchCV.
BASE_RF_PARAMS = {
    "n_estimators": 30,
    "max_depth": 2,
    "min_samples_leaf": 4,
    "random_state": RANDOM_STATE,
}
RARE_CLASS_TRAIN_LIMIT = 8


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def show_class_distribution(y: np.ndarray, target_names: list[str]) -> int:
    counts = np.bincount(y, minlength=len(target_names))

    print(f"{'class_id':>8}  {'class_name':>10}  {'count':>5}  {'share':>7}")
    for class_id, count in enumerate(counts):
        share = count / len(y)
        print(f"{class_id:>8}  {target_names[class_id]:>10}  {count:>5}  {share:>7.4f}")

    rarest_class = int(np.argmin(counts))
    imbalance_ratio = counts.max() / counts.min()
    print(f"\nСамый редкий класс: {target_names[rarest_class]} ({counts.min()} объектов)")
    print(f"Коэффициент дисбаланса max/min: {imbalance_ratio:.2f}")

    if imbalance_ratio < 1.5:
        print("Вывод: сильного дисбаланса нет, классы распределены достаточно близко.")
    else:
        print("Вывод: есть заметный дисбаланс классов.")

    return rarest_class


def make_educational_imbalanced_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    rare_class: int,
    rare_limit: int = RARE_CLASS_TRAIN_LIMIT,
) -> tuple[np.ndarray, np.ndarray]:
    """Создает учебный дисбаланс только в train, чтобы не портить честный test."""
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
) -> dict:
    print_section(f"Модель: {model_name}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy на тестовой выборке: {accuracy:.4f}")

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

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(target_names)))
    print("Матрица ошибок:")
    print(cm)

    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    display.plot(cmap="Blues", ax=ax, values_format="d", colorbar=False)
    ax.set_title(f"Confusion matrix: {model_name}")
    fig.tight_layout()
    plt.show()

    recalls = {
        class_id: report_dict[target_names[class_id]]["recall"]
        for class_id in range(len(target_names))
    }
    worst_recall = min(recalls.values())
    worst_classes = [
        class_id for class_id, recall in recalls.items() if np.isclose(recall, worst_recall)
    ]

    print(f"\nАнализ худшего класса для {model_name}:")
    if np.isclose(worst_recall, 1.0) and len(worst_classes) == len(target_names):
        print("Худшего класса нет: recall всех классов равен 1.0000.")
        worst_class = None
    else:
        worst_class = worst_classes[0]
        total_objects = int(cm[worst_class].sum())
        correct_objects = int(cm[worst_class, worst_class])
        missed_objects = total_objects - correct_objects
        print(
            f"Хуже всего предсказывается {target_names[worst_class]}: "
            f"recall={worst_recall:.4f}."
        )
        print(
            f"В test объектов этого класса: {total_objects}; "
            f"верно найдено: {correct_objects}; пропущено: {missed_objects}."
        )
        print("Причина видна в строке этого класса в матрице ошибок.")

    return {
        "accuracy": accuracy,
        "macro_f1": report_dict["macro avg"]["f1-score"],
        "report": report_dict,
        "confusion_matrix": cm,
        "worst_class": worst_class,
    }


def main() -> None:
    print_section("1. Загрузка и изучение данных")

    wine = datasets.load_wine()
    X = wine.data
    y = wine.target
    target_names = [str(name) for name in wine.target_names]

    print(f"Количество объектов: {X.shape[0]}")
    print(f"Количество признаков: {X.shape[1]}")
    print(f"Количество классов: {len(target_names)}")
    print(f"Названия классов: {target_names}")
    print(f"Названия признаков: {wine.feature_names}")

    print_section("2. Базовый EDA: распределение классов")
    rarest_class = show_class_distribution(y, target_names)

    print_section("3. Train/test split со stratify=y")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Размер X_train до учебного дисбаланса: {X_train.shape}")
    print(f"Размер X_test: {X_test.shape}")
    print("\nРаспределение классов в train до учебного дисбаланса:")
    show_class_distribution(y_train, target_names)
    print("\nРаспределение классов в test:")
    show_class_distribution(y_test, target_names)

    X_train_model, y_train_model = make_educational_imbalanced_train(
        X_train,
        y_train,
        rarest_class,
    )
    print_section("Учебный дисбаланс для практики 4")
    print(
        "Чтобы в следующей практике было возможно улучшить macro-F1 на 5-10%, "
        "редкий класс ограничен только в обучающей выборке. Test остается неизменным."
    )
    print(f"Размер X_train после учебного дисбаланса: {X_train_model.shape}")
    show_class_distribution(y_train_model, target_names)

    models = {
        "RandomForest": RandomForestClassifier(**BASE_RF_PARAMS),
        "LogisticRegression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=0.02,
                        max_iter=10000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "RandomForest balanced": RandomForestClassifier(
            **BASE_RF_PARAMS,
            class_weight="balanced",
        ),
    }

    results = {}
    for model_name, model in models.items():
        results[model_name] = evaluate_model(
            model_name,
            model,
            X_train_model,
            X_test,
            y_train_model,
            y_test,
            target_names,
        )

    print_section("4. Сравнение моделей")
    comparison = []
    for model_name, result in results.items():
        if result["worst_class"] is None:
            worst_class_name = "нет"
            worst_recall = 1.0
        else:
            worst_class_name = target_names[result["worst_class"]]
            worst_recall = result["report"][worst_class_name]["recall"]

        comparison.append(
            {
                "model": model_name,
                "accuracy": result["accuracy"],
                "macro_f1": result["macro_f1"],
                "worst_class": worst_class_name,
                "worst_recall": worst_recall,
            }
        )

    comparison.sort(key=lambda row: row["macro_f1"], reverse=True)

    print(
        f"{'model':<24} {'accuracy':>10} {'macro_f1':>10} "
        f"{'worst_class':>12} {'worst_recall':>13}"
    )
    for row in comparison:
        print(
            f"{row['model']:<24} {row['accuracy']:>10.4f} "
            f"{row['macro_f1']:>10.4f} {row['worst_class']:>12} "
            f"{row['worst_recall']:>13.4f}"
        )

    base_rf = results["RandomForest"]
    balanced_rf = results["RandomForest balanced"]
    rare_class_name = target_names[rarest_class]

    print("\nВлияние class_weight='balanced' на RandomForest:")
    print(
        f"Recall редкого класса {rare_class_name}: "
        f"{base_rf['report'][rare_class_name]['recall']:.4f} -> "
        f"{balanced_rf['report'][rare_class_name]['recall']:.4f}"
    )
    print(f"Macro-F1: {base_rf['macro_f1']:.4f} -> {balanced_rf['macro_f1']:.4f}")

    base_models = ["RandomForest", "LogisticRegression"]
    best_base_macro_f1 = max(results[name]["macro_f1"] for name in base_models)
    best_base_models = [
        name for name in base_models if np.isclose(results[name]["macro_f1"], best_base_macro_f1)
    ]

    print("\nВывод:")
    if len(best_base_models) == 1:
        print(
            f"По macro-F1 среди двух базовых моделей лучше справилась "
            f"{best_base_models[0]}: {best_base_macro_f1:.4f}."
        )
    else:
        print(
            "По macro-F1 две базовые модели справились одинаково: "
            f"{best_base_macro_f1:.4f}."
        )

    print(
        "Для итогового прототипа выбираю RandomForest balanced: "
        "он учитывает вес редкого класса и оставляет понятную точку роста для практики 4."
    )
    print(
        f"Зафиксированные настройки для следующей практики: {BASE_RF_PARAMS}, "
        "class_weight='balanced'."
    )


if __name__ == "__main__":
    main()
