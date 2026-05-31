import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
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
TEST_SIZE = 0.25


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def analyze_class_distribution(y: np.ndarray, target_names: np.ndarray) -> int:
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
        print("Вывод по EDA: сильного дисбаланса нет, классы распределены достаточно близко.")
    else:
        print("Вывод по EDA: есть заметный дисбаланс классов.")

    return rarest_class


def describe_worst_class(
    model_name: str,
    report_dict: dict,
    cm: np.ndarray,
    target_names: np.ndarray,
) -> Optional[int]:
    recalls = {
        class_id: report_dict[target_names[class_id]]["recall"]
        for class_id in range(len(target_names))
    }
    min_recall = min(recalls.values())
    worst_classes = [
        class_id for class_id, recall in recalls.items() if np.isclose(recall, min_recall)
    ]

    print(f"\nАнализ худшего класса для {model_name}:")

    if np.isclose(min_recall, 1.0) and len(worst_classes) == len(target_names):
        print(
            "Худшего класса нет: у всех классов recall=1.0000, "
            "то есть модель не пропустила ни одного объекта в тестовой выборке."
        )
        print("Матрица ошибок диагональная: все объекты каждого класса предсказаны верно.")
        return None

    worst_class = worst_classes[0]
    total_objects = int(cm[worst_class].sum())
    correct_objects = int(cm[worst_class, worst_class])
    false_negative_objects = total_objects - correct_objects

    wrong_predictions = []
    for predicted_class in range(len(target_names)):
        if predicted_class == worst_class:
            continue
        mistakes = int(cm[worst_class, predicted_class])
        if mistakes > 0:
            wrong_predictions.append(f"{target_names[predicted_class]}: {mistakes}")

    print(
        f"Хуже всего предсказывается {target_names[worst_class]}: "
        f"recall={recalls[worst_class]:.4f}."
    )
    print(
        f"В тестовой выборке объектов этого класса: {total_objects}; "
        f"верно найдено: {correct_objects}; пропущено: {false_negative_objects}."
    )

    if wrong_predictions:
        print("Ошибочно отнесены к классам: " + ", ".join(wrong_predictions) + ".")
    else:
        print("Ошибок по этому классу в матрице ошибок нет.")

    if false_negative_objects == 0:
        print("Причина выбора класса как худшего: при равном recall=1.0 это один из классов-лидеров по минимуму.")
    else:
        print(
            "Причина: recall снижается из-за пропущенных объектов этого класса "
            "в строке матрицы ошибок."
        )

    return worst_class


def evaluate_model(
    model_name: str,
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    target_names: np.ndarray,
) -> dict:
    print_section(f"Модель: {model_name}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy на тестовой выборке: {accuracy:.4f}")

    print("\nClassification report:")
    report_text = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    print(report_text)

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

    worst_class = describe_worst_class(model_name, report_dict, cm, target_names)

    return {
        "model": model,
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
    feature_names = wine.feature_names

    print(f"Количество объектов: {X.shape[0]}")
    print(f"Количество признаков: {X.shape[1]}")
    print(f"Количество классов: {len(target_names)}")
    print(f"Названия классов: {list(target_names)}")
    print(f"Названия признаков: {feature_names}")

    print_section("2. Базовый EDA: распределение классов")
    rarest_class = analyze_class_distribution(y, target_names)

    print_section("3. Train/test split со stratify=y")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Размер X_train: {X_train.shape}")
    print(f"Размер X_test: {X_test.shape}")
    print("\nРаспределение классов в train:")
    analyze_class_distribution(y_train, target_names)
    print("\nРаспределение классов в test:")
    analyze_class_distribution(y_test, target_names)

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
        ),
        "LogisticRegression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=10000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "RandomForest balanced": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
    }

    results = {}
    for model_name, model in models.items():
        results[model_name] = evaluate_model(
            model_name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            target_names,
        )

    print_section("4. Сравнение моделей")
    comparison = [
        {
            "model": model_name,
            "accuracy": result["accuracy"],
            "macro_f1": result["macro_f1"],
            "worst_class": (
                "нет"
                if result["worst_class"] is None
                else target_names[result["worst_class"]]
            ),
            "worst_class_recall": (
                1.0
                if result["worst_class"] is None
                else result["report"][target_names[result["worst_class"]]]["recall"]
            ),
        }
        for model_name, result in results.items()
    ]
    comparison.sort(key=lambda row: row["macro_f1"], reverse=True)

    print(
        f"{'model':<24} {'accuracy':>10} {'macro_f1':>10} "
        f"{'worst_class':>12} {'worst_recall':>13}"
    )
    for row in comparison:
        print(
            f"{row['model']:<24} {row['accuracy']:>10.4f} "
            f"{row['macro_f1']:>10.4f} {row['worst_class']:>12} "
            f"{row['worst_class_recall']:>13.4f}"
        )

    base_rf = results["RandomForest"]
    balanced_rf = results["RandomForest balanced"]
    base_rf_worst_class = base_rf["worst_class"]

    print("\nВлияние class_weight='balanced' на RandomForest:")
    if base_rf_worst_class is None:
        rare_class_name = target_names[rarest_class]
        base_recall = base_rf["report"][rare_class_name]["recall"]
        balanced_recall = balanced_rf["report"][rare_class_name]["recall"]
        print(
            "У обычного RandomForest нет худшего класса: recall всех классов равен 1.0000."
        )
        print(
            f"Для редкого класса {rare_class_name} recall изменился так: "
            f"{base_recall:.4f} -> {balanced_recall:.4f}"
        )
    else:
        base_rf_worst_class_name = target_names[base_rf_worst_class]
        base_recall = base_rf["report"][base_rf_worst_class_name]["recall"]
        balanced_recall = balanced_rf["report"][base_rf_worst_class_name]["recall"]
        print(
            f"Recall класса {base_rf_worst_class_name}, который был худшим у обычного RandomForest: "
            f"{base_recall:.4f} -> {balanced_recall:.4f}"
        )
    print(f"Macro-F1: {base_rf['macro_f1']:.4f} -> {balanced_rf['macro_f1']:.4f}")

    base_model_names = ["RandomForest", "LogisticRegression"]
    best_base_macro_f1 = max(results[name]["macro_f1"] for name in base_model_names)
    best_base_models = [
        name
        for name in base_model_names
        if np.isclose(results[name]["macro_f1"], best_base_macro_f1)
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
            f"RandomForest={results['RandomForest']['macro_f1']:.4f}, "
            f"LogisticRegression={results['LogisticRegression']['macro_f1']:.4f}."
        )

    rare_class_name = target_names[rarest_class]
    rare_class_recalls = {
        model_name: result["report"][rare_class_name]["recall"]
        for model_name, result in results.items()
    }
    best_for_rare_class = max(
        rare_class_recalls,
        key=lambda name: (
            rare_class_recalls[name],
            results[name]["macro_f1"],
            1 if name == "RandomForest balanced" else 0,
        ),
    )

    print(
        f"Если важно не пропустить представителей редкого класса "
        f"{rare_class_name}, выбираю {best_for_rare_class}: "
        f"recall редкого класса={rare_class_recalls[best_for_rare_class]:.4f}, "
        f"macro-F1={results[best_for_rare_class]['macro_f1']:.4f}."
    )
    top_macro_f1 = comparison[0]["macro_f1"]
    top_models = [
        row["model"] for row in comparison if np.isclose(row["macro_f1"], top_macro_f1)
    ]
    if len(top_models) == 1:
        print(
            f"Итоговая рекомендация с учетом всех метрик: {top_models[0]}, "
            "так как эта модель имеет лучший баланс качества по классам в данном запуске."
        )
    else:
        print(
            f"Итоговая рекомендация для прототипа: {best_for_rare_class}. "
            "В этом запуске лучшие модели дают одинаковые метрики, но class_weight='balanced' "
            "лучше соответствует условию про редкий класс и не ухудшает качество."
        )


if __name__ == "__main__":
    main()
