# -*- coding: utf-8 -*-
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


RANDOM_STATE = 42
TEST_SIZE = 0.25
SELECTED_FEATURES = ["MedInc", "AveRooms", "HouseAge"]

FEATURE_DESCRIPTIONS = {
    "MedInc": "медианный доход жителей района, десятки тысяч долларов",
    "AveRooms": "среднее число комнат в домохозяйстве",
    "HouseAge": "медианный возраст домов в районе, лет",
}


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_dataset():
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target
    feature_names = list(housing.feature_names)
    return X, y, feature_names


def get_selected_columns(X: np.ndarray, feature_names: list[str]) -> tuple[np.ndarray, list[int]]:
    selected_indices = [feature_names.index(feature) for feature in SELECTED_FEATURES]
    return X[:, selected_indices], selected_indices


def plot_target_histogram(y: np.ndarray) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(y, bins=35, color="steelblue", edgecolor="black", alpha=0.85)
    plt.title("Распределение целевой переменной MedHouseVal")
    plt.xlabel("Медианная стоимость дома, сотни тысяч долларов")
    plt.ylabel("Количество объектов")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()


def plot_feature_relationships(
    X_selected: np.ndarray,
    y: np.ndarray,
    selected_features: list[str],
) -> None:
    fig, axes = plt.subplots(1, len(selected_features), figsize=(15, 4.5))

    for ax, feature_values, feature_name in zip(axes, X_selected.T, selected_features):
        ax.scatter(feature_values, y, alpha=0.25, s=12)
        ax.set_title(feature_name)
        ax.set_xlabel(FEATURE_DESCRIPTIONS[feature_name])
        ax.set_ylabel("MedHouseVal, сотни тысяч долларов")
        ax.grid(alpha=0.25)

    fig.suptitle("Связь выбранных признаков с ценой дома")
    fig.tight_layout()
    plt.show()


def evaluate_regression_model(
    model_name: str,
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print_section(model_name)
    print(f"MAE: {mae:.4f} сотен тысяч долларов")
    print(f"MSE: {mse:.4f}")
    print(f"R2:  {r2:.4f}")

    return {
        "model": model,
        "y_pred": y_pred,
        "mae": mae,
        "mse": mse,
        "r2": r2,
    }


def money_text(value_in_100k: float) -> str:
    dollars = abs(value_in_100k) * 100_000
    return f"{dollars:,.0f}".replace(",", " ")


def explain_linear_coefficients(model: LinearRegression, selected_features: list[str]) -> None:
    print_section("Интерпретация коэффициентов LinearRegression")

    print("Коэффициенты model.coef_:")
    for feature_name, coefficient in zip(selected_features, model.coef_):
        print(f"{feature_name}: {coefficient:.4f}")
    print(f"Свободный член model.intercept_: {model.intercept_:.4f}")

    print("\nРасшифровка для бизнес-заказчика:")
    for feature_name, coefficient in zip(selected_features, model.coef_):
        direction = "росту" if coefficient >= 0 else "снижению"
        print(
            f"Если {FEATURE_DESCRIPTIONS[feature_name]} увеличивается на 1, "
            f"то прогнозируемая цена в среднем меняется к {direction} "
            f"примерно на {money_text(coefficient)} долларов "
            "при прочих равных признаках."
        )


def compare_models(linear_result: dict, tree_result: dict) -> dict:
    print_section("Сравнение линейной модели и дерева решений")

    print(f"{'Модель':<28} {'MAE':>10} {'MSE':>10} {'R2':>10}")
    print(
        f"{'LinearRegression':<28} "
        f"{linear_result['mae']:>10.4f} {linear_result['mse']:>10.4f} {linear_result['r2']:>10.4f}"
    )
    print(
        f"{'DecisionTreeRegressor':<28} "
        f"{tree_result['mae']:>10.4f} {tree_result['mse']:>10.4f} {tree_result['r2']:>10.4f}"
    )

    if tree_result["r2"] > linear_result["r2"]:
        print(
            "\nПо R2 лучше дерево решений: оно может учитывать нелинейные зависимости "
            "между выбранными признаками и ценой."
        )
        return tree_result

    print(
        "\nПо R2 лучше линейная регрессия: на выбранных признаках простая линейная "
        "зависимость оказалась достаточно удачной."
    )
    return linear_result


def plot_real_vs_predicted(y_test: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
    min_value = min(y_test.min(), y_pred.min())
    max_value = max(y_test.max(), y_pred.max())

    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, y_pred, alpha=0.35, s=18)
    plt.plot([min_value, max_value], [min_value, max_value], "r--", label="Идеальная линия y=x")
    plt.title(f"Реальные и предсказанные цены: {model_name}")
    plt.xlabel("Реальная цена, сотни тысяч долларов")
    plt.ylabel("Предсказанная цена, сотни тысяч долларов")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    print_section("1. Загрузка и исследование данных")

    X, y, feature_names = load_dataset()
    X_selected, selected_indices = get_selected_columns(X, feature_names)

    print("Выбран датасет California Housing из sklearn.")
    print(f"Количество объектов: {X.shape[0]}")
    print(f"Количество признаков: {X.shape[1]}")
    print(f"Все признаки: {feature_names}")
    print(f"Выбранные признаки: {SELECTED_FEATURES}")
    print(
        "Целевая переменная MedHouseVal измеряется в сотнях тысяч долларов "
        "и показывает медианную стоимость дома в районе."
    )

    print("\nСтатистика целевой переменной:")
    print(f"Минимум: {y.min():.4f}")
    print(f"Среднее: {y.mean():.4f}")
    print(f"Медиана: {np.median(y):.4f}")
    print(f"Максимум: {y.max():.4f}")

    plot_target_histogram(y)

    print_section("2. Выбранные признаки")
    for feature_name, feature_index in zip(SELECTED_FEATURES, selected_indices):
        correlation = np.corrcoef(X[:, feature_index], y)[0, 1]
        print(
            f"{feature_name}: {FEATURE_DESCRIPTIONS[feature_name]}; "
            f"корреляция с ценой = {correlation:.4f}"
        )

    plot_feature_relationships(X_selected, y, SELECTED_FEATURES)

    X_train, X_test, y_train, y_test = train_test_split(
        X_selected,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print_section("3. Обучение LinearRegression")
    print(f"Размер train: {X_train.shape}")
    print(f"Размер test: {X_test.shape}")

    linear_model = LinearRegression()
    linear_result = evaluate_regression_model(
        "LinearRegression на 3 выбранных признаках",
        linear_model,
        X_train,
        X_test,
        y_train,
        y_test,
    )
    explain_linear_coefficients(linear_model, SELECTED_FEATURES)

    tree_model = DecisionTreeRegressor(max_depth=3, random_state=RANDOM_STATE)
    tree_result = evaluate_regression_model(
        "DecisionTreeRegressor(max_depth=3)",
        tree_model,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    best_result = compare_models(linear_result, tree_result)
    best_model_name = (
        "DecisionTreeRegressor(max_depth=3)"
        if best_result is tree_result
        else "LinearRegression"
    )

    print(
        "\nНа scatter plot чем ближе точки к красной пунктирной линии y=x, "
        "тем точнее прогноз. Большой разброс вокруг линии означает заметную ошибку модели."
    )
    plot_real_vs_predicted(y_test, best_result["y_pred"], best_model_name)


if __name__ == "__main__":
    main()
