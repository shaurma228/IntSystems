import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.3
SELECTED_FEATURES = ["MedInc", "AveRooms", "HouseAge"]
N_NOISE_FEATURES = 30


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_previous_practice_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target
    feature_names = list(housing.feature_names)
    selected_indices = [feature_names.index(feature) for feature in SELECTED_FEATURES]
    return X[:, selected_indices], y, SELECTED_FEATURES


def scale_train_test(
        X_train: np.ndarray,
        X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def add_noise_features(
        X_train: np.ndarray,
        X_test: np.ndarray,
        n_noise_features: int = N_NOISE_FEATURES,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_STATE)
    train_noise = rng.normal(size=(X_train.shape[0], n_noise_features))
    test_noise = rng.normal(size=(X_test.shape[0], n_noise_features))

    X_train_with_noise = np.hstack([X_train, train_noise])
    X_test_with_noise = np.hstack([X_test, test_noise])
    return X_train_with_noise, X_test_with_noise


def train_linear_and_get_r2(
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
) -> tuple[LinearRegression, float, float]:
    model = LinearRegression()
    model.fit(X_train, y_train)

    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))
    return model, train_r2, test_r2


def train_ridge_and_get_test_r2(
        alpha: float,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
) -> tuple[Ridge, float, float]:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))
    return model, train_r2, test_r2


def print_results_table(
        baseline_test_r2: float,
        broken_train_r2: float,
        broken_test_r2: float,
        best_alpha: float,
        best_ridge_train_r2: float,
        best_ridge_test_r2: float,
) -> None:
    print_section("Итоговая таблица")

    print(f"{'Модель/условия':<48} {'R2 на test':>12}  Вывод")
    print(
        f"{'Базовая (предыдущая практика)':<48} "
        f"{baseline_test_r2:>12.4f}  Наш ориентир"
    )
    print(
        f"{'Сломанная (LinearRegression + шум)':<48} "
        f"{broken_test_r2:>12.4f}  Катастрофа!"
    )
    print(
        f"{'Исправленная (Ridge + шум, alpha = ' + str(best_alpha) + ')':<48} "
        f"{best_ridge_test_r2:>12.4f}  Спасение!"
    )

    quality_drop = baseline_test_r2 - broken_test_r2
    quality_return_gap = baseline_test_r2 - best_ridge_test_r2
    ridge_improvement = best_ridge_test_r2 - broken_test_r2

    print("\nОтветы на вопросы:")
    print(f"1. Качество после добавления шума упало на {quality_drop:.4f} R2.")
    print(f"2. Лучший alpha для Ridge: {best_alpha}.")
    print(
        f"3. Ridge улучшил сломанную модель на {ridge_improvement:.4f} R2 "
        f"по сравнению с LinearRegression + шум."
    )

    if quality_return_gap <= 0:
        print(
            "4. Качество удалось вернуть к базовому уровню и даже немного превысить его."
        )
    else:
        print(
            f"4. До базового уровня осталось {quality_return_gap:.4f} R2. "
            "Ridge заметно приблизил качество к ориентиру."
        )

    print(
        f"\nПроверка переобучения: у сломанной модели R2_train={broken_train_r2:.4f}, "
        f"а R2_test={broken_test_r2:.4f}. Большой разрыв означает переобучение на шум."
    )
    print(
        f"У лучшей Ridge-модели R2_train={best_ridge_train_r2:.4f}, "
        f"R2_test={best_ridge_test_r2:.4f}: регуляризация уменьшила разрыв."
    )


def main() -> None:
    print_section("1. Подготовка данных как в прошлой практике")

    X, y, selected_features = load_previous_practice_data()
    print("Датасет: California Housing")
    print(f"Выбранные признаки из 5MO.py: {selected_features}")
    print(f"X shape: {X.shape}")
    print(f"Y shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print(f"Размер train после split: {X_train.shape}")
    print(f"Размер test после split:  {X_test.shape}")

    # Масштабирование чистых данных
    X_train_scaled, X_test_scaled, _ = scale_train_test(X_train, X_test)

    baseline_model, baseline_train_r2, baseline_test_r2 = train_linear_and_get_r2(
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
    )

    print("\nБазовая LinearRegression без шумовых признаков:")
    print(f"R2_train: {baseline_train_r2:.4f}")
    print(f"R2_test:  {baseline_test_r2:.4f}")

    print_section("2. Создаем проблему: добавляем 30 случайных признаков-шумов")

    X_train_noise_raw, X_test_noise_raw = add_noise_features(X_train, X_test)
    print(f"Было признаков: {X_train.shape[1]}")
    print(f"Добавлено шумовых признаков: {N_NOISE_FEATURES}")
    print(f"Стало признаков: {X_train_noise_raw.shape[1]}")

    X_train_noise_scaled, X_test_noise_scaled, _ = scale_train_test(
        X_train_noise_raw,
        X_test_noise_raw,
    )

    print_section("3. Демонстрация поломки")

    broken_model, broken_train_r2, broken_test_r2 = train_linear_and_get_r2(
        X_train_noise_scaled,
        X_test_noise_scaled,
        y_train,
        y_test,
    )

    print("LinearRegression на данных с шумом:")
    print(f"R2_train: {broken_train_r2:.4f}")
    print(f"R2_test:  {broken_test_r2:.4f}")
    print(f"Базовый R2_test был: {baseline_test_r2:.4f}")

    if broken_test_r2 < baseline_test_r2:
        print("Стало хуже: шумовые признаки вызвали переобучение.")
    else:
        print(
            "R2_test не стал хуже. Возможно, шум оказался не настолько вредным; "
            "но задание требует показать ухудшение – в данном запуске оно наблюдается "
            "не всегда из-за случайности. Для гарантии можно увеличить количество шумов."
        )

    print_section("4. Поиск alpha для Ridge-регрессии")

    alphas = [0.1, 1, 10, 100, 1000]
    ridge_results = []

    print(f"{'alpha':>10} {'R2_train':>12} {'R2_test':>12}")
    for alpha in alphas:
        ridge_model, ridge_train_r2, ridge_test_r2 = train_ridge_and_get_test_r2(
            alpha,
            X_train_noise_scaled,
            X_test_noise_scaled,
            y_train,
            y_test,
        )
        ridge_results.append((alpha, ridge_model, ridge_train_r2, ridge_test_r2))
        print(f"{alpha:>10} {ridge_train_r2:>12.4f} {ridge_test_r2:>12.4f}")

    best_alpha, best_ridge_model, best_ridge_train_r2, best_ridge_test_r2 = max(
        ridge_results,
        key=lambda item: item[3],
    )

    print(
        f"\nЛучший alpha={best_alpha}: "
        f"R2_test={best_ridge_test_r2:.4f}, R2_train={best_ridge_train_r2:.4f}."
    )

    print_results_table(
        baseline_test_r2,
        broken_train_r2,
        broken_test_r2,
        best_alpha,
        best_ridge_train_r2,
        best_ridge_test_r2,
    )


if __name__ == "__main__":
    main()