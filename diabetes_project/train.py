import pickle

import yaml
from sklearn.metrics import classification_report, f1_score

from src.data import load_data, prepare_data
from src.model import create_pipelines, tune_models


def main():
    # 1. Загрузка конфига
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. Загрузка и подготовка данных
    print("Загрузка данных...")
    df = load_data(config["dataset"]["url"])
    X_train, X_test, y_train, y_test = prepare_data(
        df,
        config["dataset"]["zero_columns"],
        test_size=config["dataset"]["test_size"],
        random_state=config["dataset"]["random_state"]
    )

    # 3. Тюнинг моделей
    pipelines = create_pipelines()
    best_models_info = tune_models(
        X_train,
        y_train,
        pipelines,
        config["hyperparameters"],
        cv=config["model"]["cv"],
        scoring=config["model"]["scoring"]
    )

    # 4. Выбор лучшей модели
    best_name = max(best_models_info, key=lambda x: best_models_info[x]["score"])
    best_pipeline = best_models_info[best_name]["model"]

    print(f"\nПобедитель: {best_name}")

    # 5. Оценка на тесте и анализ ошибок
    y_pred = best_pipeline.predict(X_test)
    print("\n=== ОТЧЁТ НА ТЕСТОВОЙ ВЫБОРКЕ ===")
    print(classification_report(y_test, y_pred))

    # Анализ ошибок
    test_results = X_test.copy()
    test_results["Actual"] = y_test
    test_results["Predicted"] = y_pred
    errors = test_results[test_results["Actual"] != test_results["Predicted"]]

    print(f"\nКоличество ошибок на тесте: {len(errors)}")
    if len(errors) > 0:
        errors.to_csv("models/error_analysis.csv", index=False)
        print("Ошибочные примеры сохранены в models/error_analysis.csv")

    # 6. Сохранение
    model_data = {
        "model": best_pipeline,
        "feature_names": X_train.columns.tolist(),
        "model_name": best_name,
        "f1_test": f1_score(y_test, y_pred),
        "errors_sample": errors.head(10) if len(errors) > 0 else None
    }
    with open(config["model"]["path"], "wb") as f:
        pickle.dump(model_data, f)

    print(f"\nМодель и метаданные сохранены в {config['model']['path']}")


if __name__ == "__main__":
    main()
