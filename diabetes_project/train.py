import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)

from src.data import load_data, prepare_data, validate_columns
from src.model import create_pipelines, tune_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent


def main() -> None:
    with open(ROOT / "config" / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    ds_cfg = config["dataset"]
    model_cfg = config["model"]

    logger.info("Загрузка данных...")
    df = load_data(ds_cfg["url"], local_cache=ROOT / ds_cfg.get("local_cache", ""))
    validate_columns(df, ds_cfg.get("required_columns", []))

    X_train, X_test, y_train, y_test = prepare_data(
        df,
        ds_cfg["zero_columns"],
        test_size=ds_cfg["test_size"],
        random_state=ds_cfg["random_state"],
    )

    pipelines = create_pipelines(
        random_state=ds_cfg["random_state"],
        max_iter=model_cfg.get("max_iter", 1000),
    )
    best_models_info = tune_models(
        X_train,
        y_train,
        pipelines,
        config["hyperparameters"],
        cv=model_cfg["cv"],
        scoring=model_cfg["scoring"],
    )

    best_name = max(best_models_info, key=lambda x: best_models_info[x]["score"])
    best_pipeline = best_models_info[best_name]["model"]
    logger.info("Победитель: %s", best_name)

    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]

    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    logger.info("\n=== ОТЧЁТ НА ТЕСТОВОЙ ВЫБОРКЕ ===\n%s", classification_report(y_test, y_pred))
    logger.info("ROC-AUC: %.4f", roc_auc)

    # Матрица ошибок — позволяет видеть FP и FN, а не просто итоговый счёт
    cm = confusion_matrix(y_test, y_pred)
    fn = int(cm[1, 0])  # actual=Диабет, predicted=Нет — ложноотрицательные
    cm_path = ROOT / model_cfg.get("confusion_matrix_path", "models/confusion_matrix.png")
    cm_path.parent.mkdir(parents=True, exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Нет диабета", "Диабет"])
    disp.plot()
    plt.title("Матрица ошибок")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    logger.info("Матрица ошибок сохранена в %s", cm_path)

    test_results = X_test.copy()
    test_results["Actual"] = y_test.values
    test_results["Predicted"] = y_pred
    errors = test_results[test_results["Actual"] != test_results["Predicted"]]

    error_path = ROOT / model_cfg.get("error_analysis_path", "models/error_analysis.csv")
    if len(errors) > 0:
        error_path.parent.mkdir(parents=True, exist_ok=True)
        errors.to_csv(error_path, index=False)
        logger.info("Ошибочные примеры сохранены в %s", error_path)

    model_path = ROOT / model_cfg["path"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_data = {
        "model": best_pipeline,
        "feature_names": X_train.columns.tolist(),
        "model_name": best_name,
        "recall_test": recall,
        "roc_auc_test": roc_auc,
        "errors_sample": errors.head(10) if len(errors) > 0 else None,
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    logger.info("Модель и метаданные сохранены в %s", model_path)

    print(
        f"\nЛучшая модель — {best_name}. "
        f"Recall на новых данных — {recall:.2f}. "
        f"Чаще всего она принимает больных за здоровых ({fn} случаев)."
    )


if __name__ == "__main__":
    main()
