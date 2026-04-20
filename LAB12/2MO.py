import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import inspect
# Для борьбы с дисбалансом
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline  # используем для совместимости с SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

def _supports_param(estimator_cls, param_name: str) -> bool:
    """Надёжная проверка, принимает ли конструктор параметр.

    hasattr(..., 'class_weight') для sklearn-моделей неверно: это параметр __init__,
    а не атрибут класса.
    """

    try:
        sig = inspect.signature(estimator_cls.__init__)
    except (TypeError, ValueError):
        return False
    return param_name in sig.parameters


def main() -> None:
    data_path = Path(__file__).with_name("telecom_churn.csv")
    if not data_path.exists():
        raise FileNotFoundError(
            f"Не найден файл данных: {data_path}. "
            "Положите telecom_churn.csv рядом со скриптом или запустите из папки с CSV."
        )

    df = pd.read_csv(data_path)
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

# Определение категориальных и числовых признаков
    categorical_features = ['State', 'Area code', 'International plan', 'Voice mail plan']
    numerical_features = [col for col in X.columns if col not in categorical_features]

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'KNeighbors': KNeighborsClassifier()
    }

    best_model = None
    best_acc = 0
    best_name = ''
    for name, model in models.items():
        model.fit(X_train_processed, y_train)
        y_pred = model.predict(X_test_processed)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name

    print(f"\nЛучшая модель из первой работы: {best_name} с accuracy {best_acc:.4f}")

    print("\n" + "=" * 50)
    print("ДИАГНОСТИКА ТЕКУЩЕЙ МОДЕЛИ")
    print("=" * 50)

# Предсказания лучшей модели
    y_pred_best = best_model.predict(X_test_processed)

# Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_best, target_names=['Остался (0)', 'Ушёл (1)']))

# Матрица ошибок
    cm = confusion_matrix(y_test, y_pred_best)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Остался', 'Ушёл'])
    disp.plot(cmap='Blues')
    plt.title(f'Матрица ошибок для {best_name}')
    plt.show()

# Анализ дисбаланса
    print("\nРаспределение классов в тренировочной выборке:")
    print(y_train.value_counts())
    print("Доля ушедших: {:.2f}%".format(y_train.mean() * 100))

    print("\nЦель оптимизации: повысить Recall для класса 'Ушёл' (1) до ~0.7 при сохранении приемлемой Precision.")

    print("\n" + "=" * 50)
    print("СОЗДАНИЕ КОНВЕЙЕРА С БОРЬБОЙ С ДИСБАЛАНСОМ")
    print("=" * 50)

# Создадим предобработчик (такой же, как раньше)
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])

    model_class = type(best_model)

    # Для борьбы с дисбалансом добавим class_weight='balanced' (если модель поддерживает)
    if _supports_param(model_class, 'class_weight'):
        # random_state есть не у всех моделей; передаём только если поддерживается.
        clf_kwargs = {'class_weight': 'balanced'}
        if _supports_param(model_class, 'random_state'):
            clf_kwargs['random_state'] = 42

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model_class(**clf_kwargs))
        ])
        use_smote = False
    else:
        # Для KNeighbors (не поддерживает class_weight) используем SMOTE
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', model_class())
        ])
        use_smote = True

    print("Конвейер создан. Балансировка:", "class_weight='balanced'" if not use_smote else "SMOTE")

# 3. Системный поиск лучших параметров (GridSearchCV)
    print("\n" + "=" * 50)
    print("ПОИСК ГИПЕРПАРАМЕТРОВ С ПОМОЩЬЮ GRIDSEARCHCV")
    print("=" * 50)

# Определим сетку гиперпараметров в зависимости от типа модели
    param_grid = {}

    if model_class == LogisticRegression:
        param_grid = {
            'classifier__C': [0.01, 0.1, 1, 10, 100],
            'classifier__penalty': ['l2'],  # l1 требует solver='liblinear'
            'classifier__solver': ['lbfgs', 'liblinear']  # liblinear поддерживает l1, но для l2 тоже
        }
    elif model_class == DecisionTreeClassifier:
        param_grid = {
            'classifier__max_depth': [5, 10, 15, 20, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        }
    elif model_class == KNeighborsClassifier:
        param_grid = {
            'classifier__n_neighbors': [3, 5, 7, 9, 11],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__p': [1, 2]  # 1 - манхэттенское, 2 - евклидово
        }

# Если мы используем SMOTE, то параметры будут с префиксом 'smote__'? Но SMOTE обычно не настраиваем.
# Для простоты оставим только параметры классификатора.

# Выполним GridSearchCV с 5-кратной кросс-валидацией, метрика - f1 (так как хотим улучшить recall)
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='f1',  # можно также 'roc_auc' или 'recall', но f1 балансирует
        n_jobs=-1,
        verbose=1
    )

# Обучаем на тренировочных данных
    grid_search.fit(X_train, y_train)

    print("\nЛучшие параметры:", grid_search.best_params_)
    print("Лучшее среднее значение F1 на кросс-валидации:", grid_search.best_score_)

# 4. Финальная оценка на тестовом наборе
    print("\n" + "=" * 50)
    print("ФИНАЛЬНАЯ ОЦЕНКА НА ТЕСТОВЫХ ДАННЫХ")
    print("=" * 50)

# Лучшая модель уже обучена на всех тренировочных данных (refit=True)
    best_final_model = grid_search.best_estimator_

# Предсказания
    y_pred_final = best_final_model.predict(X_test)
# Для ROC-кривой нужны вероятности
    if hasattr(best_final_model, 'predict_proba'):
        y_proba = best_final_model.predict_proba(X_test)[:, 1]
    else:
        raise AttributeError("У лучшей модели нет метода predict_proba, ROC-AUC посчитать нельзя")

# Метрики
    accuracy = accuracy_score(y_test, y_pred_final)
    precision = precision_score(y_test, y_pred_final)
    recall = recall_score(y_test, y_pred_final)
    f1 = f1_score(y_test, y_pred_final)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print("\nClassification Report на тесте:")
    print(classification_report(y_test, y_pred_final, target_names=['Остался', 'Ушёл']))

# Построим ROC-кривую
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-кривая финальной модели')
    plt.legend(loc="lower right")
    plt.show()

# Извлечение важности признаков (если модель поддерживает)
    print("\n" + "=" * 50)
    print("АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ")
    print("=" * 50)

# Попробуем получить важность или коэффициенты
    if hasattr(best_final_model.named_steps['classifier'], 'coef_'):
        # Для линейных моделей
        coef = best_final_model.named_steps['classifier'].coef_.flatten()
        # Получим названия признаков после трансформации
        feature_names = best_final_model.named_steps['preprocessor'].get_feature_names_out()
        importance_df = pd.DataFrame({'feature': feature_names, 'coefficient': coef})
        importance_df['abs_coef'] = np.abs(coef)
        importance_df = importance_df.sort_values('abs_coef', ascending=False).head(20)
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df.head(15), x='abs_coef', y='feature')
        plt.title('Топ-15 признаков по модулю коэффициента (LogisticRegression)')
        plt.tight_layout()
        plt.show()
    elif hasattr(best_final_model.named_steps['classifier'], 'feature_importances_'):
        # Для деревьев
        importances = best_final_model.named_steps['classifier'].feature_importances_
        feature_names = best_final_model.named_steps['preprocessor'].get_feature_names_out()
        importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        importance_df = importance_df.sort_values('importance', ascending=False).head(20)
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df.head(15), x='importance', y='feature')
        plt.title('Топ-15 признаков по важности (DecisionTree)')
        plt.tight_layout()
        plt.show()
    else:
        print("Модель не предоставляет важности признаков.")

# Сравнение с исходной моделью
    print("\n" + "=" * 50)
    print("СРАВНЕНИЕ С ИСХОДНОЙ МОДЕЛЬЮ")
    print("=" * 50)

# Метрики исходной модели (лучшей из первой работы) на том же тесте
    y_pred_old = best_model.predict(X_test_processed)
    old_accuracy = accuracy_score(y_test, y_pred_old)
    old_precision = precision_score(y_test, y_pred_old)
    old_recall = recall_score(y_test, y_pred_old)
    old_f1 = f1_score(y_test, y_pred_old)
# Для старой модели ROC-AUC тоже можно посчитать, если есть predict_proba
    if hasattr(best_model, 'predict_proba'):
        old_proba = best_model.predict_proba(X_test_processed)[:, 1]
        old_roc_auc = roc_auc_score(y_test, old_proba)
    else:
        old_roc_auc = None

    comparison = pd.DataFrame({
        'Метрика': ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC'],
        'Исходная модель': [old_accuracy, old_precision, old_recall, old_f1, old_roc_auc],
        'Оптимизированная модель': [accuracy, precision, recall, f1, roc_auc]
    })
    print(comparison.to_string(index=False))

# Дополнительно можно построить матрицу ошибок для финальной модели
    cm_final = confusion_matrix(y_test, y_pred_final)
    disp_final = ConfusionMatrixDisplay(confusion_matrix=cm_final, display_labels=['Остался', 'Ушёл'])
    disp_final.plot(cmap='Blues')
    plt.title('Матрица ошибок финальной модели')
    plt.show()


if __name__ == "__main__":
    main()


