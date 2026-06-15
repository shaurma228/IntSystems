import pickle
import sys

# Настройка кодировки для корректного вывода кириллицы в консоли Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# =========================
# 1. ЗАГРУЗКА ДАННЫХ
# =========================
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

# X — признаки
# y — правильный ответ (Результат: 0 или 1)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# =========================
# 2. РАЗБИЕНИЕ ДАННЫХ
# =========================
# 80% — для обучения и проверки на этапе подбора модели
# 20% — финальный тест, который не участвует в обучении
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 3. ДВЕ РАЗНЫЕ МОДЕЛИ
# =========================

# Модель 1: Logistic Regression + масштабирование
log_reg_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))
])

# Модель 2: Decision Tree
tree_model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42,
    class_weight="balanced"
)

# =========================
# 4. КРОСС-ВАЛИДАЦИЯ
# =========================
log_reg_cv = cross_val_score(log_reg_model, X_train, y_train, cv=5, scoring="f1")
tree_cv = cross_val_score(tree_model, X_train, y_train, cv=5, scoring="f1")

print("\n=== КРОСС-ВАЛИДАЦИЯ ===")
print(f"Logistic Regression, средний F1: {log_reg_cv.mean():.4f}")
print(f"Decision Tree, средний F1:          {tree_cv.mean():.4f}")

# =========================
# 5. ОБУЧЕНИЕ ОБЕИХ МОДЕЛЕЙ
# =========================
log_reg_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)

# =========================
# 6. ОЦЕНКА НА ОТЛОЖЕННЫХ ДАННЫХ
# =========================
log_reg_pred = log_reg_model.predict(X_test)
tree_pred = tree_model.predict(X_test)

log_reg_acc = accuracy_score(y_test, log_reg_pred)
tree_acc = accuracy_score(y_test, tree_pred)

log_reg_f1 = f1_score(y_test, log_reg_pred)
tree_f1 = f1_score(y_test, tree_pred)

print("\n=== МЕТРИКИ НА ТЕСТЕ ===")
print(f"Logistic Regression -> Accuracy: {log_reg_acc:.4f}, F1: {log_reg_f1:.4f}")
print(f"Decision Tree          -> Accuracy: {tree_acc:.4f}, F1: {tree_f1:.4f}")

# =========================
# 7. ВЫБОР ЛУЧШЕЙ МОДЕЛИ
# =========================
# Выбираем по среднему F1 на кросс-валидации
if log_reg_cv.mean() >= tree_cv.mean():
    best_name = "Logistic Regression"
    best_model = log_reg_model
    best_cv_score = log_reg_cv.mean()
else:
    best_name = "Decision Tree"
    best_model = tree_model
    best_cv_score = tree_cv.mean()

# =========================
# 8. ФИНАЛЬНАЯ ПРОВЕРКА ЛУЧШЕЙ МОДЕЛИ
# =========================
best_model.fit(X_train, y_train)
best_pred = best_model.predict(X_test)

print("\n=== ОТЧЁТ О КЛАССИФИКАЦИИ (ЛУЧШАЯ МОДЕЛЬ) ===")
print(classification_report(y_test, best_pred))

# =========================
# 9. АНАЛИЗ ОШИБОК
# =========================
cm = confusion_matrix(y_test, best_pred)
tn, fp, fn, tp = cm.ravel()

print("\n=== МАТРИЦА ОШИБОК ===")
print(cm)

print("\n=== АНАЛИЗ ОШИБОК ===")
print(f"Ложноположительные (здоровых приняли за больных): {fp}")
print(f"Ложноотрицательные (больных приняли за здоровых): {fn}")

errors = X_test.copy()
errors["Actual"] = y_test.values
errors["Predicted"] = best_pred
errors = errors[errors["Actual"] != errors["Predicted"]]

print("\n=== ПРИМЕРЫ ОШИБОК (первые 10) ===")
if len(errors) > 0:
    print(errors.head(10).to_string(index=False))
else:
    print("Ошибок на тесте нет.")

# =========================
# 10. СОХРАНЕНИЕ ЛУЧШЕЙ МОДЕЛИ
# =========================
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("\nМодель сохранена в model.pkl")

# =========================
# 11. ИТОГОВЫЙ ОТЧЁТ
# =========================
print("\n=== ИТОГОВЫЙ ОТЧЁТ ===")
print(f"Лучшая модель — {best_name}.")
print(f"Её ключевая метрика на новых данных — F1 = {f1_score(y_test, best_pred):.4f}.")
print(f"Чаще всего она путает классы 0 и 1 при пограничных значениях.")
