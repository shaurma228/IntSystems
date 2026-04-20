import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


def main() -> None:
    data_path = Path(__file__).with_name("telecom_churn.csv")
    if not data_path.exists():
        raise FileNotFoundError(
            f"Не найден файл данных: {data_path}. "
            "Положите telecom_churn.csv рядом со скриптом или запустите из папки с CSV."
        )

    # 1. Загрузка данных
    df = pd.read_csv(data_path)
    print("Данные загружены. Первые 5 строк:")
    print(df.head())

    # 2. Изучение данных
    print("\nРазмерность данных:", df.shape)
    print("\nИнформация о столбцах:")
    df.info()

    print("\nСтатистика числовых признаков:")
    print(df.describe())

# Проверка пропусков
    print("\nКоличество пропусков в каждом столбце:")
    print(df.isnull().sum())

# 3. Целевая переменная – Churn
    print("\nРаспределение целевой переменной Churn:")
    print(df['Churn'].value_counts())
    print("Доля ушедших клиентов: {:.2f}%".format(df['Churn'].mean() * 100))

# Отделяем целевую переменную
    X = df.drop('Churn', axis=1)
    y = df['Churn']

# Разделение на обучающую и тестовую выборки (75% / 25%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y  # stratify для сохранения пропорции классов
    )
    print(f"\nРазмер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")

    categorical_features = ['State', 'Area code', 'International plan', 'Voice mail plan']
    # Проверим, что все перечисленные столбцы действительно категориальные и присутствуют
    print("\nКатегориальные признаки:", categorical_features)

    numerical_features = [col for col in X.columns if col not in categorical_features]
    print("Числовые признаки:", numerical_features)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
            # drop='first' для избежания мультиколлинеарности
        ])

# Применяем предобработку к тренировочным и тестовым данным
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print(f"\nФорма данных после обработки: X_train {X_train_processed.shape}, X_test {X_test_processed.shape}")

# Обучение моделей
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier()
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train_processed, y_train)
        y_pred = model.predict(X_test_processed)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"{name} - Accuracy: {acc:.4f}")

# Сравнение моделей
# Таблица результатов
    results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
    print("\nСравнение точности моделей:")
    print(results_df)

# График
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Accuracy', y='Model', hue='Model', data=results_df, palette='viridis', legend=False)
    plt.title('Сравнение точности моделей на тестовой выборке')
    plt.xlim(0, 1)
    plt.xlabel('Accuracy')
    plt.tight_layout()
    plt.show()

# Определение лучшей модели
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    print(f"\nЛучшая модель: {best_model_name} с точностью {results[best_model_name]:.4f}")

# Матрица ошибок для лучшей модели
    y_pred_best = best_model.predict(X_test_processed)
    cm = confusion_matrix(y_test, y_pred_best)

# Построим матрицу ошибок графически
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Остался (0)', 'Ушёл (1)'],
        yticklabels=['Остался (0)', 'Ушёл (1)'],
    )
    plt.title(f'Матрица ошибок для {best_model_name}')
    plt.ylabel('Истинный класс')
    plt.xlabel('Предсказанный класс')
    plt.show()

# Анализ матрицы ошибок
    tn, fp, fn, tp = cm.ravel()
    print("\nАнализ ошибок лучшей модели:")
    print(f"Верно предсказано лояльных (TN): {tn}")
    print(f"Ошибочно предсказаны как ушедшие (FP): {fp}")
    print(f"Ошибочно предсказаны как лояльные (FN): {fn}")
    print(f"Верно предсказано ушедших (TP): {tp}")


if __name__ == "__main__":
    main()

