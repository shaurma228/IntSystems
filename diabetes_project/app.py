import pickle

import gradio as gr
import numpy as np
import pandas as pd

# Загрузка модели и метаданных
with open("models/model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
feature_names = model_data["feature_names"]
model_name = model_data["model_name"]
errors_sample = model_data.get("errors_sample")


def get_feature_importance(pipeline, features):
    """Извлечение важности признаков из модели."""
    inner_model = pipeline.named_steps["model"]

    if hasattr(inner_model, "feature_importances_"):
        importances = inner_model.feature_importances_
    elif hasattr(inner_model, "coef_"):
        importances = np.abs(inner_model.coef_[0])
    else:
        return None

    df_importance = pd.DataFrame({
        "Признак": features,
        "Важность": importances
    }).sort_values(by="Важность", ascending=False)

    return df_importance


def predict(*args):
    # Создаем DataFrame из входа
    X = pd.DataFrame([args], columns=feature_names)

    # Предсказание
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]

    if pred == 1:
        result_text = "## Риск диабета ПОВЫШЕН"
    else:
        result_text = "## Риск диабета НИЗКИЙ"

    result_text += f"\n**Вероятность:** {proba * 100:.1f}%"
    result_text += f"\n*(Использована модель: {model_name})*"

    # График важности
    importance_df = get_feature_importance(model, feature_names)
    
    # Перевод названий признаков на русский для графика
    translation_map = {
        'Pregnancies': 'Беременности',
        'Glucose': 'Глюкоза',
        'BloodPressure': 'Давление',
        'SkinThickness': 'Толщина кожи',
        'Insulin': 'Инсулин',
        'BMI': 'ИМТ',
        'DiabetesPedigreeFunction': 'Наследственность',
        'Age': 'Возраст'
    }
    importance_df['Признак'] = importance_df['Признак'].map(lambda x: translation_map.get(x, x))

    return result_text, importance_df


# Создание интерфейса
with gr.Blocks() as demo:
    gr.Markdown(
        f"""
        # Система диагностики диабета
        Данная система использует машинное обучение для оценки риска наличия диабета.
        *Версия модели: {model_name} (F1-score на тесте: {model_data.get('f1_test', 0):.2f})*
        """
    )

    with gr.Tabs():
        with gr.TabItem("Диагностика"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Введите данные пациента")
                    pregnancies = gr.Number(label="Беременности", value=1, minimum=0, maximum=20)
                    glucose = gr.Number(label="Глюкоза (мг/дл)", value=100, minimum=0, maximum=300)
                    blood_pressure = gr.Number(label="Давление (мм рт.ст.)", value=70, minimum=0, maximum=200)
                    skin_thickness = gr.Number(label="Толщина кожи (мм)", value=20, minimum=0, maximum=100)
                    insulin = gr.Number(label="Инсулин (мкЕд/мл)", value=80, minimum=0, maximum=900)
                    bmi = gr.Number(label="ИМТ (кг/м²)", value=25, minimum=0, maximum=70)
                    dpf = gr.Number(label="Диабетический фактор", value=0.5, minimum=0, maximum=3)
                    age = gr.Number(label="Возраст", value=30, minimum=0, maximum=120)

                    predict_btn = gr.Button("Провести диагностику", variant="primary")

                with gr.Column(scale=1):
                    gr.Markdown("### Результат анализа")
                    output_text = gr.Markdown("Здесь появится результат после нажатия кнопки.")

                    gr.Markdown("### Вклад признаков в прогноз")
                    importance_plot = gr.BarPlot(
                        x="Важность",
                        y="Признак",
                        title="Относительное влияние показателей"
                    )

            predict_btn.click(
                fn=predict,
                inputs=[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age],
                outputs=[output_text, importance_plot]
            )

            gr.Examples(
                examples=[
                    [1, 90, 60, 20, 80, 22, 0.2, 25],
                    [6, 180, 90, 35, 200, 35, 1.2, 50],
                ],
                inputs=[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age],
            )

        with gr.TabItem("Анализ ошибок"):
            gr.Markdown("### Примеры, на которых модель ошиблась (Top 10)")
            if errors_sample is not None:
                gr.DataFrame(errors_sample)
            else:
                gr.Markdown("Ошибок на тестовой выборке не обнаружено или данные отсутствуют.")

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
