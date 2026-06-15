import pickle
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Файл модели не найден: {MODEL_PATH}\n"
        "Сначала запустите обучение: python train.py"
    )

with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
feature_names = model_data["feature_names"]
model_name = model_data["model_name"]
errors_sample = model_data.get("errors_sample")

FEATURE_LABELS = {
    "Pregnancies": "Беременности",
    "Glucose": "Глюкоза",
    "BloodPressure": "Давление",
    "SkinThickness": "Толщина кожи",
    "Insulin": "Инсулин",
    "BMI": "ИМТ",
    "DiabetesPedigreeFunction": "Наследственность",
    "Age": "Возраст",
}


def get_feature_importance(pipeline, features: list[str]) -> pd.DataFrame | None:
    """Извлечение важности признаков из модели."""
    inner_model = pipeline.named_steps["model"]

    if hasattr(inner_model, "feature_importances_"):
        importances = inner_model.feature_importances_
    elif hasattr(inner_model, "coef_"):
        importances = np.abs(inner_model.coef_[0])
    else:
        return None

    return (
        pd.DataFrame({"Признак": features, "Важность": importances})
        .assign(Признак=lambda df: df["Признак"].map(lambda x: FEATURE_LABELS.get(x, x)))
        .sort_values("Важность", ascending=False)
    )


def predict(*args) -> tuple[str, pd.DataFrame | None]:
    X = pd.DataFrame([args], columns=feature_names)

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]

    label = "ПОВЫШЕН" if pred == 1 else "НИЗКИЙ"
    result_text = (
        f"## Риск диабета {label}\n"
        f"**Вероятность:** {proba * 100:.1f}%\n"
        f"*(Использована модель: {model_name})*"
    )

    importance_df = get_feature_importance(model, feature_names)
    return result_text, importance_df


with gr.Blocks(theme=gr.themes.Soft()) as demo:
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
                        title="Относительное влияние показателей",
                    )

            predict_btn.click(
                fn=predict,
                inputs=[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age],
                outputs=[output_text, importance_plot],
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
    demo.launch()
