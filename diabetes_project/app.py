import pickle

import gradio as gr
import pandas as pd

with open("model.pkl", "rb") as f:
    model = pickle.load(f)


def predict(pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, dpf, age):
    X = pd.DataFrame([[
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, dpf, age
    ]], columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ])

    pred = model.predict(X)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0][1]
    else:
        proba = None

    if pred == 1:
        text = "Риск диабета повышен"
    else:
        text = "Риск диабета низкий"

    if proba is not None:
        text += f"\nВероятность диабета: {proba * 100:.2f}%"

    return text


with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown(
        """
        # Диагностика диабета
        Введите медицинские показатели пациента и получите прогноз модели.
        """
    )

    with gr.Row():
        with gr.Column():
            pregnancies = gr.Number(label="Беременности", value=1)
            glucose = gr.Number(label="Глюкоза", value=90)
            blood_pressure = gr.Number(label="Артериальное давление", value=60)
            skin_thickness = gr.Number(label="Толщина кожи", value=20)

        with gr.Column():
            insulin = gr.Number(label="Инсулин", value=80)
            bmi = gr.Number(label="ИМТ", value=22)
            dpf = gr.Number(label="Диабетический фактор", value=0.2)
            age = gr.Number(label="Возраст", value=25)

    predict_btn = gr.Button("Проверить")

    result = gr.Textbox(
        label="Результат",
        lines=3,
        interactive=False
    )

    predict_btn.click(
        fn=predict,
        inputs=[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age],
        outputs=result
    )

    gr.Examples(
        examples=[
            [1, 90, 60, 20, 80, 22, 0.2, 25],
            [6, 180, 90, 35, 200, 35, 1.2, 50],
        ],
        inputs=[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age],
    )

if __name__ == "__main__":
    app.launch()
