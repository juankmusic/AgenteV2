# core/ai_core/visual_engine.py
import io
import base64
import os
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI as OpenAIClient
# ===========================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAIClient(api_key=OPENAI_API_KEY)

def generate_visual_code(results: pd.DataFrame, user_query: str) -> str:
    """
    Pide al modelo que analice el DataFrame y genere código Python (matplotlib) para visualizarlo.
    """
    df_preview = results.head(10).to_markdown()

    prompt = f"""
    Eres un experto en análisis de datos y visualización en Python.
    El usuario ha solicitado: "{user_query}"

    Aquí hay una muestra de los datos:
    {df_preview}

    Analiza qué tipo de gráfico sería más informativo para esta solicitud.
    Devuelve únicamente el código Python que usa matplotlib para crear dicho gráfico.

    Reglas:
    - Usa plt.subplots() y plt.show() al final.
    - No incluyas código de carga de datos.
    - No imprimas texto ni descripciones.
    - No uses seaborn ni librerías externas.
    """

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Eres un experto en análisis de datos que genera código visual Python."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


def execute_visual_code(code: str, results: pd.DataFrame) -> str:
    """
    Ejecuta el código generado por el modelo y devuelve el gráfico como imagen en base64.
    """
    try:
        # Crear un entorno controlado
        safe_globals = {"plt": plt, "pd": pd, "results": results}
        exec(code, safe_globals)

        # Guardar imagen en buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        return f"⚠️ Error ejecutando el gráfico: {e}"


def generate_autonomous_visual(results: pd.DataFrame, user_query: str):
    """
    Combina razonamiento + ejecución del código visual.
    """
    code = generate_visual_code(results, user_query)
    image_data = execute_visual_code(code, results)
    return code, image_data
