# core/ai_core/report_synthesizer.py
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from dotenv import load_dotenv
from openai import OpenAI as OpenAIClient

# ===========================
# CONFIGURACIÓN
# ===========================
load_dotenv()
#DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
#deepseek_client = OpenAIClient(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAIClient(api_key=OPENAI_API_KEY)
# ===========================
# SÍNTESIS DE RESULTADOS
# ===========================
def synthesize_from_results(results: pd.DataFrame, user_input: str) -> str:
    """
    Toma resultados (ya consultados) y genera una interpretación natural y contextual.
    """
    if results.empty:
        return "No se encontraron datos relevantes para tu solicitud."

    # 🔹 Conversión segura para serialización JSON
    safe_results = results.copy()
    for col in safe_results.columns:
        safe_results[col] = safe_results[col].astype(str)

    data_sample = safe_results.head(10).to_dict(orient="records")

    prompt = f"""
    Eres un agente inteligente que debe analizar resultados obtenidos desde una base de datos.
    El usuario solicitó: "{user_input}"

    Estos son los datos relevantes:
    {json.dumps(data_sample, ensure_ascii=False, indent=2)}

    Tu tarea es:
    1. Comprender qué representan los datos en el contexto del usuario.
    2. Generar un informe claro, natural y analítico.
    3. Si el usuario pidió un gráfico, sugiere cuál sería el tipo adecuado.
    """


    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un analista cognitivo experto. Redacta informes claros y naturales en español."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generando informe: {e}"

# ===========================
# 2️⃣ OPCIONAL: VISUALIZACIÓN
# ===========================
def generate_visualization(results: pd.DataFrame, user_input: str) -> str:
    import io, base64, matplotlib.pyplot as plt

    try:
        if not any(w in user_input.lower() for w in ["gráfico", "grafico", "visualiza", "diagrama", "plot", "ver"]):
            return ""

        numeric_cols = results.select_dtypes(include=["number"]).columns
        categorical_cols = results.select_dtypes(include=["object", "category"]).columns

        plt.figure(figsize=(8, 5))

        if len(numeric_cols) >= 2:
            results[numeric_cols].corr().plot(kind="heatmap", cmap="coolwarm")
            plt.title("Mapa de correlación")
        elif len(numeric_cols) == 1:
            col = numeric_cols[0]
            results[col].plot(kind="hist", bins=10, alpha=0.7)
            plt.title(f"Distribución de {col}")
        elif len(categorical_cols) >= 1:
            col = categorical_cols[0]
            results[col].value_counts().head(10).plot(kind="barh", color="skyblue")
            plt.title(f"Frecuencia de {col}")
        else:
            plt.text(0.5, 0.5, "No hay datos visualizables", ha="center")

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()

        return f'<div style="text-align:center;margin-top:10px;"><img src="data:image/png;base64,{img_base64}" alt="Gráfico generado" style="max-width:100%;border-radius:12px;box-shadow:0 0 8px rgba(0,0,0,0.3)"></div>'
    except Exception as e:
        return f"<p style='color:#f87171;'>⚠️ No se pudo generar el gráfico: {e}</p>"

# ===========================
# 3️⃣ INTERFAZ PRINCIPAL
# ===========================
def generate_report(results: pd.DataFrame, user_input: str) -> str:
    report_text = synthesize_from_results(results, user_input)
    visual_html = generate_visualization(results, user_input)
    return f"<div>{report_text}</div>{visual_html}"


# ===============================