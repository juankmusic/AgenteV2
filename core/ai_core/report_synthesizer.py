# core/ai_core/report_synthesizer.py
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import OpenAI as OpenAIClient

# ===========================
# CONFIGURACIÓN
# ===========================
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
deepseek_client = OpenAIClient(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")

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
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
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
    """
    Genera una visualización adaptativa si el usuario lo pide.
    """
    try:
        if any(word in user_input.lower() for word in ["gráfico", "grafico", "visualiza", "diagrama"]):
            counts = results.select_dtypes(include=["object"]).apply(lambda x: x.value_counts().head(5))
            counts.plot(kind="barh", figsize=(8, 5), title="Resumen visual automático")
            os.makedirs("static/reports", exist_ok=True)
            path = f"static/reports/visual_{hash(user_input)}.png"
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            return f"📊 Se generó un gráfico: {path}"
        return ""
    except Exception as e:
        return f"⚠️ No se pudo generar el gráfico: {e}"

# ===========================
# 3️⃣ INTERFAZ PRINCIPAL
# ===========================
def generate_report(results: pd.DataFrame, user_input: str) -> str:
    """
    Genera el informe final combinando análisis semántico + visualización opcional.
    """
    report_text = synthesize_from_results(results, user_input)
    visual_info = generate_visualization(results, user_input)
    return f"{report_text}\n\n{visual_info}"

# ===============================