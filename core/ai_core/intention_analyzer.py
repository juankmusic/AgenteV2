# core/ai_core/intention_analyzer.py
import os
import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI as OpenAIClient 
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai_core.nlp_embeddings import generate_embedding, extract_entities

# ==============================================
# CARGAR CONFIGURACIONES Y CLIENTES
# ==============================================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

openai_client = OpenAIClient(api_key=OPENAI_API_KEY)
deepseek_client = OpenAIClient(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")

# ==============================================
# 1️⃣ INTENCIONES BASE (para embeddings semánticos)
# ==============================================
BASE_INTENTIONS = {
    "evaluar": "Evaluar o analizar el desempeño de una persona o equipo.",
    "generar_informe": "Crear o mostrar un informe o reporte visual.",
    "consultar": "Buscar información o preguntar sobre algo.",
    "guardar_resultado": "Guardar información o registrar un resultado.",
    "conversacion_general": "Conversación o saludo sin intención específica."
}

# Generar embeddings base una sola vez
BASE_EMBEDDINGS = {
    intent: generate_embedding(desc)
    for intent, desc in BASE_INTENTIONS.items()
}


# ==============================================
# 2️⃣ FUNCIÓN DE SIMILITUD SEMÁNTICA
# ==============================================
def cosine_similarity(vec1, vec2):
    if not vec1 or not vec2:
        return 0.0
    v1, v2 = np.array(vec1), np.array(vec2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


# ==============================================
# 3️⃣ DETECCIÓN DE INTENCIÓN CON DEEPSEEK
# ==============================================
def classify_intention_deepseek(text):
    """
    Usa DeepSeek para inferir qué tipo de acción quiere el usuario.
    """
    prompt = f"""
    Analiza el siguiente texto y clasifica la intención principal del usuario.
    Posibles tipos:
    - evaluar
    - generar_informe
    - consultar
    - guardar_resultado
    - conversacion_general

    Devuelve la respuesta en formato JSON válido:
    {{
        "tipo": "...",
        "subtipo": "...",
        "confianza": 0.0
    }}

    Texto:
    {text}
    """

    try:
        res = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un analista semántico experto en clasificación de intenciones. Devuelve solo JSON válido."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        raw_output = res.choices[0].message.content.strip()

        clean_output = (
            raw_output.replace("```json", "")
                      .replace("```", "")
                      .replace("\\n", "")
                      .replace("\n", "")
                      .strip()
        )

        return json.loads(clean_output)
    except Exception as e:
        print(f"⚠️ DeepSeek no pudo clasificar: {e}")
        return {"tipo": "desconocido", "subtipo": None, "confianza": 0.0}


# ==============================================
# 4️⃣ DETECCIÓN FINAL DE INTENCIÓN (híbrida)
# ==============================================
def detect_intention(text: str) -> dict:
    """
    Detecta la intención principal del texto combinando:
    - DeepSeek (razonamiento)
    - Embeddings (similitud semántica)
    """
    # 1️⃣ Predicción directa con DeepSeek
    deepseek_result = classify_intention_deepseek(text)

    # 2️⃣ Embedding del texto
    user_vec = generate_embedding(text)

    # 3️⃣ Calcular similitud con intenciones base
    similarities = {
        intent: cosine_similarity(user_vec, base_vec)
        for intent, base_vec in BASE_EMBEDDINGS.items()
    }

    best_match = max(similarities, key=similarities.get)
    best_score = similarities[best_match]

    # 4️⃣ Unificar resultados
    final_tipo = (
        deepseek_result.get("tipo")
        if deepseek_result.get("tipo") != "desconocido"
        else best_match
    )

    # 5️⃣ Calcular confianza ponderada (DeepSeek + similitud semántica)
    final_confidence = (
        0.6 * deepseek_result.get("confianza", 0.0)
        + 0.4 * best_score
    )

    # 6️⃣ Extraer entidades semánticas (usando nlp_embeddings)
    entidades = extract_entities(text)

    return {
        "tipo": final_tipo,
        "subtipo": deepseek_result.get("subtipo"),
        "confianza": round(final_confidence, 3),
        "entidades": entidades,
    }
