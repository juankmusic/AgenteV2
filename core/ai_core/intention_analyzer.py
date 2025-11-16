# core/ai_core/intention_analyzer.py
import os
import json
import logging
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI as OpenAIClient

# Import dependencias internas (usar import absoluto para evitar manipulación de sys.path)
from core.ai_core.nlp_embeddings import generate_embedding, extract_entities

load_dotenv()
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# ==============================================
# CARGAR CONFIGURACIONES Y CLIENTES
# ==============================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

openai_client = None
deepseek_client = None
try:
    if OPENAI_API_KEY:
        openai_client = OpenAIClient(api_key=OPENAI_API_KEY)
    else:
        logger.warning("OPENAI_API_KEY no definida; algunas funciones de intención pueden fallar.")
    if DEEPSEEK_API_KEY:
        # En este código mantenemos deepseek_client por compatibilidad futura
        deepseek_client = OpenAIClient(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
except Exception as e:
    logger.warning("No se pudo inicializar cliente(s) de OpenAI/DeepSeek: %s", e)
    openai_client = openai_client or None
    deepseek_client = deepseek_client or None

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

# No generar embeddings en import-time: lazy init
_BASE_EMBEDDINGS_CACHE = None

def _init_base_embeddings():
    global _BASE_EMBEDDINGS_CACHE
    if _BASE_EMBEDDINGS_CACHE is not None:
        return _BASE_EMBEDDINGS_CACHE
    cache = {}
    for intent, desc in BASE_INTENTIONS.items():
        try:
            vec = generate_embedding(desc)
            cache[intent] = vec  # puede ser None si la generación falla
        except Exception as e:
            logger.exception("Error generando embedding base para %s: %s", intent, e)
            cache[intent] = None
    _BASE_EMBEDDINGS_CACHE = cache
    return cache

# ==============================================
# 2️⃣ FUNCIÓN DE SIMILITUD SEMÁNTICA
# ==============================================
def cosine_similarity(vec1, vec2):
    """
    Devuelve similitud coseno segura entre dos vectores.
    Si alguno es None o tiene norma 0, devuelve 0.0.
    """
    try:
        if vec1 is None or vec2 is None:
            return 0.0
        v1 = np.array(vec1, dtype=float)
        v2 = np.array(vec2, dtype=float)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))
    except Exception as e:
        logger.exception("Error en cosine_similarity: %s", e)
        return 0.0

# Helper para extraer bloque JSON de salida del LLM
def _extract_json_block(text: str):
    if not isinstance(text, str):
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start:end+1]

# ==============================================
# 3️⃣ DETECCIÓN DE INTENCIÓN CON DEEPSEEK (LLM)
# ==============================================
def classify_intention_deepseek(text):
    """
    Usa un LLM (aquí openai_client) para inferir la intención.
    Devuelve dict con keys: tipo, subtipo, confianza.
    Mantiene contrato original, pero hace parsing robusto.
    """
    if not openai_client:
        logger.warning("classify_intention_deepseek: cliente OpenAI no inicializado.")
        return {"tipo": "desconocido", "subtipo": None, "confianza": 0.0}

    # <<< INICIO DE LA MEJORA DEL PROMPT >>>
    prompt = f"""
    Analiza el siguiente texto y clasifica la intención principal del usuario.

    Aquí están las posibles intenciones y cómo diferenciarlas:

    1.  **evaluar**: Cuando el usuario quiere analizar el rendimiento o desempeño de algo/alguien.
        * Ejemplos: "Cómo fue el rendimiento de Ana?", "Califica al equipo de ventas"

    2.  **generar_informe**: Cuando el usuario pide un análisis complejo, un resumen de datos, una comparación, o datos para un gráfico.
        * Ejemplos: "Analiza las ventas del último trimestre", "Dame un resumen de los gastos", "Compara el producto A vs B", "Muéstrame los datos de ventas por región"

    3.  **consultar**: Cuando el usuario busca un dato específico, un hecho o una lista simple. No requiere análisis, solo búsqueda.
        * Ejemplos: "¿Cuál es el email de Juan?", "¿Cuántos empleados hay en el departamento de IT?", "Lista los productos en stock"

    4.  **guardar_resultado**: Cuando el usuario pide explícitamente guardar, registrar o insertar datos.
        * Ejemplos: "Guarda esta nota", "Registra una nueva venta"

    5.  **conversacion_general**: Saludos, despedidas o charla casual.
        * Ejemplos: "Hola", "Cómo estás?", "Gracias"

    Devuelve la respuesta en formato JSON válido:
    {{
        "tipo": "...",
        "subtipo": "...",
        "confianza": 0.0
    }}

    Texto a analizar:
    "{text}"
    """
    # <<< FIN DE LA MEJORA DEL PROMPT >>>

    try:
        res = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un analista semántico experto en clasificación de intenciones. Devuelve solo JSON válido."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1 # <-- Bajar la temperatura para que sea más predecible
        )
        raw_output = res.choices[0].message.content.strip()

        # ... (El resto de tu lógica de parsing de JSON está bien) ...
        json_block = _extract_json_block(raw_output)
        if json_block:
            try:
                parsed = json.loads(json_block)
                return {
                    "tipo": parsed.get("tipo"),
                    "subtipo": parsed.get("subtipo"),
                    "confianza": float(parsed.get("confianza", 0.0))
                }
            except json.JSONDecodeError:
                # Intentar parsear raw completo
                try:
                    parsed = json.loads(raw_output)
                    return {
                        "tipo": parsed.get("tipo"),
                        "subtipo": parsed.get("subtipo"),
                        "confianza": float(parsed.get("confianza", 0.0))
                    }
                except Exception as e:
                    logger.warning("classify_intention_deepseek: no se pudo parsear JSON del LLM: %s", e)
                    return {"tipo": "desconocido", "subtipo": None, "confianza": 0.0}
        else:
            # No bloque JSON: intentar parseo directo o fallback
            try:
                parsed = json.loads(raw_output)
                return {
                    "tipo": parsed.get("tipo"),
                    "subtipo": parsed.get("subtipo"),
                    "confianza": float(parsed.get("confianza", 0.0))
                }
            except Exception:
                logger.warning("classify_intention_deepseek: LLM no devolvió JSON, devolviendo desconocido.")
                return {"tipo": "desconocido", "subtipo": None, "confianza": 0.0}

    except Exception as e:
        logger.exception("⚠️ LLM (OpenAI) no pudo clasificar: %s", e)
        return {"tipo": "desconocido", "subtipo": None, "confianza": 0.0}

# ==============================================
# 4️⃣ DETECCIÓN FINAL DE INTENCIÓN (híbrida)
# ==============================================
def detect_intention(text: str) -> dict:
    """
    Detecta la intención combinando:
    - clasificación LLM (classify_intention_deepseek)
    - similitud con intenciones base (embeddings)
    - fallback por palabras clave
    Devuelve dict con keys: tipo, subtipo, confianza, entidades, texto
    """

    logger.debug("🔍 [DEBUG] Texto recibido para análisis de intención: %s", text)

    # 1️⃣ Predicción directa con LLM
    deepseek_result = classify_intention_deepseek(text)
    logger.debug("🧠 [DEBUG] Resultado DeepSeek: %s", deepseek_result)

    # 2️⃣ Embedding del texto
    user_vec = None
    try:
        user_vec = generate_embedding(text)
    except Exception as e:
        logger.exception("Error generando embedding del usuario: %s", e)
        user_vec = None

    # 3️⃣ Obtener embeddings base (lazy init)
    base_embeddings = _init_base_embeddings()

    # 3.1 Calcular similitudes de forma segura
    similarities = {}
    for intent, base_vec in base_embeddings.items():
        try:
            sim = cosine_similarity(user_vec, base_vec)
        except Exception as e:
            logger.exception("Error calculando similitud para %s: %s", intent, e)
            sim = 0.0
        similarities[intent] = sim

    # Elegir mejor match (manejar caso en que similarities esté vacío)
    if similarities:
        best_match = max(similarities, key=similarities.get)
        best_score = float(similarities.get(best_match, 0.0) or 0.0)
    else:
        best_match = "conversacion_general"
        best_score = 0.0

    # 4️⃣ Unificar resultados
    final_tipo = deepseek_result.get("tipo")
    if final_tipo in ["desconocido", None] or best_score < 0.55:
        final_tipo = best_match

    # 5️⃣ Fallback semántico adicional por palabras clave (sin cambiar la lógica original)
    txt = (text or "").lower()
    if any(w in txt for w in ["informe", "reporte", "analisis", "mostrar", "muestrame", "tabla", "consultar", "ver datos"]):
        final_tipo = "generar_informe"
    elif any(w in txt for w in ["evaluar", "calificar", "desempeño", "rendimiento"]):
        final_tipo = "evaluar"
    elif any(w in txt for w in ["guardar", "registrar", "insertar"]):
        final_tipo = "guardar_resultado"
    else:
        final_tipo = final_tipo or "conversacion_general"

    # 6️⃣ Calcular confianza combinada y extraer entidades
    deepseek_conf = float(deepseek_result.get("confianza", 0.0) or 0.0)
    final_confidence = round((0.6 * deepseek_conf + 0.4 * best_score), 3)

    entidades = extract_entities(text)

    logger.info("✅ [DEBUG] Intención final detectada: %s (confianza=%s)", final_tipo, final_confidence)

    return {
        "tipo": final_tipo,
        "subtipo": deepseek_result.get("subtipo"),
        "confianza": final_confidence,
        "entidades": entidades,
        "texto": text
    }