# core/ai_core/intention_analyzer.py

# ============================================
# IMPORTACIONES - Librerías que necesitamos
# ============================================
import os  # Para trabajar con variables de entorno y rutas del sistema
import json  # Para manejar datos en formato JSON
import logging  # Para registrar mensajes de lo que hace el programa
import numpy as np  # Para operaciones matemáticas con vectores
from dotenv import load_dotenv  # Para cargar variables de entorno desde archivo .env
from openai import OpenAI as OpenAIClient  # Cliente para conectarse a la API de OpenAI

# Importar funciones de otros archivos de nuestro proyecto
from core.ai_core.nlp_embeddings import generate_embedding, extract_entities

# ============================================
# CONFIGURACIÓN INICIAL
# ============================================
load_dotenv()  # Carga las variables de entorno desde el archivo .env
logger = logging.getLogger(__name__)  # Crea un logger para este archivo
if not logger.handlers:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))  # Configura el nivel de logging

# ==============================================
# CARGAR CONFIGURACIONES Y CLIENTES
# ==============================================
# Las claves API son como contraseñas para conectarnos a los servicios de IA
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Obtener la clave de OpenAI
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # Obtener la clave de DeepSeek

# Variables donde guardaremos las conexiones a los servicios de IA
openai_client = None
deepseek_client = None

# Intentar conectarse a los servicios
try:
    # Si tenemos la clave de OpenAI, creamos el cliente
    if OPENAI_API_KEY:
        openai_client = OpenAIClient(api_key=OPENAI_API_KEY)
    else:
        logger.warning("OPENAI_API_KEY no definida; algunas funciones de intención pueden fallar.")
    
    # Si tenemos la clave de DeepSeek, creamos su cliente (para uso futuro)
    if DEEPSEEK_API_KEY:
        deepseek_client = OpenAIClient(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
except Exception as e:
    # Si algo sale mal al conectar, registramos el error y seguimos sin clientes
    logger.warning("No se pudo inicializar cliente(s) de OpenAI/DeepSeek: %s", e)
    openai_client = openai_client or None
    deepseek_client = deepseek_client or None

# ==============================================
# 1️⃣ INTENCIONES BASE (para embeddings semánticos)
# ==============================================
# Este diccionario contiene las intenciones principales que el sistema puede detectar
# Cada intención tiene una descripción que se usará para comparar con lo que dice el usuario
BASE_INTENTIONS = {
    "evaluar": "Evaluar o analizar el desempeño de una persona o equipo.",
    "generar_informe": "Crear o mostrar un informe o reporte visual.",
    "consultar": "Buscar información o preguntar sobre algo.",
    "guardar_resultado": "Guardar información o registrar un resultado.",
    "conversacion_general": "Conversación o saludo sin intención específica."
}

# Variable para guardar los embeddings (vectores) de las intenciones base
# Usamos None para indicar que aún no se han generado (lazy loading)
_BASE_EMBEDDINGS_CACHE = None

def _init_base_embeddings():
    """
    Esta función genera los embeddings (vectores numéricos) para cada intención base.
    
    Los embeddings son como "huellas digitales" de las palabras que permiten comparar
    qué tan parecidas son dos frases aunque usen palabras diferentes.
    
    Solo se ejecuta UNA VEZ la primera vez que se necesita (lazy initialization).
    
    Retorna:
        Un diccionario con las intenciones y sus vectores correspondientes
    """
    global _BASE_EMBEDDINGS_CACHE  # Usamos la variable global
    
    # Si ya se generaron antes, devolvemos el cache
    if _BASE_EMBEDDINGS_CACHE is not None:
        return _BASE_EMBEDDINGS_CACHE
    
    cache = {}  # Diccionario donde guardaremos los vectores
    
    # Para cada intención y su descripción
    for intent, desc in BASE_INTENTIONS.items():
        try:
            # Generamos el vector (embedding) de la descripción
            vec = generate_embedding(desc)
            cache[intent] = vec  # Guardamos el vector (puede ser None si falla)
        except Exception as e:
            # Si algo sale mal, registramos el error y guardamos None
            logger.exception("Error generando embedding base para %s: %s", intent, e)
            cache[intent] = None
    
    # Guardamos el cache para no tener que regenerar los embeddings
    _BASE_EMBEDDINGS_CACHE = cache
    return cache

# ==============================================
# 2️⃣ FUNCIÓN DE SIMILITUD SEMÁNTICA
# ==============================================
def cosine_similarity(vec1, vec2):
    """
    Calcula qué tan parecidos son dos vectores usando similitud coseno.
    
    La similitud coseno mide el ángulo entre dos vectores:
    - 1.0 = totalmente iguales
    - 0.0 = completamente diferentes
    - -1.0 = totalmente opuestos
    
    Esta función es "segura" porque maneja casos especiales:
    - Si algún vector es None, devuelve 0.0
    - Si algún vector tiene norma 0, devuelve 0.0
    
    Parámetros:
        vec1: Primer vector a comparar
        vec2: Segundo vector a comparar
        
    Retorna:
        Un número entre 0.0 y 1.0 indicando qué tan similares son
    """
    try:
        # Si alguno de los vectores es None, no podemos comparar
        if vec1 is None or vec2 is None:
            return 0.0
        
        # Convertimos los vectores a arrays de numpy (para hacer cálculos matemáticos)
        v1 = np.array(vec1, dtype=float)
        v2 = np.array(vec2, dtype=float)
        
        # Calculamos la norma (magnitud) de cada vector
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        # Si alguna norma es 0, no podemos dividir, devolvemos 0
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Fórmula de similitud coseno: (v1 · v2) / (||v1|| * ||v2||)
        return float(np.dot(v1, v2) / (norm1 * norm2))
    except Exception as e:
        # Si algo sale mal en el cálculo, registramos el error y devolvemos 0
        logger.exception("Error en cosine_similarity: %s", e)
        return 0.0

# ==============================================
# FUNCIÓN AUXILIAR: Extraer JSON de texto
# ==============================================
def _extract_json_block(text: str):
    """
    Esta función busca un bloque JSON dentro de un texto.
    
    A veces el LLM (modelo de lenguaje) devuelve texto extra antes o después del JSON.
    Esta función encuentra el JSON buscando las llaves { y }.
    
    Parámetros:
        text: El texto donde buscar el JSON
        
    Retorna:
        El texto del JSON extraído, o None si no lo encuentra
    """
    # Verificar que el texto sea una cadena de texto
    if not isinstance(text, str):
        return None
    
    # Buscar la primera llave de apertura {
    start = text.find("{")
    # Buscar la última llave de cierre }
    end = text.rfind("}")
    
    # Si no encontramos ambas llaves, o están mal ordenadas, devolvemos None
    if start == -1 or end == -1 or end <= start:
        return None
    
    # Extraemos y devolvemos el texto entre las llaves (incluyéndolas)
    return text[start:end+1]

# ==============================================
# 3️⃣ DETECCIÓN DE INTENCIÓN CON DEEPSEEK (LLM)
# ==============================================
def classify_intention_deepseek(text):
    """
    Esta función usa un LLM (Large Language Model - modelo de lenguaje grande)
    para entender qué quiere hacer el usuario.
    
    El LLM es como un "cerebro artificial" que lee el texto y dice:
    "Esta persona quiere evaluar algo" o "Esta persona quiere un informe"
    
    Parámetros:
        text: El texto que escribió el usuario
        
    Retorna:
        Un diccionario con:
        - tipo: La intención detectada (evaluar, generar_informe, etc.)
        - subtipo: Un subtipo más específico (puede ser None)
        - confianza: Qué tan seguro está (0.0 a 1.0)
    """
    # Si no tenemos conexión con OpenAI, no podemos usar el LLM
    if not openai_client:
        logger.warning("classify_intention_deepseek: cliente OpenAI no inicializado.")
        return {"tipo": "desconocido", "subtipo": None, "confianza": 0.0}

    # <<< INICIO DE LA MEJORA DEL PROMPT >>>
    # Este es el "prompt" o instrucción que le damos al LLM
    # Le explicamos muy claramente qué debe hacer y cómo diferenciar las intenciones
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
        # Hacemos la petición al LLM (OpenAI GPT)
        res = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # El modelo específico de OpenAI que usamos
            messages=[
                # Mensaje del sistema: define el rol del LLM
                {"role": "system", "content": "Eres un analista semántico experto en clasificación de intenciones. Devuelve solo JSON válido."},
                # Mensaje del usuario: el texto a analizar
                {"role": "user", "content": prompt}
            ],
            temperature=0.1  # Temperatura baja = respuestas más predecibles y consistentes
        )
        
        # Obtenemos la respuesta del LLM
        raw_output = res.choices[0].message.content.strip()

        # Intentamos extraer el bloque JSON de la respuesta
        json_block = _extract_json_block(raw_output)
        
        if json_block:
            # Si encontramos el JSON, intentamos parsearlo
            try:
                parsed = json.loads(json_block)
                return {
                    "tipo": parsed.get("tipo"),
                    "subtipo": parsed.get("subtipo"),
                    "confianza": float(parsed.get("confianza", 0.0))
                }
            except json.JSONDecodeError:
                # Si falló el parseo del bloque, intentamos parsear la respuesta completa
                try:
                    parsed = json.loads(raw_output)
                    return {
                        "tipo": parsed.get("tipo"),
                        "subtipo": parsed.get("subtipo"),
                        "confianza": float(parsed.get("confianza", 0.0))
                    }
                except Exception as e:
                    # Si nada funcionó, registramos el error y devolvemos "desconocido"
                    logger.warning("classify_intention_deepseek: no se pudo parsear JSON del LLM: %s", e)
                    return {"tipo": "desconocido", "subtipo": None, "confianza": 0.0}
        else:
            # No encontramos bloque JSON, intentamos parsear la respuesta completa
            try:
                parsed = json.loads(raw_output)
                return {
                    "tipo": parsed.get("tipo"),
                    "subtipo": parsed.get("subtipo"),
                    "confianza": float(parsed.get("confianza", 0.0))
                }
            except Exception:
                # Si no hay JSON válido en ninguna parte, devolvemos "desconocido"
                logger.warning("classify_intention_deepseek: LLM no devolvió JSON, devolviendo desconocido.")
                return {"tipo": "desconocido", "subtipo": None, "confianza": 0.0}

    except Exception as e:
        # Si hubo algún error al comunicarnos con el LLM, lo registramos
        logger.exception("⚠️ LLM (OpenAI) no pudo clasificar: %s", e)
        return {"tipo": "desconocido", "subtipo": None, "confianza": 0.0}

# ==============================================
# 4️⃣ DETECCIÓN FINAL DE INTENCIÓN (híbrida)
# ==============================================
def detect_intention(text: str) -> dict:
    """
    ESTA ES LA FUNCIÓN PRINCIPAL DEL ARCHIVO.
    
    Detecta la intención del usuario combinando TRES métodos:
    1. LLM (modelo de lenguaje): pregunta a la IA qué quiere el usuario
    2. Embeddings semánticos: compara vectores para ver similitud
    3. Palabras clave: busca palabras específicas como "informe", "evaluar", etc.
    
    Es como tener 3 expertos dando su opinión y luego decidir la respuesta final.
    
    Parámetros:
        text: El texto que escribió el usuario
        
    Retorna:
        Un diccionario completo con:
        - tipo: La intención final detectada
        - subtipo: Un subtipo específico (si aplica)
        - confianza: Qué tan seguros estamos (0.0 a 1.0)
        - entidades: Cosas importantes extraídas del texto (nombres, fechas, etc.)
        - texto: El texto original del usuario
    """

    logger.debug("🔍 [DEBUG] Texto recibido para análisis de intención: %s", text)

    # ============================================
    # MÉTODO 1: Predicción directa con LLM
    # ============================================
    # Le preguntamos al modelo de lenguaje qué intención detecta
    deepseek_result = classify_intention_deepseek(text)
    logger.debug("🧠 [DEBUG] Resultado DeepSeek: %s", deepseek_result)

    # ============================================
    # MÉTODO 2: Embedding del texto del usuario
    # ============================================
    # Generamos el vector (embedding) del texto del usuario
    user_vec = None
    try:
        user_vec = generate_embedding(text)
    except Exception as e:
        logger.exception("Error generando embedding del usuario: %s", e)
        user_vec = None

    # Obtenemos los embeddings de las intenciones base (se generan solo una vez)
    base_embeddings = _init_base_embeddings()

    # Calculamos qué tan similar es el texto del usuario a cada intención base
    similarities = {}
    for intent, base_vec in base_embeddings.items():
        try:
            # Calculamos similitud entre el vector del usuario y el vector de la intención
            sim = cosine_similarity(user_vec, base_vec)
        except Exception as e:
            logger.exception("Error calculando similitud para %s: %s", intent, e)
            sim = 0.0
        similarities[intent] = sim

    # Elegimos la intención con mayor similitud
    if similarities:
        best_match = max(similarities, key=similarities.get)  # La intención más parecida
        best_score = float(similarities.get(best_match, 0.0) or 0.0)  # Su puntuación
    else:
        # Si no hay similitudes calculadas, usamos conversación general por defecto
        best_match = "conversacion_general"
        best_score = 0.0

    # ============================================
    # UNIFICAR RESULTADOS de LLM y embeddings
    # ============================================
    # Tomamos el tipo detectado por el LLM
    final_tipo = deepseek_result.get("tipo")
    
    # Si el LLM no está seguro O la similitud es baja, usamos el mejor match de embeddings
    if final_tipo in ["desconocido", None] or best_score < 0.55:
        final_tipo = best_match

    # ============================================
    # MÉTODO 3: Fallback con palabras clave
    # ============================================
    # Como última capa de seguridad, buscamos palabras clave específicas en el texto
    txt = (text or "").lower()  # Convertimos a minúsculas
    
    # Si encontramos palabras relacionadas con informes, es generar_informe
    if any(w in txt for w in ["informe", "reporte", "analisis", "mostrar", "muestrame", "tabla", "consultar", "ver datos"]):
        final_tipo = "generar_informe"
    # Si encontramos palabras relacionadas con evaluación, es evaluar
    elif any(w in txt for w in ["evaluar", "calificar", "desempeño", "rendimiento"]):
        final_tipo = "evaluar"
    # Si encontramos palabras relacionadas con guardar, es guardar_resultado
    elif any(w in txt for w in ["guardar", "registrar", "insertar"]):
        final_tipo = "guardar_resultado"
    else:
        # Si no encontramos nada específico, mantenemos lo detectado antes o conversación general
        final_tipo = final_tipo or "conversacion_general"

    # ============================================
    # CALCULAR CONFIANZA COMBINADA
    # ============================================
    # Combinamos la confianza del LLM (60%) con la similitud de embeddings (40%)
    deepseek_conf = float(deepseek_result.get("confianza", 0.0) or 0.0)
    final_confidence = round((0.6 * deepseek_conf + 0.4 * best_score), 3)

    # ============================================
    # EXTRAER ENTIDADES del texto
    # ============================================
    # Las entidades son cosas importantes como nombres de personas, fechas, lugares, etc.
    entidades = extract_entities(text)

    logger.info("✅ [DEBUG] Intención final detectada: %s (confianza=%s)", final_tipo, final_confidence)

    # ============================================
    # DEVOLVER RESULTADO FINAL
    # ============================================
    return {
        "tipo": final_tipo,  # La intención detectada
        "subtipo": deepseek_result.get("subtipo"),  # El subtipo (si hay)
        "confianza": final_confidence,  # Qué tan seguros estamos
        "entidades": entidades,  # Cosas importantes extraídas del texto
        "texto": text  # El texto original del usuario
    }