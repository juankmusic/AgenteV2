# core/ai_core/nlp_embeddings.py

# ============================================
# IMPORTACIONES - Librerías que necesitamos
# ============================================
import os  # Para trabajar con variables de entorno y sistema operativo
import logging  # Para registrar mensajes de lo que hace el programa
import hashlib  # Para crear "huellas digitales" únicas de textos (hash)
import uuid  # Para generar identificadores únicos universales
import json  # Para trabajar con datos en formato JSON
from typing import Optional, List, Dict, Any  # Para indicar qué tipo de datos usamos

from dotenv import load_dotenv  # Para cargar variables de entorno desde archivo .env
from openai import OpenAI as OpenAIClient  # Cliente para conectarse a OpenAI
from db.connection import connect_db  # Nuestra función para conectar a la base de datos
import psycopg2  # Librería para trabajar con PostgreSQL
import numpy as np  # Para operaciones matemáticas con vectores

# ============================================
# CONFIGURACIÓN INICIAL
# ============================================
load_dotenv()  # Carga las variables de entorno desde el archivo .env

# Configurar el logger (cuaderno de registro del programa)
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Si no hay configuración previa, creamos una básica
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# ============================================
# OBTENER CLAVES DE API
# ============================================
# Las claves API son como contraseñas para usar los servicios de IA
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Clave de OpenAI
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # Clave de DeepSeek

# Verificar que tengamos la clave de OpenAI (es obligatoria)
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY no definida en entorno. Algunas funcionalidades no funcionarán.")

# ============================================
# INICIALIZAR CLIENTES DE IA
# ============================================
# Variables donde guardaremos las conexiones a los servicios
openai_client: Optional[OpenAIClient] = None
if OPENAI_API_KEY:
    openai_client = OpenAIClient(api_key=OPENAI_API_KEY)

# Cliente de DeepSeek (para uso futuro)
deepseek_client = None
if DEEPSEEK_API_KEY:
    try:
        # DeepSeek usa una API compatible con OpenAI
        deepseek_client = OpenAIClient(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
    except Exception as e:
        logger.warning("No se pudo inicializar deepseek_client: %s", e)
        deepseek_client = None

# ============================================
# FUNCIONES AUXILIARES INTERNAS
# ============================================

def _to_pgvector_literal(embedding: List[float]) -> str:
    """
    Convierte una lista de números (el embedding/vector) a un formato que PostgreSQL entiende.
    
    Los embeddings son listas de números como [0.1, 0.5, 0.3, ...]
    PostgreSQL necesita que se escriban como texto: '[0.1,0.5,0.3,...]'
    
    Esta función hace esa conversión.
    
    Parámetros:
        embedding: Lista de números decimales que representan el vector
        
    Retorna:
        Una cadena de texto con el formato '[num1,num2,num3,...]'
    """
    return "[" + ",".join(map(str, embedding)) + "]"

def _safe_close_cursor_conn(cur, conn):
    """
    Cierra de forma segura la conexión a la base de datos y el cursor.
    
    Un cursor es como un "puntero" que usamos para ejecutar consultas en la base de datos.
    Esta función intenta cerrar ambos, y si falla, no rompe el programa.
    
    Parámetros:
        cur: El cursor de la base de datos
        conn: La conexión a la base de datos
    """
    # Intentar cerrar el cursor
    try:
        if cur:
            cur.close()
    except Exception:
        pass  # Si falla, no importa, seguimos
    
    # Intentar cerrar la conexión
    try:
        if conn:
            conn.close()
    except Exception:
        pass  # Si falla, no importa, seguimos

def _extract_json_block(text: str) -> Optional[str]:
    """
    Busca y extrae un bloque JSON dentro de un texto.
    
    A veces la IA devuelve texto extra antes o después del JSON.
    Esta función encuentra el JSON buscando las llaves { y }.
    
    Parámetros:
        text: El texto donde buscar el JSON
        
    Retorna:
        El texto del JSON extraído, o None si no lo encuentra
    """
    # Verificar que sea una cadena de texto
    if not isinstance(text, str):
        return None
    
    # Buscar la primera llave de apertura { y la última de cierre }
    start = text.find("{")
    end = text.rfind("}")
    
    # Si encontramos ambas y están en el orden correcto, extraemos el JSON
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        return candidate
    
    return None  # No encontramos un JSON válido

# ============================================
# API DE EMBEDDINGS (funciones principales)
# ============================================

def generate_embedding(text: str) -> Optional[List[float]]:
    """
    Genera un embedding (vector numérico) para un texto usando OpenAI.
    
    Un embedding es como una "huella digital matemática" del texto.
    Textos similares tienen embeddings similares, aunque usen palabras diferentes.
    
    Por ejemplo:
    - "El gato está feliz" y "El minino está contento" tendrán embeddings parecidos
    - "El gato está feliz" y "La computadora está rota" tendrán embeddings diferentes
    
    Parámetros:
        text: El texto para el cual queremos generar el embedding
        
    Retorna:
        Una lista de números (el vector), o None si algo sale mal
    """
    # Verificar que tengamos conexión con OpenAI
    if not openai_client:
        logger.error("generate_embedding: cliente OpenAI no inicializado.")
        return None
    
    # Verificar que el texto no esté vacío
    if not text:
        logger.warning("generate_embedding: texto vacío.")
        return None
    
    try:
        # Llamar a la API de OpenAI para generar el embedding
        resp = openai_client.embeddings.create(
            model="text-embedding-3-small",  # Modelo específico de embeddings
            input=text  # El texto a convertir en vector
        )
        
        # Extraer el vector de la respuesta
        vector = resp.data[0].embedding
        
        # Asegurar que todos los valores sean números decimales (floats)
        return [float(x) for x in vector]
    except Exception as e:
        # Si algo sale mal, registrar el error y devolver None
        logger.exception("Error generando embedding: %s", e)
        return None

def store_embedding(text: str, source_filename: Optional[str] = None, page_number: Optional[int] = None, embedding: Optional[List[float]] = None) -> bool:
    """
    Guarda un texto y su embedding en la base de datos.
    
    Esta función almacena:
    - El texto original (chunk)
    - Su embedding (vector)
    - De dónde viene (archivo fuente y página)
    - Un hash único para evitar duplicados
    
    Parámetros:
        text: El texto a guardar
        source_filename: Nombre del archivo de origen (opcional)
        page_number: Número de página del documento (opcional)
        embedding: El vector del texto (si ya está calculado, para evitar recalcularlo)
        
    Retorna:
        True si se guardó correctamente, False si hubo error
    """
    conn = None  # Variable para la conexión a la base de datos
    cur = None   # Variable para el cursor de la base de datos
    
    try:
        # Si no nos dieron el embedding, lo generamos aquí
        if embedding is None:
            embedding = generate_embedding(text)
            if embedding is None:
                logger.warning("store_embedding: no se pudo generar embedding, se omite inserción.")
                return False

        # Crear un hash único del texto (como una huella digital)
        # Esto nos ayuda a detectar si ya guardamos este texto antes
        chunk_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        
        # Convertir el embedding al formato que PostgreSQL entiende
        embedding_literal = _to_pgvector_literal(embedding)

        # Conectar a la base de datos
        conn = connect_db()
        if not conn:
            logger.error("store_embedding: no se pudo conectar a DB.")
            return False
        cur = conn.cursor()

        # Insertar el texto y su embedding en la tabla document_embeddings
        # ON CONFLICT: si ya existe un texto con ese hash, no hacer nada (evitar duplicados)
        cur.execute("""
            INSERT INTO document_embeddings (id, text_chunk, source_filename, page_number, chunk_hash, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (chunk_hash) DO NOTHING
        """, (str(uuid.uuid4()), text, source_filename, page_number, chunk_hash, embedding_literal))

        # Guardar los cambios en la base de datos
        conn.commit()
        logger.debug("Embedding guardado para chunk hash=%s", chunk_hash)
        return True
    except Exception as e:
        # Si algo sale mal, registrar el error y revertir cambios
        logger.exception("Error guardando embedding: %s", e)
        if conn:
            try:
                conn.rollback()  # Deshacer cualquier cambio
            except Exception:
                pass
        return False
    finally:
        # Siempre cerrar la conexión y el cursor al terminar
        _safe_close_cursor_conn(cur, conn)

def search_similar_embeddings(query_text: str, top_k: int = 3, similarity_threshold: float = 0.0) -> List[Dict[str, Any]]:
    """
    Busca los textos más similares a una consulta en la base de datos.
    
    Esta función:
    1. Convierte la consulta en un embedding
    2. Compara ese embedding con todos los embeddings guardados en la base de datos
    3. Devuelve los textos más similares
    
    Es como buscar documentos relacionados usando "significado" en lugar de palabras exactas.
    
    Parámetros:
        query_text: El texto que queremos buscar
        top_k: Cuántos resultados queremos (por defecto 3)
        similarity_threshold: Qué tan similares deben ser (0.0 = todo, 1.0 = idénticos)
        
    Retorna:
        Una lista de diccionarios con los textos encontrados y su información
    """
    results: List[Dict[str, Any]] = []  # Lista donde guardaremos los resultados
    
    # Verificar que la consulta no esté vacía
    if not query_text:
        logger.debug("search_similar_embeddings: query vacía.")
        return results
    
    # Si no nos dieron top_k, usar 3 por defecto
    if top_k is None:
        top_k = 3
    
    try:
        # Generar el embedding de la consulta del usuario
        query_embedding = generate_embedding(query_text)
        if not query_embedding:
            logger.warning("search_similar_embeddings: no se pudo generar embedding de la query.")
            return results

        # Convertir el embedding al formato de PostgreSQL
        query_embedding_literal = _to_pgvector_literal(query_embedding)

        # Conectar a la base de datos
        conn = connect_db()
        if not conn:
            logger.error("search_similar_embeddings: no se pudo conectar a DB.")
            return results
        cur = conn.cursor()

        # Validar que top_k sea un número razonable (entre 1 y 200)
        top_k_int = int(top_k)
        if top_k_int <= 0 or top_k_int > 200:
            top_k_int = 3

        # Consulta SQL para buscar embeddings similares
        # El operador <=> calcula la distancia entre vectores
        # 1 - distancia = similitud (más cercano a 1 = más similar)
        sql = """
            SELECT text_chunk, source_filename, page_number, chunk_hash,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM document_embeddings
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        cur.execute(sql, (query_embedding_literal, query_embedding_literal, top_k_int))

        # Obtener todos los resultados
        rows = cur.fetchall()
        
        # Procesar cada resultado
        for r in rows:
            texto, src, page, chash, sim = r[0], r[1], r[2], r[3], r[4]
            sim_val = float(sim) if sim is not None else 0.0
            
            # Solo incluir resultados que superen el umbral de similitud
            if sim_val >= float(similarity_threshold):
                results.append({
                    "texto": texto,
                    "source_filename": src,
                    "page_number": page,
                    "chunk_hash": chash,
                    "similaridad": sim_val
                })
        
        logger.debug("search_similar_embeddings: encontrados %d resultados (threshold=%s)", len(results), similarity_threshold)
        
        # Cerrar cursor y conexión
        cur.close()
        conn.close()
        return results
    except Exception as e:
        # Si algo sale mal, registrar el error y devolver lista vacía
        logger.exception("Error en search_similar_embeddings: %s", e)
        return results

# ============================================
# EXTRACCIÓN DE ENTIDADES (NLP)
# ============================================

def extract_entities(text: str) -> Dict[str, Any]:
    """
    Extrae entidades importantes de un texto usando inteligencia artificial.
    
    Las entidades son cosas importantes como:
    - Personas (nombres)
    - Áreas (departamentos, equipos)
    - Virtudes (cualidades mencionadas)
    - Periodos (fechas, años)
    - Acciones (verbos importantes)
    
    Por ejemplo, del texto "Juan del equipo Omega tuvo buen rendimiento en 2024"
    extraería: persona=Juan, area=Omega, periodo=2024, virtud=buen rendimiento
    
    Parámetros:
        text: El texto del cual queremos extraer entidades
        
    Retorna:
        Un diccionario con las entidades encontradas, organizadas por categoría
    """
    # Verificar que tengamos conexión con OpenAI
    if not openai_client:
        logger.error("extract_entities: cliente OpenAI no inicializado.")
        return {"error": "OpenAI client not configured"}

    # Si el texto está vacío, devolver resultado vacío
    if not text:
        return {"texto": "", "entidades": {}}

    # Instrucciones para la IA sobre qué hacer
    prompt = f"""
    Extrae las entidades relevantes del siguiente texto.
    Clasifícalas en categorías: persona, area, virtud, periodo, accion, entidad_adicional.
    Devuelve la respuesta en formato JSON válido, sin texto adicional.

    Texto:
    {text}
    """

    try:
        # Llamar a la IA para que extraiga las entidades
        res = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Modelo de OpenAI a usar
            messages=[
                {"role": "system", "content": "Eres un extractor semántico. Responde sólo con JSON válido."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2  # Temperatura baja = respuestas más consistentes
        )
        
        # Obtener la respuesta de la IA
        raw_output = res.choices[0].message.content.strip()

        # Intentar extraer el JSON de la respuesta
        json_block = _extract_json_block(raw_output)
        if json_block:
            try:
                # Intentar convertir el bloque JSON a diccionario de Python
                parsed = json.loads(json_block)
                return parsed
            except json.JSONDecodeError:
                # Si falla, intentar parsear la respuesta completa
                try:
                    parsed = json.loads(raw_output)
                    return parsed
                except json.JSONDecodeError:
                    # Si nada funciona, devolver la respuesta cruda
                    logger.warning("extract_entities: no se pudo parsear JSON, devolviendo raw en entidades_raw.")
                    return {"entidades_raw": raw_output, "texto_original": text}
        else:
            # No hay bloque JSON: intentar parseo directo y luego fallback
            try:
                parsed = json.loads(raw_output)
                return parsed
            except json.JSONDecodeError:
                return {"entidades_raw": raw_output, "texto_original": text}

    except Exception as e:
        # Si algo sale mal, registrar el error
        logger.exception("Error en extract_entities: %s", e)
        return {"error": str(e), "texto_original": text}

# ============================================
# FUNCIÓN PRINCIPAL DE ANÁLISIS
# ============================================

def analyze_text(text: str) -> Dict[str, Any]:
    """
    Procesa un texto de forma completa:
    1. Genera su embedding (vector)
    2. Lo guarda en la base de datos
    3. Extrae las entidades importantes
    
    Es como hacer un análisis completo del texto en un solo paso.
    
    Parámetros:
        text: El texto a analizar
        
    Retorna:
        Un diccionario con:
        - texto: El texto original
        - embedding: El vector generado
        - entidades: Las entidades extraídas
    """
    logger.info("Analizando texto (len=%d)...", len(text) if text else 0)
    
    # Generar el embedding del texto (solo una vez para eficiencia)
    embedding = generate_embedding(text)
    
    if embedding:
        # Guardar el embedding en la base de datos (pasamos el embedding para no regenerarlo)
        stored = store_embedding(text, embedding=embedding)
        if not stored:
            logger.debug("analyze_text: store_embedding devolvió False para el chunk.")
    else:
        logger.debug("analyze_text: no se generó embedding.")

    # Extraer las entidades del texto
    entities = extract_entities(text)

    # Devolver todo junto
    return {
        "texto": text,
        "embedding": embedding,
        "entidades": entities
    }

# ============================================
# BÚSQUEDA SEMÁNTICA SOBRE ESQUEMA
# ============================================

def find_related_table(user_query: str, top_k: int = 3, similarity_threshold: float = 0.2) -> List[Dict[str, Any]]:
    """
    Busca las tablas de la base de datos más relacionadas con lo que pregunta el usuario.
    
    Esta función es muy útil para entender qué tabla consultar cuando el usuario
    hace una pregunta en lenguaje natural.
    
    Por ejemplo, si el usuario dice "quiero ver datos de empleados",
    esta función encontrará que la tabla "usuario" o "colaborador" es la más relevante.
    
    Parámetros:
        user_query: La pregunta o consulta del usuario
        top_k: Cuántas tablas candidatas devolver (por defecto 3)
        similarity_threshold: Qué tan relacionadas deben estar (0.2 por defecto)
        
    Retorna:
        Una lista de tablas candidatas con su información y nivel de similitud
    """
    results: List[Dict[str, Any]] = []  # Lista para guardar resultados
    
    # Verificar que la consulta no esté vacía
    if not user_query:
        logger.debug("find_related_table: consulta vacía.")
        return results

    try:
        # Generar embedding de la consulta del usuario
        query_embedding = generate_embedding(user_query)
        if not query_embedding:
            logger.warning("find_related_table: no se pudo generar embedding para la query.")
            return results

        # Convertir al formato de PostgreSQL
        query_embedding_literal = _to_pgvector_literal(query_embedding)

        # Conectar a la base de datos
        conn = connect_db()
        if not conn:
            logger.error("find_related_table: no se pudo conectar a DB.")
            return results

        cur = conn.cursor()

        # Buscar en embeddings que vengan del archivo semantic_schema
        # Estos embeddings contienen descripciones de las tablas del sistema
        sql = """
            SELECT text_chunk, 1 - (embedding <=> %s::vector) AS similarity
            FROM document_embeddings
            WHERE source_filename = 'semantic_schema'
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        cur.execute(sql, (query_embedding_literal, query_embedding_literal, top_k))

        # Obtener resultados
        rows = cur.fetchall()
        cur.close()
        conn.close()

        # Procesar cada resultado
        for text_chunk, sim in rows:
            sim_val = float(sim) if sim is not None else 0.0
            
            # Solo incluir si supera el umbral de similitud
            if sim_val < similarity_threshold:
                continue

            # Intentar extraer el nombre de la tabla y su descripción del texto
            table_name = None
            description = None
            for line in text_chunk.splitlines():
                if line.lower().startswith("tabla:"):
                    table_name = line.split(":", 1)[1].strip()
                elif line.lower().startswith("descripción:"):
                    description = line.split(":", 1)[1].strip()

            # Agregar la tabla candidata a los resultados
            results.append({
                "tabla": table_name or "desconocida",
                "descripcion": description or "",
                "similitud": sim_val,
                "texto_raw": text_chunk
            })

        logger.info("find_related_table: %d tablas candidatas encontradas.", len(results))
        return results

    except Exception as e:
        # Si algo sale mal, registrar el error y devolver lista vacía
        logger.exception("Error en find_related_table: %s", e)
        return results

# ============================================
# PERSISTENCIA DEL PLAN DE CONVERSACIÓN
# ============================================
# Estas funciones guardan y recuperan los planes de acción de la IA
# para mantener contexto entre mensajes del chat

def store_action_plan(session_id: str, plan_data: Dict[str, Any], executed_sql: Optional[str] = None) -> bool:
    """
    Guarda el plan de acción de la IA y el SQL ejecutado.
    
    Esto es importante para que la IA "recuerde" qué hizo en mensajes anteriores.
    Por ejemplo, si el usuario pidió "datos del equipo omega" y luego dice "muéstralo en gráfico",
    la IA necesita recordar qué consulta SQL usó antes.
    
    Parámetros:
        session_id: Identificador único de la sesión de chat
        plan_data: El plan de acción completo (qué, cómo, con qué datos)
        executed_sql: La consulta SQL que se ejecutó (si aplica)
        
    Retorna:
        True si se guardó correctamente, False si hubo error
    """
    conn = None  # Variable para la conexión
    cur = None   # Variable para el cursor
    
    try:
        # Añadir el SQL ejecutado al plan de datos
        plan_data['executed_sql'] = executed_sql

        # Convertir el plan a texto JSON (para guardarlo como string)
        plan_json_str = json.dumps(plan_data, ensure_ascii=False)
        
        # Crear un hash único para este plan (combinando session_id y datos únicos)
        chunk_hash = hashlib.sha256(f"{session_id}-{len(plan_json_str)}-{os.urandom(4).hex()}".encode("utf-8")).hexdigest()

        # Conectar a la base de datos
        conn = connect_db()
        if not conn:
            logger.error("store_action_plan: no se pudo conectar a DB.")
            return False
        cur = conn.cursor()

        # Guardar el plan en la tabla document_embeddings
        # Lo guardamos con source_filename especial para identificarlo después
        cur.execute("""
            INSERT INTO document_embeddings (id, text_chunk, source_filename, page_number, chunk_hash, embedding)
            VALUES (%s, %s, %s, %s, %s, NULL)
        """, (str(uuid.uuid4()), plan_json_str, f"chat_plan_{session_id}", 0, chunk_hash))

        # Guardar los cambios
        conn.commit()
        logger.debug("Plan de acción y SQL guardados para sesión %s", session_id)
        return True
    except Exception as e:
        # Si algo sale mal, registrar el error y revertir cambios
        logger.exception("Error guardando plan de acción: %s", e)
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        return False
    finally:
        # Siempre cerrar la conexión y el cursor
        _safe_close_cursor_conn(cur, conn)

def get_last_action_plan(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Recupera el último plan de acción guardado para una sesión de chat.
    
    Esto permite que la IA "recuerde" qué hizo en el mensaje anterior
    y pueda dar continuidad a la conversación.
    
    Por ejemplo, si el usuario pidió "datos de ventas" y luego dice "ahora en gráfico",
    la IA recupera el plan anterior para saber qué datos mostrar en el gráfico.
    
    Parámetros:
        session_id: Identificador único de la sesión de chat
        
    Retorna:
        El plan de acción más reciente (como diccionario), o None si no hay ninguno
    """
    conn = None  # Variable para la conexión
    cur = None   # Variable para el cursor
    
    try:
        # Conectar a la base de datos
        conn = connect_db()
        if not conn:
            logger.error("get_last_action_plan: no se pudo conectar a DB.")
            return None
        cur = conn.cursor()

        # Buscar el plan más reciente para este session_id
        # Ordenamos por id descendente para obtener el más nuevo primero
        sql = """
            SELECT text_chunk
            FROM document_embeddings
            WHERE source_filename = %s
            ORDER BY id DESC
            LIMIT 1
        """
        cur.execute(sql, (f"chat_plan_{session_id}",))

        # Obtener el resultado
        row = cur.fetchone()
        if row:
            # Convertir el texto JSON de vuelta a diccionario de Python
            plan_json_str = row[0]
            return json.loads(plan_json_str)
        
        return None  # No hay plan guardado para esta sesión
    except Exception as e:
        # Si algo sale mal, registrar el error y devolver None
        logger.exception("Error recuperando plan de acción: %s", e)
        return None
    finally:
        # Siempre cerrar la conexión y el cursor
        _safe_close_cursor_conn(cur, conn)