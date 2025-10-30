# core/ai_core/nlp_embeddings.py
import os
import logging
import hashlib
import uuid
import json
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI as OpenAIClient
from db.connection import connect_db
import psycopg2
import numpy as np

load_dotenv()

logger = logging.getLogger(__name__)
if not logger.handlers:
    # Si no hay handlers configurados en la app, configuramos uno básico
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# Claves de API (validadas)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY no definida en entorno. Algunas funcionalidades no funcionarán.")
# Inicializar cliente OpenAI si hay key
openai_client: Optional[OpenAIClient] = None
if OPENAI_API_KEY:
    openai_client = OpenAIClient(api_key=OPENAI_API_KEY)

# Si en el futuro DeepSeek tiene SDK distinto, inicializarlo aquí.
deepseek_client = None
if DEEPSEEK_API_KEY:
    try:
        # Si Deepseek usa la misma API REST compatible con OpenAI, dejar config; si no, sustituir.
        deepseek_client = OpenAIClient(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
    except Exception as e:
        logger.warning("No se pudo inicializar deepseek_client: %s", e)
        deepseek_client = None

# ------------------------------
# Utilities internas
# ------------------------------
def _to_pgvector_literal(embedding: List[float]) -> str:
    """Convierte lista de floats a literal '[v1,v2,...]' para insertar/castear en pgvector.
    Nota: depende del tipo de columna en la BD; si usas tipo vector de pgvector la query debe usar '%s::vector'."""
    return "[" + ",".join(map(str, embedding)) + "]"

def _safe_close_cursor_conn(cur, conn):
    try:
        if cur:
            cur.close()
    except Exception:
        pass
    try:
        if conn:
            conn.close()
    except Exception:
        pass

def _extract_json_block(text: str) -> Optional[str]:
    """
    Intenta extraer un bloque JSON del texto (p. ej. contenido entre primer '{' y último '}').
    Si no encuentra algo que parezca JSON válido, devuelve None.
    """
    if not isinstance(text, str):
        return None
    # Buscar primer y último brace que probablemente delimiten JSON
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        return candidate
    return None

# ------------------------------
# API de embeddings
# ------------------------------
def generate_embedding(text: str) -> Optional[List[float]]:
    """
    Genera embedding con OpenAI y devuelve lista de floats.
    Retorna None en caso de error.
    """
    if not openai_client:
        logger.error("generate_embedding: cliente OpenAI no inicializado.")
        return None
    if not text:
        logger.warning("generate_embedding: texto vacío.")
        return None
    try:
        resp = openai_client.embeddings.create(model="text-embedding-3-small", input=text)
        vector = resp.data[0].embedding
        # Asegurar que sea lista de floats
        return [float(x) for x in vector]
    except Exception as e:
        logger.exception("Error generando embedding: %s", e)
        return None

def store_embedding(text: str, source_filename: Optional[str] = None, page_number: Optional[int] = None, embedding: Optional[List[float]] = None) -> bool:
    """
    Guarda en la tabla document_embeddings un chunk y su embedding.
    Si embedding es None, lo genera aquí (pero preferible pasar embedding para evitar doble llamada).
    Devuelve True si se guardó o ya existía, False si hubo error.
    """
    conn = None
    cur = None
    try:
        if embedding is None:
            embedding = generate_embedding(text)
            if embedding is None:
                logger.warning("store_embedding: no se pudo generar embedding, se omite inserción.")
                return False

        # Normalizar hash
        chunk_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        embedding_literal = _to_pgvector_literal(embedding)

        conn = connect_db()
        if not conn:
            logger.error("store_embedding: no se pudo conectar a DB.")
            return False
        cur = conn.cursor()

        # Insert parametrizado; asumimos que la columna embedding aceptará literal con casting en la consulta cliente (p. ej. '%s'::vector)
        cur.execute("""
            INSERT INTO document_embeddings (id, text_chunk, source_filename, page_number, chunk_hash, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (chunk_hash) DO NOTHING
        """, (str(uuid.uuid4()), text, source_filename, page_number, chunk_hash, embedding_literal))

        conn.commit()
        logger.debug("Embedding guardado para chunk hash=%s", chunk_hash)
        return True
    except Exception as e:
        logger.exception("Error guardando embedding: %s", e)
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        return False
    finally:
        _safe_close_cursor_conn(cur, conn)

def search_similar_embeddings(query_text: str, top_k: int = 3, similarity_threshold: float = 0.0) -> List[Dict[str, Any]]:
    """
    Genera embedding de la query y busca los top_k chunks más similares en la BD usando pgvector.
    Devuelve lista de objetos: {texto, similaridad, source_filename, page_number, chunk_hash}
    """
    results: List[Dict[str, Any]] = []
    if not query_text:
        logger.debug("search_similar_embeddings: query vacía.")
        return results
    if top_k is None:
        top_k = 3
    try:
        query_embedding = generate_embedding(query_text)
        if not query_embedding:
            logger.warning("search_similar_embeddings: no se pudo generar embedding de la query.")
            return results

        query_embedding_literal = _to_pgvector_literal(query_embedding)

        conn = connect_db()
        if not conn:
            logger.error("search_similar_embeddings: no se pudo conectar a DB.")
            return results
        cur = conn.cursor()

        # Validar top_k para evitar inyección al construir la query si fuera necesario
        top_k_int = int(top_k)
        if top_k_int <= 0 or top_k_int > 200:
            top_k_int = 3

        # Query parametrizada: pasamos el literal y el límite como parámetro
        # NOTA: si la columna embedding es tipo vector de pgvector, la comparación funciona con '%s'::vector
        sql = """
            SELECT text_chunk, source_filename, page_number, chunk_hash,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM document_embeddings
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        cur.execute(sql, (query_embedding_literal, query_embedding_literal, top_k_int))

        rows = cur.fetchall()
        # filas: (text_chunk, source_filename, page_number, chunk_hash, similarity)
        for r in rows:
            texto, src, page, chash, sim = r[0], r[1], r[2], r[3], r[4]
            sim_val = float(sim) if sim is not None else 0.0
            if sim_val >= float(similarity_threshold):
                results.append({
                    "texto": texto,
                    "source_filename": src,
                    "page_number": page,
                    "chunk_hash": chash,
                    "similaridad": sim_val
                })
        logger.debug("search_similar_embeddings: encontrados %d resultados (threshold=%s)", len(results), similarity_threshold)
        cur.close()
        conn.close()
        return results
    except Exception as e:
        logger.exception("Error en search_similar_embeddings: %s", e)
        return results

# ------------------------------
# Extracción de entidades (NLP)
# ------------------------------
def extract_entities(text: str) -> Dict[str, Any]:
    """
    Extrae entidades relevantes del texto usando LLM.
    Se espera que el LLM devuelva JSON con keys categorizadas.
    Retorna un dict con las entidades o un campo 'entidades_raw' en caso de fallo de parseo.
    """
    if not openai_client:
        logger.error("extract_entities: cliente OpenAI no inicializado.")
        return {"error": "OpenAI client not configured"}

    if not text:
        return {"texto": "", "entidades": {}}

    prompt = f"""
    Extrae las entidades relevantes del siguiente texto.
    Clasifícalas en categorías: persona, area, virtud, periodo, accion, entidad_adicional.
    Devuelve la respuesta en formato JSON válido, sin texto adicional.

    Texto:
    {text}
    """

    try:
        res = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un extractor semántico. Responde sólo con JSON válido."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        raw_output = res.choices[0].message.content.strip()

        # Intentar parsear buscando un bloque JSON en la salida
        json_block = _extract_json_block(raw_output)
        if json_block:
            try:
                parsed = json.loads(json_block)
                return parsed
            except json.JSONDecodeError:
                # Intentar parseo del raw completo (por si LLM devolvió JSON sin texto adicional)
                try:
                    parsed = json.loads(raw_output)
                    return parsed
                except json.JSONDecodeError:
                    logger.warning("extract_entities: no se pudo parsear JSON, devolviendo raw en entidades_raw.")
                    return {"entidades_raw": raw_output, "texto_original": text}
        else:
            # No hay bloque JSON claro: intentar parseo directo y luego fallback
            try:
                parsed = json.loads(raw_output)
                return parsed
            except json.JSONDecodeError:
                return {"entidades_raw": raw_output, "texto_original": text}

    except Exception as e:
        logger.exception("Error en extract_entities: %s", e)
        return {"error": str(e), "texto_original": text}

# ------------------------------
# Función principal de análisis
# ------------------------------
def analyze_text(text: str) -> Dict[str, Any]:
    """
    Procesa un texto: genera embedding (una sola vez), guarda el embedding y extrae entidades.
    Retorna dict consistente con keys: texto, embedding, entidades.
    """
    logger.info("Analizando texto (len=%d)...", len(text) if text else 0)
    embedding = generate_embedding(text)
    if embedding:
        # Pasar embedding para evitar regenerarlo en store_embedding
        stored = store_embedding(text, embedding=embedding)
        if not stored:
            logger.debug("analyze_text: store_embedding devolvió False para el chunk.")
    else:
        logger.debug("analyze_text: no se generó embedding.")

    entities = extract_entities(text)

    return {
        "texto": text,
        "embedding": embedding,
        "entidades": entities
    }
# ------------------------------
# Búsqueda semántica sobre esquema (semantic_schema)
# ------------------------------

def find_related_table(user_query: str, top_k: int = 3, similarity_threshold: float = 0.2) -> List[Dict[str, Any]]:
    """
    Busca las tablas del esquema más relacionadas con la consulta del usuario,
    usando los embeddings del archivo semantic_schema.json almacenados en document_embeddings.
    Devuelve lista con {tabla, descripcion, similitud, texto_raw}.
    """

    results: List[Dict[str, Any]] = []
    if not user_query:
        logger.debug("find_related_table: consulta vacía.")
        return results

    try:
        # Generar embedding de la consulta del usuario
        query_embedding = generate_embedding(user_query)
        if not query_embedding:
            logger.warning("find_related_table: no se pudo generar embedding para la query.")
            return results

        query_embedding_literal = _to_pgvector_literal(query_embedding)

        conn = connect_db()
        if not conn:
            logger.error("find_related_table: no se pudo conectar a DB.")
            return results

        cur = conn.cursor()

        sql = """
            SELECT text_chunk, 1 - (embedding <=> %s::vector) AS similarity
            FROM document_embeddings
            WHERE source_filename = 'semantic_schema'
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        cur.execute(sql, (query_embedding_literal, query_embedding_literal, top_k))

        rows = cur.fetchall()
        cur.close()
        conn.close()

        for text_chunk, sim in rows:
            sim_val = float(sim) if sim is not None else 0.0
            if sim_val < similarity_threshold:
                continue

            # Intentar extraer nombre de tabla y descripción del texto almacenado
            table_name = None
            description = None
            for line in text_chunk.splitlines():
                if line.lower().startswith("tabla:"):
                    table_name = line.split(":", 1)[1].strip()
                elif line.lower().startswith("descripción:"):
                    description = line.split(":", 1)[1].strip()

            results.append({
                "tabla": table_name or "desconocida",
                "descripcion": description or "",
                "similitud": sim_val,
                "texto_raw": text_chunk
            })

        logger.info("find_related_table: %d tablas candidatas encontradas.", len(results))
        return results

    except Exception as e:
        logger.exception("Error en find_related_table: %s", e)
        return results