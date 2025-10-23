# core/ai_core/nlp_embeddings.py
import os
from openai import OpenAI as OpenAIClient
import hashlib
import uuid
import psycopg2
import numpy as np
import pandas as pd
from db.connection import connect_db
from dotenv import load_dotenv

# Cargar las variables de entorno 
load_dotenv()

# Claves de API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Clientes de API
openai_client = OpenAIClient(api_key=OPENAI_API_KEY)
deepseek_client = OpenAIClient(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")

# --------------------------------------------------------------
# FUNCIÓN PARA GENERAR EMBEDDINGS CON OPENAI
# --------------------------------------------------------------
def generate_embedding(text: str):
    """Genera un embedding y lo guarda en la base de datos."""
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        vector = response.data[0].embedding
        return vector
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return None
def store_embedding(text, source_filename=None, page_number=None):
    """Guarda el embedding del texto en la base de datos PostgreSQL."""
    try:
        conn = connect_db()
        cur = conn.cursor()
        chunk_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        embedding = generate_embedding(text)

        if not embedding:
            print("⚠️ No se generó embedding, se omite inserción.")
            return

        embedding_str = "[" + ",".join(map(str, embedding)) + "]"

        cur.execute("""
            INSERT INTO document_embeddings (id, text_chunk, source_filename, page_number, chunk_hash, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (chunk_hash) DO NOTHING
        """, (str(uuid.uuid4()), text, source_filename, page_number, chunk_hash, embedding_str))
        conn.commit()
        cur.close()
        conn.close()
        print(f"✅ Embedding guardado para chunk: {text[:50]}...")

    except Exception as e:
        print(f"❌ Error guardando embedding: {e}")
def search_similar_embeddings(query_text: str, top_k: int = 3):
    """
    Busca los textos más similares al embedding del texto dado usando pgvector.
    Devuelve los fragmentos de texto y su similitud.
    """
    try:
        conn = connect_db()
        cur = conn.cursor()

        # 1️⃣ Generar embedding para la consulta
        query_embedding = generate_embedding(query_text)
        if not query_embedding:
            print("⚠️ No se pudo generar el embedding de búsqueda.")
            return []

        # 2️⃣ Convertir a formato compatible con pgvector
        query_embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

        # 3️⃣ Ejecutar búsqueda semántica
        cur.execute(f"""
            SELECT 
                text_chunk, 
                1 - (embedding <=> %s::vector) AS similarity
            FROM document_embeddings
            ORDER BY embedding <=> %s::vector
            LIMIT {top_k};
        """, (query_embedding_str, query_embedding_str))

        rows = cur.fetchall()
        cur.close()
        conn.close()

        results = [{"texto": r[0], "similaridad": float(r[1])} for r in rows]
        return results

    except Exception as e:
        print(f"❌ Error en búsqueda semántica: {e}")
        return []

# --------------------------------------------------------------
# FUNCIÓN PARA EXTRAER ENTIDADES CLAVE USANDO DEEPSEEK
# --------------------------------------------------------------
def extract_entities(text: str) -> dict:
    """
    Utiliza el modelo de DeepSeek para detectar entidades o conceptos
    relevantes en el texto del usuario: nombres, fechas, áreas, virtudes, etc.
    """
    try:
        prompt = f"""
        Extrae las entidades relevantes del siguiente texto.
        Clasifícalas en categorías: persona, área, virtud, periodo, acción.
        Devuelve la respuesta en formato JSON válido, sin usar bloques de código ni texto adicional.

        Texto:
        {text}
        """

        res = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Usamos el modelo de OpenAI para consistencia mientras tanto
            messages=[
                {"role": "system", "content": "Eres un extractor semántico experto en análisis de texto. Devuelve siempre JSON válido."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        raw_output = res.choices[0].message.content.strip()

        # 🔹 Limpieza del formato (elimina ```json, ``` y \n extraños)
        clean_output = (
            raw_output.replace("```json", "")
                      .replace("```", "")
                      .replace("\\n", "")
                      .replace("\n", "")
                      .strip()
        )

        # 🔹 Intentar convertir a JSON
        import json
        try:
            entities = json.loads(clean_output)
        except json.JSONDecodeError:
            entities = {"texto": text, "entidades_raw": clean_output}

        return entities

    except Exception as e:
        print(f"❌ Error extrayendo entidades: {e}")
        return {"error": str(e)}

# --------------------------------------------------------------
# FUNCIÓN PRINCIPAL DE ANÁLISIS NLP COMPLETO
# --------------------------------------------------------------
def analyze_text(text: str) -> dict:
    """
    Procesa un texto: genera embedding + extrae entidades.
    Devuelve un diccionario estructurado con la información.
    """
    print(f"🧠 Analizando texto: {text[:60]}...")

    embedding = generate_embedding(text)

    # 🔹 Guardar el embedding en la base de datos
    if embedding:
        store_embedding(text)

    entities = extract_entities(text)

    return {
        "texto": text,
        "embedding": embedding,
        "entidades": entities
    }