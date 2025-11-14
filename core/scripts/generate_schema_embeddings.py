# core/scripts/generate_schema_embeddings.py
"""
Genera embeddings a partir del archivo semantic_schema.json
y los guarda en la tabla public.document_embeddings de PostgreSQL.
"""

import os
import sys
import json
import hashlib
import uuid
from datetime import datetime

import psycopg2
from openai import OpenAI
from dotenv import load_dotenv

# Permitir importar el módulo de conexión
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(BASE_DIR)
from db.connection import connect_db  # ✅ usa tu conexión existente

# Cargar variables de entorno
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ Falta la variable OPENAI_API_KEY en el archivo .env")

client = OpenAI(api_key=OPENAI_API_KEY)


# ======================================================
# FUNCIONES AUXILIARES
# ======================================================
def create_embedding(text: str):
    """Genera un embedding con el modelo text-embedding-3-small"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def calculate_hash(text: str) -> str:
    """Crea un hash único basado en el contenido"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def insert_embedding_to_db(id_val, text_chunk, source_filename, page_number, chunk_hash, embedding):
    conn = connect_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO public.document_embeddings
            (id, text_chunk, source_filename, page_number, chunk_hash, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (chunk_hash) DO UPDATE
            SET text_chunk = EXCLUDED.text_chunk,
                embedding = EXCLUDED.embedding;
        """, (id_val, text_chunk, source_filename, page_number, chunk_hash, embedding))
        conn.commit()
    except Exception as e:
        print(f"❌ Error al insertar embedding: {e}")
    finally:
        cur.close()
        conn.close()


# ======================================================
# PROCESO PRINCIPAL
# ======================================================
def main():
    schema_path = os.path.join(BASE_DIR, "core", "data", "semantic_schema.json")

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"❌ No se encontró el archivo {schema_path}")

    print("📘 Cargando archivo semantic_schema.json...")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    print("🧠 Generando embeddings para cada tabla del esquema semántico...\n")

    for table_name, data in schema.items():
        desc = data.get("description", "")
        cols = ", ".join(data["columns"].keys())
        syns = ", ".join(data.get("synonyms", []))

        # Texto que representará a esta tabla en el espacio vectorial
        text_block = (
            f"Tabla: {table_name}\n"
            f"Descripción: {desc}\n"
            f"Columnas: {cols}\n"
            f"Sinónimos: {syns}\n"
        )

        try:
            embedding = create_embedding(text_block)
            chunk_hash = calculate_hash(text_block)
            record_id = str(uuid.uuid4())

            insert_embedding_to_db(
                record_id,
                text_block,
                "semantic_schema",
                None,
                chunk_hash,
                embedding
            )

            print(f"✅ Embedding guardado para tabla '{table_name}'")

        except Exception as e:
            print(f"⚠️ Error procesando '{table_name}': {e}")

    print("\n✨ Proceso completado: embeddings del esquema generados con éxito.")


if __name__ == "__main__":
    main()