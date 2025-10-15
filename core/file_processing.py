# core/file_processing.py
import os
import re
import uuid
import pandas as pd
import pypdf
from core.chatbot_logic import OPENAI_CLIENT
from db.connection import connect_db


# ===============================
# EMBEDDINGS
# ===============================
def get_embeddings(text_list, model="text-embedding-3-small"):
    try:
        res = OPENAI_CLIENT.embeddings.create(model=model, input=text_list)
        return [r.embedding for r in res.data]
    except Exception as e:
        print(f"❌ Error generando embeddings: {e}")
        return [None] * len(text_list)


def save_chunk(text, source_filename, page_number, chunk_hash, embedding):
    conn = connect_db()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO document_embeddings (id, text_chunk, source_filename, page_number, chunk_hash, embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_hash) DO NOTHING
            """, (str(uuid.uuid4()), text, source_filename, page_number, chunk_hash, embedding))
            conn.commit()
    finally:
        conn.close()


def search_similar_chunks(query_embedding, source_filename=None, limit=5, similarity_threshold=0.35):
    """
    Busca los chunks más similares en la BD.
    Si se pasa source_filename, filtra sólo por ese archivo.
    Retorna lista de tuplas: (text_chunk, source_filename, page_number, similarity)
    """
    conn = connect_db()
    if not conn:
        print("❌ No se pudo conectar a la BD para buscar chunks")
        return []

    try:
        with conn.cursor() as cur:
            if source_filename:
                print(f"🔍 Buscando chunks para archivo: {source_filename}")
                cur.execute("""
                    SELECT text_chunk, source_filename, page_number,
                    1 - (embedding <=> %s::vector) as similarity
                    FROM document_embeddings
                    WHERE source_filename ILIKE %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (query_embedding, f"%{source_filename}%", query_embedding, limit))
            else:
                print("🔍 Buscando chunks en todos los archivos")
                cur.execute("""
                    SELECT text_chunk, source_filename, page_number,
                    1 - (embedding <=> %s::vector) as similarity
                    FROM document_embeddings
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (query_embedding, query_embedding, limit))

            rows = cur.fetchall()
            print(f"✅ Se encontraron {len(rows)} chunks en BD")

            # Filtramos por similarity
            filtered = [(t, s, p, sim) for t, s, p, sim in rows if sim >= similarity_threshold]
            print(f"✅ {len(filtered)} chunks pasaron el threshold ({similarity_threshold})")

            # Opcional: imprimir los primeros resultados para depuración
            for t, s, p, sim in filtered:
                print(f"Chunk: {t[:50]}..., File: {s}, Page: {p}, Sim: {sim:.3f}")

            return filtered

    except Exception as e:
        print(f"❌ Error búsqueda: {e}")
        return []

    finally:
        conn.close()



# ===============================
# PROCESAMIENTO DE DOCUMENTOS
# ===============================
def document_exists(filename):
    """Verifica si un documento ya fue cargado en la base."""
    conn = connect_db()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM document_embeddings WHERE source_filename = %s", (filename,))
            count = cur.fetchone()[0]
            return count > 0
    finally:
        conn.close()


def delete_document_embeddings(filename):
    """Elimina los embeddings de un documento específico."""
    conn = connect_db()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM document_embeddings WHERE source_filename = %s", (filename,))
            conn.commit()
    finally:
        conn.close()
    print(f"🧹 Embeddings previos de '{filename}' eliminados correctamente.")


def process_pdf_and_save_chunks(filepath, chunk_size=800):
    """Divide un PDF en fragmentos y guarda sus embeddings."""
    filename = os.path.basename(filepath)
    try:
        reader = pypdf.PdfReader(open(filepath, "rb"))
    except Exception as e:
        print(f"❌ Error leyendo PDF: {e}")
        return

    chunks = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = re.sub(r'\s+', ' ', text.strip())
        if not text:
            continue
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            chunks.append((chunk, page_num + 1))

    print(f"📄 {len(chunks)} fragmentos listos de {filename}")
    embeddings = get_embeddings([c for c, _ in chunks])

    for i, (chunk, page) in enumerate(chunks):
        if embeddings[i] is not None:
            save_chunk(chunk, filename, page, str(hash(chunk))[:10], embeddings[i])

    print(f"✅ PDF '{filename}' indexado correctamente.\n")


def process_csv_and_save_chunks(filepath, batch_size=50):
    """Lee un CSV, genera embeddings por filas y los guarda en la base."""
    filename = os.path.basename(filepath)
    try:
        df = pd.read_csv(filepath, sep=None, engine="python", encoding="latin1")
    except Exception as e:
        print(f"❌ Error leyendo CSV: {e}")
        return

    print(f"📊 CSV '{filename}' con {df.shape[0]} filas y {df.shape[1]} columnas")
    chunks = []

    for i, row in df.iterrows():
        text = ", ".join([f"{c}: {v}" for c, v in zip(df.columns, row.values)])
        chunks.append(text)

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        embs = get_embeddings(batch)
        for j, txt in enumerate(batch):
            if embs[j] is not None:
                save_chunk(txt, filename, i + j + 1, str(hash(txt))[:10], embs[j])

    print(f"✅ CSV '{filename}' procesado e indexado.\n")


def handle_document_load(filepath, filetype):
    """
    Procesa un documento PDF o CSV para embeddings.
    En la versión web no se pregunta al usuario, simplemente:
    - Si ya existe, lo elimina y reprocesa automáticamente.
    - Si no existe, lo procesa normalmente.
    """
    filename = os.path.basename(filepath)
    already_processed = document_exists(filename)

    if already_processed:
        print(f"🔁 Reprocesando {filetype.upper()} '{filename}' para esta sesión...")
        delete_document_embeddings(filename)

    print(f"📄 Procesando {filetype.upper()} '{filename}'...")

    if filetype.lower() == "pdf":
        process_pdf_and_save_chunks(filepath)
    elif filetype.lower() == "csv":
        process_csv_and_save_chunks(filepath)
    else:
        print("⚠️ Tipo de archivo desconocido.")
