# core/ai_core/schema_loader.py
import json
import os

def load_schema():
    """
    Carga los esquemas semántico y de embeddings desde core/data.
    Si el archivo de embeddings no existe, devuelve un diccionario vacío.
    """
    base_path = os.path.join(os.path.dirname(__file__), "..", "data")  # apunta a core/data
    semantic_path = os.path.join(base_path, "semantic_schema.json")
    embedding_path = os.path.join(base_path, "embedding_schema.json")

    # Cargar esquema semántico
    if not os.path.exists(semantic_path):
        raise FileNotFoundError(f"❌ No se encontró semantic_schema.json en {semantic_path}")
    with open(semantic_path, "r", encoding="utf-8") as f:
        semantic_schema = json.load(f)

    # Cargar embeddings si existen, si no, devolver vacío
    if os.path.exists(embedding_path):
        with open(embedding_path, "r", encoding="utf-8") as f:
            embedding_schema = json.load(f)
    else:
        print(f"⚠️ embedding_schema.json no encontrado en {embedding_path}, usando diccionario vacío")
        embedding_schema = {}

    return semantic_schema, embedding_schema