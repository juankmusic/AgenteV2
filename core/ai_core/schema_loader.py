import json
import os
import logging
from typing import Dict, Any

# 👇 ==================================================
# 👇 ¡EL BUG ESTABA AQUÍ!
# 👇 Cambiamos la importación absoluta por una relativa (con un .)
# 👇 ==================================================
try:
    # Importar desde el mismo directorio (core/ai_core)
    from .nlp_embeddings import generate_embedding
except ImportError:
    # Fallback por si la estructura cambia
    try:
        from core.ai_core.nlp_embeddings import generate_embedding
    except ImportError as e:
        logging.critical(f"FALLO CRÍTICO: No se puede importar generate_embedding. {e}")
        # Si esto falla, el módulo no es utilizable
        raise ImportError("No se pudo importar nlp_embeddings. El cargador de esquema no puede funcionar.") from e
# 👆 ==================================================
# 👆 FIN DE LA CORRECCIÓN
# 👆 ==================================================


logger = logging.getLogger(__name__)

# --- NUEVA FUNCIÓN (Sin cambios, pero ahora la importación funciona) ---
def _generate_and_cache_embeddings(semantic_schema: Dict[str, Any], embedding_path: str) -> Dict[str, Any]:
    """
    Genera embeddings para cada tabla y columna en el semantic_schema
    y los guarda en el embedding_path como un JSON.
    """
    logger.info("Generando nuevos embeddings de esquema... Esto puede tardar un momento.")
    
    embedding_schema = {}
    
    # (La importación ya se hizo arriba, así que 'generate_embedding' existe aquí)

    all_tables_schema = semantic_schema.get("tables", {})
    if not all_tables_schema:
        logger.error("El 'semantic_schema.json' no contiene la clave 'tables' o está vacía. No se pueden generar embeddings.")
        return {}

    # 1. Generar embeddings para Tablas
    for table_name, table_data in all_tables_schema.items():
        try:
            keywords = ", ".join(table_data.get("keywords", []))
            desc = table_data.get("description", "")
            text_to_embed = f"Tabla: {table_name}. Descripción: {desc}. Palabras clave: {keywords}."
            
            embedding = generate_embedding(text_to_embed) # <--- Esta llamada ahora SÍ funcionará
            
            if embedding:
                embedding_schema[table_name] = {
                    "text": text_to_embed,
                    "embedding": embedding,
                    "type": "table"
                }
            else:
                 logger.warning(f"No se pudo generar embedding para la tabla {table_name} (texto: {text_to_embed})")
        except Exception as e:
            logger.error(f"Error generando embedding para tabla {table_name}: {e}")

    # 2. Generar embeddings para Columnas
    for table_name, table_data in all_tables_schema.items():
        columns = table_data.get("columns", {})
        if not columns:
            continue
            
        for col_name, col_data in columns.items():
            try:
                keywords = ", ".join(col_data.get("keywords", []))
                desc = col_data.get("description", "")
                text_to_embed = f"Columna: {col_name} en la tabla {table_name}. Descripción: {desc}. Palabras clave: {keywords}."
                
                embedding = generate_embedding(text_to_embed) # <--- Esta llamada ahora SÍ funcionará
                
                if embedding:
                    key = f"{table_name}.{col_name}"
                    embedding_schema[key] = {
                        "text": text_to_embed,
                        "embedding": embedding,
                        "type": "column"
                    }
                else:
                    logger.warning(f"No se pudo generar embedding para la columna {col_name} (texto: {text_to_embed})")
            except Exception as e:
                logger.error(f"Error generando embedding para columna {col_name}: {e}")

    # 3. Guardar el archivo en caché
    try:
        if not embedding_schema:
            logger.error("El schema de embeddings generado está vacío. No se guardará en caché.")
            return {}
            
        with open(embedding_path, "w", encoding="utf-8") as f:
            json.dump(embedding_schema, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Embeddings de esquema guardados exitosamente en {embedding_path}")
    except Exception as e:
        logger.error(f"❌ No se pudo guardar el caché de embeddings en {embedding_path}: {e}")

    return embedding_schema

# --- FUNCIÓN PRINCIPAL (Sin cambios) ---
def load_schema():
    """
    Carga el esquema semántico.
    Carga el esquema de embeddings desde el caché (embedding_schema.json).
    Si el caché no existe, lo genera basado en el esquema semántico.
    """
    base_path = os.path.join(os.path.dirname(__file__), "..", "data") 
    semantic_path = os.path.join(base_path, "semantic_schema.json")
    embedding_path = os.path.join(base_path, "embedding_schema.json") 

    # 1. Cargar esquema semántico (Obligatorio)
    if not os.path.exists(semantic_path):
        raise FileNotFoundError(f"❌ No se encontró semantic_schema.json en {semantic_path}")
    
    with open(semantic_path, "r", encoding="utf-8") as f:
        semantic_schema = json.load(f)

    # 2. Cargar o Generar esquema de embeddings
    if os.path.exists(embedding_path):
        try:
            with open(embedding_path, "r", encoding="utf-8") as f:
                embedding_schema = json.load(f)
            
            # 👇 MEJORA: Verificar si el archivo cargado está vacío
            if not embedding_schema:
                logger.warning("⚠️ El caché de embeddings 'embedding_schema.json' estaba vacío. Regenerando...")
                embedding_schema = _generate_and_cache_embeddings(semantic_schema, embedding_path)
            else:
                logger.info("✅ Embeddings de esquema cargados desde el caché.")
                
        except Exception as e:
            logger.warning(f"⚠️ No se pudo leer embedding_schema.json (está corrupto?). Generando de nuevo... Error: {e}")
            embedding_schema = _generate_and_cache_embeddings(semantic_schema, embedding_path)
    else:
        logger.warning(f"⚠️ embedding_schema.json no encontrado. Generando por primera vez...")
        embedding_schema = _generate_and_cache_embeddings(semantic_schema, embedding_path)

    # 3. Validar si la generación falló y devuelve vacío
    if not embedding_schema:
        logger.error("❌ El esquema de embeddings está vacío. El planificador semántico no funcionará.")
        embedding_schema = {}

    return semantic_schema, embedding_schema