# core/scripts/generate_semantic_schema.py
"""
Genera un archivo JSON con el esquema semántico enriquecido de la base de datos PostgreSQL.
Incluye una estructura de keywords y descripción para cada tabla y columna.
"""

import os
import json
import sys
from psycopg2.extras import RealDictCursor
from typing import Dict, Any, List

# Agregamos el path raíz del proyecto para poder importar db.connection
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(BASE_DIR)

from db.connection import connect_db  # Usamos la conexión existente


# ============================================================
# FUNCIÓN: EXTRAER EL ESQUEMA DE LA BASE DE DATOS POSTGRESQL
# ============================================================

def get_database_schema():
    """Extrae estructura de tablas, columnas, claves primarias y foráneas."""
    conn = connect_db()
    if conn is None:
        raise ConnectionError("❌ No se pudo conectar a la base de datos.")

    cursor = conn.cursor(cursor_factory=RealDictCursor)
    schema = {}

    # Obtener todas las tablas del esquema público
    cursor.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        AND table_name NOT LIKE 'pg_%' AND table_name NOT LIKE 'sql_%'
        ORDER BY table_name;
    """)
    tables = [r["table_name"] for r in cursor.fetchall()]

    for table in tables:
        # Columnas
        cursor.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position;
        """, (table,))
        
        # Mapear a formato semántico de columna
        columns: Dict[str, Dict[str, str | List[str]]] = {}
        for r in cursor.fetchall():
            col_name = r["column_name"]
            data_type = r["data_type"]
            
            # --- Enriquecimiento automático de columnas ---
            keywords = [col_name]
            description = f"Columna de tipo {data_type}."
            
            if "nombre" in col_name:
                keywords.extend(["nombre", "título", "identificador"])
                description = "Contiene el nombre o título de la entidad."
            elif "valor_respuesta" in col_name or "puntaje" in col_name:
                keywords.extend(["puntaje", "score", "valor", "métrica", "resultado"])
                description = "El valor numérico o puntaje principal."
            elif "id" in col_name:
                keywords.append("id")
            
            columns[col_name] = {
                "type": data_type,
                "description": description,
                "keywords": keywords
            }
        
        # Claves primarias (Guardadas para referencia, no semántica)
        cursor.execute("""
            SELECT kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu 
            ON tc.constraint_name = kcu.constraint_name
            WHERE tc.constraint_type = 'PRIMARY KEY' AND tc.table_name = %s;
        """, (table,))
        pk = [r["column_name"] for r in cursor.fetchall()]

        # 4. Claves foráneas (Guardadas para referencia, no semántica)
        cursor.execute("""
            SELECT
                kcu.column_name,
                ccu.table_name AS foreign_table,
                ccu.column_name AS foreign_column
            FROM
                information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                  ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage AS ccu
                  ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = %s;
        """, (table,))
        fk_data = cursor.fetchall()
        fks = {
            r["column_name"]: {
                "references": f"{r['foreign_table']}.{r['foreign_column']}"
            }
            for r in fk_data
        }

        schema[table] = {
            "columns": columns,
            "primary_key": pk,
            "foreign_keys": fks
        }

    cursor.close()
    conn.close()
    return schema


# ============================================================
# FUNCIÓN: ENRIQUECER EL ESQUEMA CON SEMÁNTICA (Keywords de alto nivel)
# ============================================================

def enrich_semantic_info(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Agrega descripciones y keywords de alto nivel a las tablas clave."""
    # Lista de sinónimos para tablas clave que el Agente usará para mapeo.
    table_enrichment = {
        "usuario": {
            "description": "Contiene los datos personales, credenciales y rol de los colaboradores o empleados.",
            "keywords": ["colaborador", "empleado", "persona", "trabajador", "user", "usuarios"]
        },
        "evaluacion": {
            "description": "Guarda las evaluaciones de desempeño o retroalimentaciones hechas a los usuarios.",
            "keywords": ["desempeño", "valoración", "feedback", "revisión", "evaluacion", "evaluaciones"]
        },
        "equipo": {
            "description": "Grupos, departamentos o áreas de la empresa.",
            "keywords": ["equipo", "departamento", "área", "grupo", "unidad", "facultad"]
        },
        "respuesta": {
            "description": "Respuestas seleccionadas o escritas por los usuarios.",
            "keywords": ["respuesta", "contestación", "opción", "resultado"]
        }
        # Se puede añadir más tablas aquí si se necesitan
    }

    for table, data in table_enrichment.items():
        if table in schema:
            # Añadir descripción y keywords de alto nivel
            schema[table]["description"] = data["description"]
            # Usar la clave 'keywords' en lugar de 'synonyms' para uniformidad con las columnas
            schema[table]["keywords"] = data["keywords"]
            
            # Asegurar que 'keywords' exista en la tabla para la lógica del planificador
            if 'keywords' not in schema[table]:
                 schema[table]['keywords'] = []

    return schema


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def main():
    print("📘 Extrayendo esquema desde PostgreSQL...")
    schema = get_database_schema()

    print("✨ Enriqueciendo con información semántica...")
    enriched_schema = enrich_semantic_info(schema)

    # PASO CRÍTICO: Envolver el esquema enriquecido en la clave 'tables'
    final_output = {
        "tables": enriched_schema
    }

    # Guardar JSON
    output_dir = os.path.join(BASE_DIR, "core", "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "semantic_schema.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"✅ Archivo generado correctamente en: {output_path}")

# ============================================================
# EJECUCIÓN DEL SCRIPT
# ============================================================
if __name__ == "__main__":
    try:
        main()
    except ConnectionError as e:
        print(f"Error al iniciar el generador de esquema: {e}")