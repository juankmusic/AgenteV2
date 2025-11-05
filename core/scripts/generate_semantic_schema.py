# core/scripts/generate_semantic_schema.py
"""
Genera un archivo JSON con el esquema semántico enriquecido de la base de datos PostgreSQL.
Usa la conexión definida en db/connection.py y las variables del archivo .env.
"""

import os
import json
import sys
from psycopg2.extras import RealDictCursor

# Agregamos el path raíz del proyecto para poder importar db.connection
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(BASE_DIR)

from db.connection import connect_db  # ✅ Usamos tu conexión existente


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

    # 1️⃣ Obtener todas las tablas del esquema público
    cursor.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """)
    tables = [r["table_name"] for r in cursor.fetchall()]

    for table in tables:
        # 2️⃣ Columnas
        cursor.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position;
        """, (table,))
        columns = {r["column_name"]: r["data_type"] for r in cursor.fetchall()}

        # 3️⃣ Claves primarias
        cursor.execute("""
            SELECT kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu 
            ON tc.constraint_name = kcu.constraint_name
            WHERE tc.constraint_type = 'PRIMARY KEY' AND tc.table_name = %s;
        """, (table,))
        pk = [r["column_name"] for r in cursor.fetchall()]

        # 4️⃣ Claves foráneas
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
# FUNCIÓN: ENRIQUECER EL ESQUEMA CON SEMÁNTICA
# ============================================================

def enrich_semantic_info(schema):
    """Agrega descripciones y sinónimos a las tablas clave."""
    enrichment = {
        "usuario": {
            "description": "Contiene los datos personales, credenciales y rol de los colaboradores o empleados.",
            "synonyms": ["colaborador", "empleado", "persona", "trabajador", "user"]
        },
        "cursos": {
            "description": "Registra los cursos de formación disponibles en la plataforma, con sus fechas y categorías.",
            "synonyms": ["curso", "formación", "capacitación", "entrenamiento", "learning"]
        },
        "evaluacion": {
            "description": "Guarda las evaluaciones de desempeño o retroalimentaciones hechas a los usuarios.",
            "synonyms": ["desempeño", "valoración", "feedback", "revisión"]
        },
        "progreso_usuario": {
            "description": "Registra el avance y progreso de cada usuario en los cursos asignados.",
            "synonyms": ["progreso", "avance", "seguimiento", "tracking"]
        },
        "proyecto": {
            "description": "Información de proyectos o iniciativas desarrolladas por los usuarios.",
            "synonyms": ["plan", "actividad", "iniciativa"]
        },
        "indicador": {
            "description": "Contiene métricas o KPIs asociados a desempeño o proyectos.",
            "synonyms": ["kpi", "métrica", "indicador de desempeño"]
        },
        "pregunta": {
            "description": "Preguntas que forman parte de encuestas, evaluaciones o formularios.",
            "synonyms": ["ítem", "reactivo", "pregunta de encuesta"]
        },
        "respuesta": {
            "description": "Respuestas seleccionadas o escritas por los usuarios.",
            "synonyms": ["contestación", "opción", "resultado"]
        },
        "categoria": {
            "description": "Define categorías o grupos para clasificar cursos, recursos u otros elementos.",
            "synonyms": ["grupo", "tipo", "clase"]
        }
    }

    for table, data in enrichment.items():
        if table in schema:
            schema[table]["description"] = data["description"]
            schema[table]["synonyms"] = data["synonyms"]

    return schema


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def main():
    print("📘 Extrayendo esquema desde PostgreSQL...")
    schema = get_database_schema()

    print("✨ Enriqueciendo con información semántica...")
    enriched_schema = enrich_semantic_info(schema)

    # Guardar JSON
    output_dir = os.path.join(BASE_DIR, "core", "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "semantic_schema.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched_schema, f, indent=2, ensure_ascii=False)

    print(f"✅ Archivo generado correctamente en: {output_path}")


if __name__ == "__main__":
    main()