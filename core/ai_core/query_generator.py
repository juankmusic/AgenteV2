import os
import re
import json
from dotenv import load_dotenv
from openai import OpenAI as OpenAIClient
from db.connection import connect_db

# ======================================================
# CONFIGURACIÓN INICIAL
# ======================================================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAIClient(api_key=OPENAI_API_KEY)

# ======================================================
# 1️⃣ FUNCIÓN: OBTENER ESTRUCTURA DE LA BASE DE DATOS
# ======================================================
def get_database_schema():
    """
    Recupera todas las tablas y columnas del esquema 'public' en PostgreSQL.
    Excluye vistas y tablas del sistema (pg_*, sql_*).
    Devuelve un diccionario estructurado {tabla: [{columna, tipo}, ...]}.
    """
    schema = {}
    conn = connect_db()
    if not conn:
        print("❌ No se pudo conectar a la base de datos.")
        return schema

    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public'
                AND table_name NOT LIKE 'pg_%'
                AND table_name NOT LIKE 'sql_%'
                ORDER BY table_name, ordinal_position;
            """)
            rows = cur.fetchall()

            for table, col, dtype in rows:
                schema.setdefault(table, []).append({
                    "columna": col,
                    "tipo": dtype
                })

        print(f"📚 Esquema detectado: {len(schema)} tablas encontradas.")
        return schema

    except Exception as e:
        print(f"❌ Error obteniendo esquema: {e}")
        return {}
    finally:
        conn.close()

# ======================================================
# 2️⃣ FUNCIÓN: VALIDAR CONSULTAS SQL
# ======================================================
def validate_sql(sql: str) -> bool:
    """
    Analiza la consulta SQL generada y bloquea comandos destructivos o no seguros.
    Rechaza DROP, TRUNCATE, ALTER, DELETE o UPDATE sin cláusula WHERE.
    """
    forbidden = r"\b(DROP|TRUNCATE|ALTER|DELETE(?!.*WHERE)|UPDATE(?!.*WHERE))\b"
    if re.search(forbidden, sql, re.IGNORECASE):
        print("⚠️ SQL potencialmente peligroso detectado y rechazado.")
        return False
    return True

# ======================================================
# 3️⃣ FUNCIÓN PRINCIPAL: GENERAR CONSULTA SQL
# ======================================================
def generate_sql_with_openai(plan: dict, schema: dict) -> dict:
    """
    Usa un modelo de lenguaje para generar una consulta SQL válida y segura
    basada en el plan de acción del agente y el esquema real de la base.
    """
    plan_json = json.dumps(plan, ensure_ascii=False, indent=2)
    schema_json = json.dumps(schema, ensure_ascii=False, indent=2)

    prompt = f"""
    Eres un generador experto de SQL para PostgreSQL 15 (compatible con pgvector).
    Tienes acceso a una base de datos con las siguientes tablas y columnas:

    {schema_json}

    Tu tarea es generar una consulta SQL válida y segura
    basada en el siguiente plan de acción del agente inteligente:

    {plan_json}

    Reglas:
    - No uses DROP, ALTER ni TRUNCATE.
    - Usa SELECT, INSERT o UPDATE según corresponda.
    - Si el plan es de tipo 'evaluar' o 'guardar_resultado', genera un INSERT.
    - Si es de tipo 'consultar_datos' o 'generar_informe', genera un SELECT.
    - Si el plan menciona embeddings, analiza o busca similitud, usa la tabla 'document_embeddings' con su columna 'embedding'.
    - Devuelve SOLO un JSON válido con este formato exacto:
    {{
        "sql": "SELECT ...",
        "descripcion": "Explicación natural de lo que hará la consulta"
    }}
    """

    try:
        res = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un generador SQL experto en PostgreSQL y pgvector. Devuelve solo JSON válido."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        raw_output = res.choices[0].message.content.strip()
        clean_output = (
            raw_output.replace("```json", "")
                      .replace("```", "")
                      .strip()
        )

        # Intentar parsear el JSON
        result = json.loads(clean_output)
        sql = result.get("sql", "").strip()

        # Validación de seguridad
        if not validate_sql(sql):
            return {"sql": None, "descripcion": "Consulta rechazada por motivos de seguridad."}

        print(f"✅ SQL generado:\n{sql}")
        return result

    except json.JSONDecodeError:
        print(f"❌ Error: la respuesta del modelo no es JSON válido.\nSalida cruda:\n{raw_output}")
        return {"sql": None, "descripcion": "Respuesta no JSON del modelo."}
    except Exception as e:
        print(f"❌ Error generando SQL: {e}")
        return {"sql": None, "descripcion": str(e)}

# ======================================================
# 4️⃣ FUNCIÓN: INTERFAZ PÚBLICA
# ======================================================
def generate_query_from_plan(plan: dict) -> dict:
    """
    Punto de entrada principal.
    Lee la estructura de la base de datos y genera una consulta SQL
    a partir del plan de acción del agente inteligente.
    """
    schema = get_database_schema()
    if not schema:
        print("⚠️ No se pudo obtener el esquema de la base de datos.")
        return {"sql": None, "descripcion": "Error al obtener el esquema."}

    result = generate_sql_with_openai(plan, schema)

    if result.get("sql"):
        print(f"📜 Consulta generada exitosamente.")
    else:
        print(f"⚠️ No se pudo generar la consulta SQL.")

    return result
