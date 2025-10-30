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
    Recupera todas las tablas y columnas del esquema 'public' en PostgreSQL,
    junto con las claves foráneas para que el modelo pueda generar consultas más completas.
    """
    schema = {}
    foreign_keys = []  # Asegúrate de definir esta variable fuera de la consulta

    conn = connect_db()
    if not conn:
        print("❌ No se pudo conectar a la base de datos.")
        return schema, foreign_keys  # Cambié para devolver las claves foráneas también

    try:
        with conn.cursor() as cur:
            # Obtener tablas y columnas
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
            
            # Obtener relaciones entre tablas (clave foránea)
            cur.execute("""
                SELECT
                    tc.table_name, kcu.column_name, ccu.table_name AS foreign_table_name,
                    ccu.column_name AS foreign_column_name
                FROM 
                    information_schema.table_constraints AS tc
                JOIN 
                    information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                JOIN 
                    information_schema.constraint_column_usage AS ccu
                    ON ccu.constraint_name = tc.constraint_name
                WHERE 
                    tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = 'public';
            """)
            foreign_keys = cur.fetchall()

        print(f"📚 Esquema con relaciones detectado: {len(schema)} tablas encontradas.")
        return schema, foreign_keys  # Devuelve las claves foráneas también

    except Exception as e:
        print(f"❌ Error obteniendo esquema: {e}")
        return {}, []  # Devuelve listas vacías en caso de error
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
def generate_sql_with_openai(plan: dict, schema: dict, foreign_keys: list) -> dict:
    """
    Usa un modelo de lenguaje para generar una consulta SQL válida y segura
    basada en el plan de acción del agente y el esquema real de la base.
    """
    plan_json = json.dumps(plan, ensure_ascii=False, indent=2)
    schema_json = json.dumps(schema, ensure_ascii=False, indent=2)
    foreign_keys_json = json.dumps(foreign_keys, ensure_ascii=False, indent=2)

    prompt = f"""
    Eres un generador experto de SQL para PostgreSQL. Tu tarea es generar consultas SQL válidas y seguras basadas en los siguientes datos:

    1️⃣ **Plan de acción del agente**:
    {plan_json}

    2️⃣ **Esquema de la base de datos**:
    {schema_json}

    3️⃣ **Relaciones importantes**: 
    A continuación se listan las claves foráneas que puedes utilizar para realizar los `JOIN` entre tablas:
    {foreign_keys_json}

    4️⃣ **Instrucciones**:
    - Debes utilizar las claves foráneas para realizar `JOIN` entre las tablas relacionadas cuando sea necesario.
    - Si el plan menciona una tabla o columna, interpreta el contexto semántico para saber qué tabla y qué columna utilizar.
    - Asegúrate de que la consulta sea **segura** y que no contenga comandos destructivos como `DROP`, `DELETE` sin `WHERE`, o `TRUNCATE`.
    - Genera una consulta que **respete las relaciones y restricciones de la base de datos**.
    - **NO agregues filtros automáticos a menos que el plan indique explícitamente un valor para el filtro.**
    - Usa únicamente los nombres de tablas que existen en el esquema proporcionado.
    - No inventes nombres de tablas aunque existan sinónimos en el plan o el texto.

    Recuerda: nunca pidas datos al usuario, la información que necesitas está en la base de datos.

    **Salida esperada**:
    Genera SOLO un JSON válido con la estructura:
    {{
        "sql": "SELECT ...",
        "descripcion": "Breve descripción de lo que hace la consulta generada."
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
    schema, foreign_keys = get_database_schema()  # Ahora obtiene también las claves foráneas
    if not schema:
        print("⚠️ No se pudo obtener el esquema de la base de datos.")
        return {"sql": None, "descripcion": "Error al obtener el esquema."}

    result = generate_sql_with_openai(plan, schema, foreign_keys)  # Pasa las claves foráneas también

    if result.get("sql"):
        print(f"📜 Consulta generada exitosamente.")
    else:
        print(f"⚠️ No se pudo generar la consulta SQL.")

    return result

