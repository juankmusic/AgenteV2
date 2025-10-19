# core/ai_core/query_generator.py

import os
import json
import psycopg2
from dotenv import load_dotenv
from openai import OpenAI as OpenAIClient
from core.ai_core.dynamic_planner import plan_actions
from db.connection import connect_db

# ======================================================
# CONFIGURACIÓN
# ======================================================
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
deepseek_client = OpenAIClient(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")

# ======================================================
# 1️⃣ FUNCIÓN: OBTENER ESTRUCTURA DE LA BASE DE DATOS
# ======================================================
def get_database_schema():
    """
    Recupera las tablas y columnas de la base de datos.
    Devuelve un diccionario con la estructura.
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
                ORDER BY table_name, ordinal_position;
            """)
            rows = cur.fetchall()
            for table, col, dtype in rows:
                schema.setdefault(table, []).append({"columna": col, "tipo": dtype})
        return schema
    finally:
        conn.close()

# ======================================================
# 2️⃣ FUNCIÓN: VALIDAR CONSULTA SQL
# ======================================================
def validate_sql(sql: str) -> bool:
    """
    Rechaza SQL peligrosos (DROP, DELETE sin WHERE, etc.)
    """
    sql_upper = sql.upper()
    forbidden = ["DROP", "TRUNCATE", "ALTER", "DELETE FROM", "UPDATE"]
    if any(cmd in sql_upper for cmd in forbidden):
        print("⚠️ SQL potencialmente peligroso detectado.")
        return False
    return True

# ======================================================
# 3️⃣ FUNCIÓN PRINCIPAL: GENERAR CONSULTA CON DEEPSEEK
# ======================================================
def generate_sql_with_deepseek(plan: dict, schema: dict) -> dict:
    """
    Usa el modelo de lenguaje para generar una consulta SQL válida
    según la estructura de la base de datos y el plan del agente.
    """
    plan_json = json.dumps(plan, ensure_ascii=False, indent=2)
    schema_json = json.dumps(schema, ensure_ascii=False, indent=2)

    prompt = f"""
    Eres un generador experto de consultas SQL para PostgreSQL.
    Tu tarea es crear una consulta SQL **válida y segura**
    basada en el siguiente plan de acción y la estructura de base de datos.

    ⚙️ Plan del agente:
    {plan_json}

    🧩 Estructura de la base de datos:
    {schema_json}

    Reglas:
    - No modifiques la estructura de la base de datos.
    - No uses comandos DROP, TRUNCATE, ALTER o DELETE sin WHERE.
    - Si el plan es de tipo 'evaluar_colaborador' o 'guardar_datos', genera un INSERT.
    - Si es de tipo 'consultar_datos' o 'generar_reporte', genera un SELECT.
    - Devuelve SOLO un JSON válido con el siguiente formato:

    {{
        "sql": "<consulta generada>",
        "descripcion": "<explicación natural de lo que hará la consulta>"
    }}
    """

    try:
        res = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Eres un generador SQL experto. Devuelve siempre JSON válido."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )

        raw_output = res.choices[0].message.content.strip()

        clean_output = (
            raw_output.replace("```json", "")
                      .replace("```", "")
                      .replace("\\n", "")
                      .replace("\n", "")
                      .strip()
        )

        result = json.loads(clean_output)
        sql = result.get("sql", "").strip()

        if not validate_sql(sql):
            return {
                "sql": None,
                "descripcion": "Consulta rechazada por motivos de seguridad."
            }

        return result

    except Exception as e:
        print(f"❌ Error generando SQL: {e}")
        return {"sql": None, "descripcion": str(e)}

# ======================================================
# 4️⃣ FUNCIÓN: INTERFAZ PRINCIPAL
# ======================================================
def generate_query_from_plan(plan: dict) -> dict:
    """
    Combina lectura del esquema + generación SQL automática.
    """
    schema = get_database_schema()
    if not schema:
        return {"sql": None, "descripcion": "No se pudo obtener el esquema de la base de datos."}

    result = generate_sql_with_deepseek(plan, schema)
    return result

# ======================================================
# 5️⃣ PRUEBA LOCAL
# ======================================================
if __name__ == "__main__":
    # Plan simulado desde el Dynamic Planner
    example_plan = {
        "accion": "evaluar_colaborador",
        "parametros": {
            "colaborador": "María",
            "virtudes": ["colaborativa", "resiliente"],
            "areas_mejora": ["liderazgo"]
        }
    }

    sql_result = generate_query_from_plan(example_plan)
    print("\n🧠 SQL generado por el agente:")
    print(json.dumps(sql_result, indent=2, ensure_ascii=False))
