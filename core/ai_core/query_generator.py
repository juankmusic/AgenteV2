# core/ai_core/query_generator.py
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
    junto con las claves foráneas y mapeo de columnas para validación.
    """
    schema = {}
    columns_by_table = {}  # NUEVO: columnas por tabla para validación
    foreign_keys = []  # Asegúrate de definir esta variable fuera de la consulta

    conn = connect_db()
    if not conn:
        print("❌ No se pudo conectar a la base de datos.")
        return schema, columns_by_table, foreign_keys

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
                columns_by_table.setdefault(table, []).append(col)
            
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
        return schema, columns_by_table, foreign_keys

    except Exception as e:
        print(f"❌ Error obteniendo esquema: {e}")
        return {}, {}, []
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
# 3️⃣ FUNCIÓN: VALIDAR COLUMNAS EXISTENTES
# ======================================================
def validate_sql_columns(sql: str, columns_by_table: dict) -> bool:
    """
    Revisa que todas las columnas mencionadas en la consulta existan en su tabla correspondiente.
    """
    for table, cols in columns_by_table.items():
        # Buscar columnas con patrón table.col
        pattern = re.compile(rf"{table}\.(\w+)", re.IGNORECASE)
        for col in pattern.findall(sql):
            if col not in cols:
                print(f"❌ Columna inválida detectada: {table}.{col}")
                return False
    return True

# ======================================================
# 4️⃣ FUNCIÓN PRINCIPAL: GENERAR CONSULTA SQL
# ======================================================
def generate_sql_with_openai(plan: dict, schema: dict, foreign_keys: list, columns_by_table: dict) -> dict:
    """
    Usa un modelo de lenguaje para generar una consulta SQL válida y segura
    basada en el plan de acción del agente y el esquema real de la base.
    """
    plan_json = json.dumps(plan, ensure_ascii=False, indent=2)

    # 🔹 FILTRO: Excluir tablas que no deben usarse
    TABLES_PROHIBIDAS = ["respuesta"]
    filtered_schema = {k: v for k, v in schema.items() if k not in TABLES_PROHIBIDAS}

    schema_json = json.dumps(filtered_schema, ensure_ascii=False, indent=2)
    foreign_keys_json = json.dumps(foreign_keys, ensure_ascii=False, indent=2)
    columns_json = json.dumps(columns_by_table, ensure_ascii=False, indent=2)

    prompt = f"""
    Eres un generador experto de SQL para PostgreSQL. Tu tarea es generar consultas SQL válidas y seguras basadas en los siguientes datos:

    1️⃣ **Plan de acción del agente**:
    {plan_json}

    2️⃣ **Esquema de la base de datos**:
    {schema_json}

    3️⃣ **Relaciones importantes**: 
    A continuación se listan las claves foráneas que puedes utilizar para realizar los `JOIN` entre tablas:
    {foreign_keys_json}
    
    - Estas son las columnas disponibles por tabla:
    {columns_json}
    - Solo puedes usar estas columnas. No inventes nombres.

    4️⃣ **Instrucciones**:
    - Debes utilizar las claves foráneas para realizar `JOIN` entre las tablas relacionadas cuando sea necesario.
    - Si el plan menciona una tabla o columna, interpreta el contexto semántico para saber qué tabla y qué columna utilizar.
    - Asegúrate de que la consulta sea **segura** y que no contenga comandos destructivos como `DROP`, `DELETE` sin `WHERE`, o `TRUNCATE`.
    - Genera una consulta que **respete las relaciones y restricciones de la base de datos**.
    - Usa solo columnas que existen en el esquema provisto.
    - Si necesitas un valor de otra tabla, realiza un JOIN usando las claves foráneas.
    - No inventes nombres de columnas o tablas.
    - Asegúrate de que la consulta sea segura y respete las relaciones y restricciones de la base de datos.
    - La columna valor_respuesta NUNCA está en las tablas de respuestas. Está en la tabla iluo. Si quieres calcular promedios o usar ese valor, debes hacer JOIN con iluo usando id_iluo.

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

        result = json.loads(clean_output)
        sql = result.get("sql", "").strip()

        # Validación de seguridad
        if not validate_sql(sql):
            return {"sql": None, "descripcion": "Consulta rechazada por motivos de seguridad."}

        # NUEVA VALIDACIÓN: columnas existentes
        if not validate_sql_columns(sql, columns_by_table):
            return {"sql": None, "descripcion": "Consulta usa columnas inexistentes."}

        print(f"✅ SQL generado:\n{sql}")
        return result

    except json.JSONDecodeError:
        print(f"❌ Error: la respuesta del modelo no es JSON válido.\nSalida cruda:\n{raw_output}")
        return {"sql": None, "descripcion": "Respuesta no JSON del modelo."}
    except Exception as e:
        print(f"❌ Error generando SQL: {e}")
        return {"sql": None, "descripcion": str(e)}

# ======================================================
# 5️⃣ FUNCIÓN: INTERFAZ PÚBLICA
# ======================================================
def generate_query_from_plan(plan: dict) -> dict:
    """
    Punto de entrada principal.
    Verifica si la acción es 'reutilizar_consulta' para evitar la generación de SQL,
    luego lee la estructura de la base de datos y genera una consulta SQL
    a partir del plan de acción del agente inteligente.
    """
    
    # 🛑 NUEVA LÓGICA: REUTILIZAR CONSULTA
    if plan.get("accion") == "reutilizar_consulta":
        previous_sql = plan.get("meta", {}).get("previous_sql")
        if previous_sql:
            print(f"🔄 Reutilizando SQL anterior (solicitud de cambio de visualización).")
            return {
                "sql": previous_sql,
                "descripcion": f"Reutilizando consulta para cambiar visualización a {plan.get('meta', {}).get('visualization', 'tabla')}."
            }
        else:
            print("⚠️ Acción 'reutilizar_consulta' detectada, pero sin SQL previo. Volviendo a generación normal.")
            
    # Flujo normal de generación
    schema, columns_by_table, foreign_keys = get_database_schema()
    if not schema:
        print("⚠️ No se pudo obtener el esquema de la base de datos.")
        return {"sql": None, "descripcion": "Error al obtener el esquema."}

    result = generate_sql_with_openai(plan, schema, foreign_keys, columns_by_table)

    if result.get("sql"):
        print(f"📜 Consulta generada exitosamente.")
    else:
        print(f"⚠️ No se pudo generar la consulta SQL.")

    return result