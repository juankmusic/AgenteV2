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
    Ten en cuenta que el plan puede incluir 'tablas_relacionadas'.
    Si existen, combina los datos mediante JOIN basados en las claves lógicas (por ejemplo: id_rol, id_cargo, id_equipo, id_usuario, etc.).
    Nunca pidas datos al usuario: la información está en la base de datos que ya conoces.

    Dispones de la siguiente información del agente inteligente:
    Plan de acción del agente:
    {json.dumps(plan, ensure_ascii=False, indent=2)}

    Estructura de la base de datos:
    {schema_json}

    INSTRUCCIONES IMPORTANTES:

    1️⃣ **Contexto Semántico**
    El texto del usuario y su intención provienen de un modelo cognitivo con embeddings y detección de intención.
    Esto significa que debes interpretar lo que el usuario *quiere* hacer, no solo las palabras literales.
    Ejemplo:
    - Si el usuario habla de “colaboradores” o “empleados”, probablemente se refiere a la tabla `usuario`.
    - Si menciona “roles”, “cargos” o “equipos”, debes considerar las tablas `rol`, `cargo` y `equipo` respectivamente.
    - Si menciona “desempeño”, “nivel de contribución” o “evaluación”, involucra `nivel_contribucion` o `evaluacion`.
    - Si una tabla contiene columnas como 'nombre' o 'descripcion', usa esas columnas en lugar de IDs.
    - Evita mostrar valores numéricos de referencia (como id_rol o id_cargo).
    - Siempre usa JOINs para obtener nombres legibles.
    2️⃣ **Relaciones comunes**
    Estas relaciones existen y puedes usarlas libremente para crear JOINs:
    - usuario.id_rol → rol.id
    - usuario.id_cargo → cargo.id
    - usuario.id_equipo → equipo.id
    - usuario.nivel_contribucion_id → nivel_contribucion.id
    - evaluacion.id_usuario → usuario.id

    3️⃣ **Objetivo**
    Genera una consulta SQL *válida y segura* que satisfaga el propósito del plan del agente, usando las tablas necesarias.
    Usa filtros si el plan incluye criterios (como equipo, año, periodo, etc.).

    4️⃣ **Política de Seguridad**
    - Nunca muestres columnas sensibles como contraseñas o embeddings.
    - Devuelve solo información general, resumida o agregada (por ejemplo: conteos, promedios, o listas de nombres y roles).
    - Si hay dudas sobre qué mostrar, prioriza información no sensible.

    5️⃣ **Formato de salida**
    Devuelve SOLO un JSON válido con esta estructura exacta:
    {{
    "sql": "SELECT ...",
    "descripcion": "Breve explicación natural de la consulta generada"
    }}

    Tu salida debe ser estrictamente JSON válido, sin texto adicional.
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
