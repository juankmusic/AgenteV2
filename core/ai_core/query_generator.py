# ======================================================
# Archivo: core/ai_core/query_generator.py (El que debes guardar)
# ======================================================
import os
import re
import json
from dotenv import load_dotenv
from openai import OpenAI as OpenAIClient
from db.connection import connect_db
from typing import Dict, Any, Optional, List
from core.exceptions import InvalidOperationError

# ======================================================
# CONFIGURACIÓN INICIAL
# ======================================================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAIClient(api_key=OPENAI_API_KEY)

# ======================================================
# 1️⃣ FUNCIÓN: OBTENER ESTRUCTURA DE LA BASE DE DATOS (Corregida)
# ======================================================
def get_database_schema():
    """
    Recupera todas las tablas y columnas del esquema 'public' en PostgreSQL,
    en la estructura de diccionario anidado que espera el validador.
    """
    schema = {} # <-- El esquema se construirá aquí
    foreign_keys = [] 

    conn = connect_db()
    if not conn:
        print("❌ No se pudo conectar a la base de datos.")
        return schema, foreign_keys 

    try:
        with conn.cursor() as cur:
            
            # --- Corrección Lógica: Construir el 'schema' directamente ---
            
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
            
            # Construir el diccionario anidado que _get_column_type espera
            for table, col, dtype in rows:
                if table not in schema:
                    # Inicializa la estructura de diccionario anidada
                    schema[table] = {"columns": {}} 
                
                # Añade la columna y su tipo al diccionario 'columns'
                schema[table]["columns"][col] = dtype
            
            # --- Fin de la corrección ---

            # Obtener relaciones entre tablas (clave foránea) - (Sin cambios)
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

        # Usamos 'schema' que ahora está poblado
        print(f"📚 Esquema con relaciones detectado: {len(schema)} tablas encontradas.")
        return schema, foreign_keys

    except Exception as e:
        print(f"❌ Error obteniendo esquema: {e}")
        return {}, [] 
    finally:
        conn.close()


# ======================================================
# 2️⃣ FUNCIÓN: VALIDAR CONSULTAS SQL (Sin cambios)
# ======================================================
def validate_sql(sql: str) -> bool:
    """
    Analiza la consulta SQL generada y bloquea comandos destructivos o no seguros.
    """
    forbidden = r"\b(DROP|TRUNCATE|ALTER|DELETE(?!.*WHERE)|UPDATE(?!.*WHERE))\b"
    if re.search(forbidden, sql, re.IGNORECASE):
        print("⚠️ SQL potencialmente peligroso detectado y rechazado.")
        return False
    return True

# ======================================================
# VALIDACIÓN DE LÓGICA DEL PLAN (Corregida)
# ======================================================

def _get_column_type(table_name: str, col_name: str, schema: Dict[str, Any]) -> Optional[str]:
    """Obtiene el tipo de dato de una columna del schema (Ahora funcionará)."""
    try:
        # Acceder a la estructura: schema -> table_name -> 'columns' -> col_name
        table_info = schema.get(table_name, {})
        columns_dict = table_info.get("columns", {})
        col_type_raw = columns_dict.get(col_name)

        if not col_type_raw:
            return None
            
        # Normalizar tipos
        tipo = col_type_raw.lower()
        if "char" in tipo or "text" in tipo:
            return "text"
        if "int" in tipo or "numeric" in tipo or "real" in tipo or "double" in tipo or "money" in tipo:
            return "numeric"
        return tipo
        
    except Exception as e:
        print(f"⚠️ Error interno en _get_column_type: {e}")
        return None


def validate_plan_logic(plan: Dict[str, Any], schema: Dict[str, Any]):
    """
    Valida la lógica del plan antes de generar SQL.
    (Lógica de 'if not table' corregida)
    """
    aggs = plan.get("meta", {}).get("aggregations", [])
    table = plan.get("entidades", {}).get("tabla")

    # Si no hay tabla, debemos verificar si se pidieron agregaciones.
    if not table:
        if aggs: # Si hay agregaciones pero no tabla
            raise InvalidOperationError(
                f"No pude identificar sobre qué tabla aplicar la operación.",
                suggested_action="Por favor, especifica sobre qué datos quieres operar (ej. 'evaluaciones', 'usuarios')."
            )
        # Si no hay tabla Y no hay agregaciones, es un chat general,
        # así que omitimos la validación.
        print("⚠️ validate_plan_logic: No se encontró tabla en el plan, se omite validación.")
        return 

    # Si el schema (de la BD real) no tiene esa tabla
    if table not in schema:
        # La IA alucinó una tabla
         raise InvalidOperationError(
            f"La tabla '{table}' mencionada en el plan no existe en la base de datos.",
            suggested_action="Por favor, reformula tu pregunta usando tablas o entidades conocidas."
        )

    for agg in aggs:
        op = agg.get("op")
        col_name = agg.get("col")

        # Validación para operaciones matemáticas (SUM, AVG)
        if op in ["sum", "avg"]:
            if col_name == "*":
                raise InvalidOperationError(
                    f"La operación '{op}' no se puede aplicar a todas las columnas (*).",
                    suggested_action="¿Quizás quisiste decir 'contar' (count)?"
                )
            
            # Esta llamada ahora usará el 'schema' con la estructura correcta
            col_type = _get_column_type(table, col_name, schema)
            
            if col_type == "text":
                # Esta excepción AHORA SÍ se debe lanzar
                raise InvalidOperationError(
                    f"No es posible realizar la operación matemática '{op}' en la columna '{col_name}', ya que es de tipo texto.",
                    suggested_action=f"¿Quizás quisiste 'contar' (count) los registros de '{col_name}'?"
                )
            
            if col_type is None and col_name != "*":
                # La IA pudo alucinar una columna que no existe
                raise InvalidOperationError(
                    f"La columna '{col_name}' no parece existir en la tabla '{table}'.",
                    suggested_action="¿Podrías verificar el nombre de la columna?"
                )

# ======================================================
# 3️⃣ FUNCIÓN PRINCIPAL: GENERAR CONSULTA SQL (Modificada)
# ======================================================
# 👇 ESTA ES LA LÍNEA CLAVE QUE FALTA EN TU ARCHIVO
def generate_sql_with_openai(plan: dict, schema: dict, foreign_keys: list, last_error: Optional[str] = None) -> dict:
    """
    Usa un modelo de lenguaje para generar una consulta SQL válida y segura.
    Acepta 'last_error' para autocorrección.
    """
    plan_json = json.dumps(plan, ensure_ascii=False, indent=2)

    TABLES_PROHIBIDAS = ["respuesta"]
    filtered_schema = {k: v for k, v in schema.items() if k not in TABLES_PROHIBIDAS}

    schema_json = json.dumps(filtered_schema, ensure_ascii=False, indent=2)
    foreign_keys_json = json.dumps(foreign_keys, ensure_ascii=False, indent=2)

    # --- Sección del prompt para el error ---
    error_prompt_section = "" # Inicia vacío
    if last_error:
        # Si hay un error, lo formateamos para el prompt
        error_prompt_section = f"""
    3b. ⚠️ **ERROR A CORREGIR**:
    Tu intento anterior de generar esta consulta falló.
    **Error de la BD**: "{last_error}"
    Analiza este error. Si es 'Undefined Table' o 'Undefined Column', 
    es probable que hayas inventado un nombre. Revisa el esquema y el plan para corregirlo.
    No repitas el mismo error.
    """

    prompt = f"""
    Eres un generador experto de SQL para PostgreSQL. Tu tarea es generar consultas SQL válidas y seguras basadas en los siguientes datos:

    1️⃣ **Plan de acción del agente**:
    {plan_json}

    2️⃣ **Esquema de la base de datos (Tablas, Columnas y Tipos)**:
    {schema_json}

    3️⃣ **Relaciones (Claves Foráneas)**: 
    {foreign_keys_json}

    {error_prompt_section}

    4️⃣ **Instrucciones**:
    - Debes utilizar las claves foráneas para realizar `JOIN` entre las tablas relacionadas cuando sea necesario.
    - Si el plan menciona una tabla o columna, interpreta el contexto semántico para saber qué tabla y qué columna utilizar.
    - Asegúrate de que la consulta sea **segura** y que no contenga comandos destructivos como `DROP`, `DELETE` sin `WHERE`, o `TRUNCATE`.
    - Genera una consulta que **respete las relaciones y restricciones de la base de datos**.

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
# 4️⃣ FUNCIÓN: INTERFAZ PÚBLICA (Modificada)
# ======================================================
# 👇 Y ESTA ES LA OTRA LÍNEA CLAVE QUE FALTA EN TU ARCHIVO
def generate_query_from_plan(plan: dict, last_error: Optional[str] = None) -> dict:
    """
    Punto de entrada principal.
    Acepta 'last_error' para pasarlo al generador de SQL.
    """
    
    # --- Lógica de reutilización (Sin cambios) ---
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
            
    # --- Flujo normal de generación ---
    schema, foreign_keys = get_database_schema()
    if not schema:
        print("⚠️ No se pudo obtener el esquema de la base de datos.")
        return {"sql": None, "descripcion": "Error al obtener el esquema."} 
        
    try:
        # Validación previa (Sin cambios)
        validate_plan_logic(plan, schema)
    
    # --- Captura de error de validación (Sin cambios) ---
    except InvalidOperationError as e:
        print(f"⚠️ Validación de plan fallida: {e}")
        return {
            "sql": None, 
            "descripcion": str(e), 
            "error_type": "LogicalError", 
            "suggested_action": e.suggested_action
        }
        
    # Si la validación pasa, generamos el SQL
    # 👇 Se pasa 'last_error' al generador
    result = generate_sql_with_openai(plan, schema, foreign_keys, last_error)

    if result.get("sql"):
        print(f"📜 Consulta generada exitosamente.")
    else:
        print(f"⚠️ No se pudo generar la consulta SQL.")

    return result