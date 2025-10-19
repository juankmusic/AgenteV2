# core/ai_core/query_generator.py
from openai import OpenAI
from db.connection import connect_db

client = OpenAI()

DB_SCHEMA = """
Tablas disponibles:
1. evaluaciones(id, colaborador, area, puntaje, periodo, fecha)
2. colaboradores(id, nombre, cargo, area)
"""

def generate_query(intent_description: str):
    prompt = f"""
Eres un experto en SQL. Genera una consulta SQL segura solo de lectura.
Base de datos:
{DB_SCHEMA}

Intención del usuario:
{intent_description}

Responde solo con la consulta SQL.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": prompt}]
    )
    sql_query = response.choices[0].message.content.strip()

    # Seguridad: solo permitimos SELECT
    if not sql_query.lower().startswith("select"):
        raise ValueError("Solo se permiten consultas SELECT")
    return sql_query


def execute_query(sql_query):
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(sql_query)
            return cur.fetchall()
    finally:
        conn.close()
