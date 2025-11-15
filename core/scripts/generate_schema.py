import psycopg2
from collections import defaultdict


# ----------------------------
# 1. INFERENCIA DE DOMINIO
# ----------------------------
def infer_category(table_name: str) -> str:
    """
    Categorización automática para que el modelo
    NUNCA mezcle dominios de negocio.
    """
    t = table_name.lower()

    if "competencia" in t:
        return "competencias_transversales"
    if "analisis_organizacional" in t:
        return "analisis_organizacional"
    if "factores_clave" in t:
        return "factores_clave_exito"
    if "docente" in t:
        return "competencias_docentes"
    if "iluo" in t:
        return "iluo"
    if "usuario" in t:
        return "usuarios"

    # fallback seguro
    return "general"


# ----------------------------
# 2. EXTRACCIÓN DEL ESQUEMA
# ----------------------------
def extract_schema(connection_string: str):
    conn = psycopg2.connect(connection_string)
    cur = conn.cursor()

    # ---------- COLUMNAS ----------
    cur.execute("""
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position;
    """)

    columns_by_table = defaultdict(list)
    for table, col in cur.fetchall():
        columns_by_table[table].append(col)

    # ---------- FOREIGN KEYS ----------
    cur.execute("""
        SELECT
            tc.table_name,
            kcu.column_name,
            ccu.table_name AS foreign_table,
            ccu.column_name AS foreign_column
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu 
            ON tc.constraint_name = kcu.constraint_name
        JOIN information_schema.constraint_column_usage ccu
            ON ccu.constraint_name = tc.constraint_name
        WHERE tc.constraint_type = 'FOREIGN KEY'
          AND tc.table_schema = 'public'
        ORDER BY tc.table_name;
    """)

    foreign_keys = defaultdict(list)
    for table, col, ft, fc in cur.fetchall():
        foreign_keys[table].append({
            "column": col,
            "ref_table": ft,
            "ref_column": fc
        })

    cur.close()
    conn.close()

    # ---------- ENSAMBLE FINAL ----------
    result = {}

    for table in columns_by_table:
        result[table] = {
            "columns": columns_by_table[table],
            "category": infer_category(table),
            "foreign_keys": foreign_keys.get(table, [])
        }

    return result


# ----------------------------
# 3. USO DIRECTO
# ----------------------------
if __name__ == "__main__":
    CONNECTION = "dbname=bdgestion_agente user=postgres password=postgre host=localhost port=5432"

    schema = extract_schema(CONNECTION)

    print("\n=== ESQUEMA OPTIMIZADO ===")
    for table, info in schema.items():
        print(f"\n{table}:")
        print("  category:", info["category"])
        print("  columns:", info["columns"])
        print("  foreign_keys:", info["foreign_keys"])
