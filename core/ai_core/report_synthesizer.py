# core/ai_core/report_synthesizer.py

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI as OpenAIClient
import matplotlib
matplotlib.use("Agg")  # backend sin interfaz gráfica (ideal para servidores)
import matplotlib.pyplot as plt

from markdown import markdown 

# ===========================
# CONFIGURACIÓN
# ===========================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAIClient(api_key=OPENAI_API_KEY)

# Columnas sensibles que deben enmascararse si aparecen
SENSITIVE_COLS = {"correo", "contrasena", "password", "email", "telefono", "tel"}

# Palabras que indican que el usuario quiere una tabla o un gráfico
TABLE_KEYWORDS = ["tabla", "muéstrame una tabla", "muestrame una tabla", "mostrar tabla", "mostrar una tabla", "tabla con"]
GRAPH_KEYWORDS = ["gráfico", "grafico", "visualiza", "muestra un gráfico", "plot", "grafique", "visualizar"]
# Tipos específicos de gráficos
GRAPH_TYPE_KEYWORDS = {
    "barras": "bar",
    "barras horizontales": "barh",
    "circular": "pie",
    "pastel": "pie",
    "líneas": "line",
    "lineas": "line",
    "dispersión": "scatter",
    "puntos": "scatter"
}

# --- MODIFICACIÓN: Esta función ahora solo es una opción de fallback ---
def _detect_graph_type(user_input: str) -> str:
    """
    Detecta el tipo de gráfico solicitado por el usuario a partir del texto.
    """
    txt = user_input.lower()
    for key, gtype in GRAPH_TYPE_KEYWORDS.items():
        if key in txt:
            return gtype
    return "auto"  # por defecto


# ===========================
# UTILIDADES
# ===========================
def _user_wants_table(user_input: str) -> bool:
    txt = user_input.lower()
    return any(k in txt for k in TABLE_KEYWORDS)

# Esta función ya no es estrictamente necesaria si se usa la memoria,
# pero la mantenemos para compatibilidad con solicitudes directas.
def _user_wants_graph(user_input: str) -> bool:
    txt = user_input.lower()
    return any(k in txt for k in GRAPH_KEYWORDS)

def _mask_sensitive_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if col.lower() in SENSITIVE_COLS:
            df[col] = df[col].apply(lambda v: "****@oculto" if pd.notna(v) else v)
    return df

def _limit_and_stringify(df: pd.DataFrame, max_rows: int = 5) -> pd.DataFrame:
    sample = df.head(max_rows).copy()
    # Convertir a strings para evitar problemas de serialización
    for c in sample.columns:
        sample[c] = sample[c].astype(str)
    return sample

# ===========================
# 1️⃣ SÍNTESIS DE RESULTADOS (solo texto)
# ===========================
def synthesize_from_results(results: pd.DataFrame, user_input: str) -> str:
    """
    Analiza los resultados de la consulta y genera un informe narrativo y seguro.
    No muestra tablas ni datos sensibles.
    """

    # Limitar muestra para enviar al LLM
    sample = _limit_and_stringify(results, max_rows=5)
    data_sample = sample.to_dict(orient="records")
    today = datetime.now().strftime("%d/%m/%Y %H:%M")

    # Construir prompt controlado
    prompt = f"""
    Eres un analista cognitivo que genera informes concisos y confidenciales.
    Tu tarea es describir los datos sin agregar ejemplos ni suposiciones.
    Si un valor no tiene un significado textual (como IDs numéricos), no lo interpretes ni inventes.
    El usuario pidió: "{user_input}"

    Solo tienes una muestra limitada de los datos (para contexto), no muestres tablas ni datos sensibles:
    {json.dumps(data_sample, ensure_ascii=False, indent=2)}

    Instrucciones:
    - Redacta un informe ejecutivo y analítico en español.
    - No incluyas tablas ni listados de datos.
    - No muestres ni reconstruyas información sensible (correos, contraseñas, etc.).
    - No sugieras gráficos si el usuario no los pidió explícitamente.
    - Evita firmas o placeholders como [Su Nombre] o [Fecha Actual]; incluye la fecha real.
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un generador de informes profesional. Responde en español con tono analítico y claro."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        analysis = response.choices[0].message.content.strip()
        header = f"### Informe Analítico Automatizado\n**Fecha de generación:** {today}\n\n"
        return header + analysis
    except Exception as e:
        # 🛑 Corregido: Usar clase CSS para errores en lugar de estilo en línea fijo
        return f"<p class='error-message'>❌ Error generando informe: {e}</p>"

def generate_table_html(results: pd.DataFrame, user_input: str) -> str:
    """
    Genera una tabla HTML solo si el usuario la solicita. 
    Enmascara columnas sensibles y limita el número de filas/columnas.
    """
    if not _user_wants_table(user_input):
        return ""

    if results is None or results.empty:
        return "<p>⚠️ No hay datos para mostrar en tabla.</p>"

    # Enmascarar columnas sensibles
    df = _mask_sensitive_columns(results)

    # Limitar columnas: elegimos hasta 8 columnas (priorizamos no mostrar demasiadas)
    max_cols = 8
    cols = list(df.columns)[:max_cols]
    df_small = df[cols].head(10).copy()  # máximo 10 filas visibles

    # Mejorar visualización: transformar NA en vacío
    df_small = df_small.fillna("")

    # 🛑 Corregido: Se elimina el bloque de estilo completo.
    # style = """ ... """

    html_table = df_small.to_html(classes="aigr-table", index=False, escape=True)
    # Se usa la clase 'aigr-card' para el div contenedor.
    title = "<div class='aigr-card'><strong>Tabla: datos relevantes (vista limitada)</strong>"
    # 🛑 Corregido: Se elimina el estilo en línea del footer y se usa la clase 'table-footer'.
    footer = "<p class='table-footer'>Nota: la tabla muestra una vista limitada y columnas sensibles están enmascaradas.</p></div>"

    # 🛑 Corregido: Solo se retorna el contenido (sin 'style' en la concatenación)
    return title + html_table + footer

# ===========================
# 2️⃣ OPCIONAL: VISUALIZACIÓN
# ===========================
# 👉 MODIFICACIÓN: Aceptar chart_type de la memoria de sesión
def generate_visualization(results: pd.DataFrame, user_input: str, chart_type: str = None) -> str:
    """
    Genera un gráfico basado en la intención del usuario (memoria o detección).
    """

    # Determinar el tipo de gráfico a usar: 1. Memoria/Explicit; 2. Detección por texto; 3. Auto
    final_chart_type = chart_type if chart_type else _detect_graph_type(user_input)

    # Si no se ha solicitado un gráfico y no hay tipo en memoria, no generar.
    if not _user_wants_graph(user_input) and not chart_type:
        return ""


    try:
        if results is None or results.empty:
            return "<p>⚠️ No hay datos para graficar.</p>"

        numeric_cols = results.select_dtypes(include=["number"]).columns
        categorical_cols = results.select_dtypes(include=["object", "category"]).columns

        # Se asume que el backend matplotlib está configurado para colores neutros o se controlan por CSS si se inyectan como SVG/HTML (aquí es PNG)
        plt.figure(figsize=(8, 5))

        # ========== Tipos de gráfico solicitados (final_chart_type) ==========
        if final_chart_type in ["bar", "barh"]:
            # Asumimos la primera categórica es el eje X y la primera numérica es el eje Y
            col_labels = categorical_cols[0] if len(categorical_cols) else results.columns[0]
            col_values = numeric_cols[0] if len(numeric_cols) else None

            if col_values:
                # Caso 1: Consulta ya Agregada (rol, cantidad_usuarios) o dos columnas
                # Usar las columnas directamente y establecer la categórica como índice para graficar
                plot_data = results.set_index(col_labels)[col_values]
                plot_data.plot(kind=final_chart_type)
                plt.xlabel(col_labels)
                plt.ylabel(col_values)
            else:
                # Caso 2: Contar filas por categoría (datos crudos)
                results[col_labels].value_counts().head(10).plot(kind=final_chart_type)
                plt.ylabel("Conteo de registros") # Etiqueta correcta para value_counts
            
            plt.title(f"Gráfico de {final_chart_type} de {col_labels}")

        elif final_chart_type == "pie":
            # Lógica de Detección de Columna de Valores
            col_labels = categorical_cols[0] if len(categorical_cols) else results.columns[0]
            col_sizes = numeric_cols[0] if len(numeric_cols) else None
            
            if col_sizes:
                # Caso 1: Consulta ya Agregada (como la tuya: rol, cantidad_usuarios)
                # Usar las columnas directamente. Esto corrige el 33%, 33%, 33%
                sizes = results[col_sizes]
                labels = results[col_labels]
            else:
                # Caso 2: DataFrame con datos crudos (se necesita contar)
                # Usar value_counts, limitado a 6 categorías para no saturar el pie.
                vc = results[col_labels].value_counts().head(6)
                labels = vc.index
                sizes = vc.values

            plt.figure(figsize=(6,6))
            plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
            plt.title(f"Distribución de {col_labels}")
            plt.axis("equal")  # círculo perfecto


        elif final_chart_type == "line":
            if len(numeric_cols) >= 2:
                results.plot(x=numeric_cols[0], y=numeric_cols[1:], kind="line")
            else:
                col = numeric_cols[0] if len(numeric_cols) else results.columns[0]
                results[col].plot(kind="line")
            plt.title("Evolución temporal o secuencial")

        elif final_chart_type == "scatter":
            # Permitir scatter incluso si las columnas son categóricas
            if len(categorical_cols) >= 2:
                x = pd.factorize(results[categorical_cols[0]])[0]
                y = pd.factorize(results[categorical_cols[1]])[0]
                plt.scatter(x, y)

                # Etiquetas de los ticks
                plt.xticks(range(len(results[categorical_cols[0]].unique())), results[categorical_cols[0]].unique(), rotation=45)
                plt.yticks(range(len(results[categorical_cols[1]].unique())), results[categorical_cols[1]].unique())

                plt.xlabel(categorical_cols[0])
                plt.ylabel(categorical_cols[1])
                plt.title("Gráfico de dispersión categórico")
            else:
                plt.text(0.5, 0.5, "No hay suficientes columnas para dispersión", ha="center")


        else:  # auto (modo actual) o tipo no reconocido
            if len(numeric_cols) >= 2:
                corr = results[numeric_cols].corr()
                plt.imshow(corr, cmap="coolwarm", aspect="auto")
                plt.colorbar()
                plt.title("Mapa de Correlación")
            elif len(categorical_cols) >= 1:
                col = categorical_cols[0]
                vc = results[col].value_counts().head(10)
                vc.plot(kind="barh")
                plt.title(f"Frecuencia de {col}")
            else:
                plt.text(0.5, 0.5, "No hay datos visualizables", ha="center")

        # ===================================================
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()

        # 🛑 Corregido: Se elimina el estilo en línea de la imagen (excepto para dimensiones y bordes que son fijos)
        return f'<div style="text-align:center;margin-top:12px;"><img src="data:image/png;base64,{img_base64}" alt="Gráfico generado" style="max-width:100%;border-radius:10px;box-shadow:0 6px 18px rgba(2,6,23,0.08)"></div>'

    except Exception as e:
        # 🛑 Corregido: Usar clase CSS para errores
        return f"<p class='error-message'>⚠️ No se pudo generar el gráfico: {e}</p>"


# ===========================
# 4️⃣ INTERFAZ PRINCIPAL
# ===========================
# 👉 MODIFICACIÓN: Aceptar chart_type de la memoria de sesión
def generate_report(results: pd.DataFrame, user_input: str, chart_type: str = None) -> str:
    
    report_text = synthesize_from_results(results, user_input)
    table_html = generate_table_html(results, user_input)
    
    # 🎯 PUNTO CLAVE: Pasar el tipo de gráfico recordado (chart_type) a la visualización
    visual_html = generate_visualization(results, user_input, chart_type) 

    # convertir markdown a HTML 
    report_html = markdown(report_text)

    # 🛑 Corregido: Se elimina el style en línea con color fijo y se usa la clase 'bot-report'
    return f"<div class='bot-report'>{report_html}</div>{table_html}{visual_html}"
