
# core/ai_core/report_synthesizer.py

import os
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI as OpenAIClient

# 👇 INICIO CAMBIOS PARA PLOTLY
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio 
# pio.to_html es clave para incrustar
# Se eliminan los imports de matplotlib, io, base64.
# 👆 FIN CAMBIOS PARA PLOTLY

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
    """

    # Limitar muestra para enviar al LLM
    sample = _limit_and_stringify(results, max_rows=5)
    data_sample = sample.to_dict(orient="records")
    today = datetime.now().strftime("%d/%m/%Y %H:%M")

    # Construir prompt controlado (omito por brevedad, es el mismo código)
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
    # ... (Función sin cambios, enfocada solo en la tabla HTML)
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

    html_table = df_small.to_html(classes="aigr-table", index=False, escape=True)
    title = "<div class='aigr-card'><strong>Tabla: datos relevantes (vista limitada)</strong>"
    footer = "<p class='table-footer'>Nota: la tabla muestra una vista limitada y columnas sensibles están enmascaradas.</p></div>"

    return title + html_table + footer


# ===========================
# 2️⃣ OPCIONAL: VISUALIZACIÓN (MIGRACIÓN A PLOTLY)
# ===========================
def generate_visualization(results: pd.DataFrame, user_input: str, chart_type: str = None) -> str:
    """
    Genera un gráfico interactivo usando Plotly.
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
        
        fig = go.Figure() # Figura por defecto si no se puede generar

        # 🎯 CONFIGURACIÓN DE LAYOUT CLARO FORZADO 💡
        # Asegura que el texto sea negro y el fondo blanco para legibilidad universal.
        light_theme_layout = go.Layout(
            paper_bgcolor='white',      # Fondo blanco para el área externa (donde van títulos)
            plot_bgcolor='white',       # Fondo blanco para el área de trazado
            font=dict(
                color='black'           # Texto general en negro
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(0, 0, 0, 0.1)',
                linecolor='black',
                tickfont=dict(color='black'),
                title_font=dict(color='black')
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(0, 0, 0, 0.1)',
                linecolor='black',
                tickfont=dict(color='black'),
                title_font=dict(color='black')
            ),
            title=dict(
                font=dict(color='black') # Título principal en negro
            ),
            legend=dict(
                font=dict(color='black'),
                bgcolor='rgba(255, 255, 255, 0.7)'
            )
        )

        # ========== Tipos de gráfico solicitados (final_chart_type) ==========
        if final_chart_type in ["bar", "barh"]:
            # Asumimos la primera categórica es el eje X y la primera numérica es el eje Y
            col_labels = categorical_cols[0] if len(categorical_cols) else results.columns[0]
            col_values = numeric_cols[0] if len(numeric_cols) else None
            
            orientation = 'h' if final_chart_type == 'barh' else 'v'
            x_col = col_values if orientation == 'h' else col_labels
            y_col = col_labels if orientation == 'h' else col_values

            if col_values:
                # Caso 1: Consulta ya Agregada (rol, cantidad_usuarios)
                fig = px.bar(results, x=x_col, y=y_col, orientation=orientation)
                title = f"Gráfico de {final_chart_type} de {col_labels} vs {col_values}"
            else:
                # Caso 2: Contar filas por categoría (datos crudos)
                count_data = results[col_labels].value_counts().head(10).reset_index()
                count_data.columns = ['Etiqueta', 'Conteo']
                
                x_count = 'Conteo' if orientation == 'h' else 'Etiqueta'
                y_count = 'Etiqueta' if orientation == 'h' else 'Conteo'
                
                fig = px.bar(count_data, x=x_count, y=y_count, orientation=orientation)
                title = f"Frecuencia (Conteo) de {col_labels}"
            
            fig.update_layout(title_text=title)


        elif final_chart_type == "pie":
            col_labels = categorical_cols[0] if len(categorical_cols) else results.columns[0]
            col_sizes = numeric_cols[0] if len(numeric_cols) else None
            
            if col_sizes:
                # Caso 1: Consulta ya Agregada
                fig = px.pie(results, values=col_sizes, names=col_labels)
            else:
                # Caso 2: DataFrame con datos crudos (se necesita contar)
                vc = results[col_labels].value_counts().head(6).reset_index()
                vc.columns = ['Etiqueta', 'Conteo']
                fig = px.pie(vc, values='Conteo', names='Etiqueta')
            
            fig.update_layout(title_text=f"Distribución de {col_labels}")


        elif final_chart_type == "line":
            fig = go.Figure()

            # Detectar columnas
            numeric_cols = results.select_dtypes(include=["number"]).columns
            x_candidates = results.select_dtypes(include=["datetime", "object", "category"]).columns

            if len(numeric_cols) == 0:
                fig.add_annotation(
                    text="No hay columnas numéricas para graficar",
                    xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
                )
            else:
                x_col = x_candidates[0] if len(x_candidates) > 0 else results.index
                y_cols = numeric_cols

                # Graficar cada columna numérica como línea
                for y in y_cols:
                    fig.add_trace(
                        go.Scatter(
                            x=results[x_col],
                            y=results[y],
                            mode='lines+markers',
                            name=str(y)
                        )
                    )

                fig.update_layout(title_text="Gráfico de líneas genérico")



        elif final_chart_type == "scatter":
            # Intentar dispersión con dos columnas (numéricas o categóricas)
            all_cols = list(results.columns)
            if len(all_cols) >= 2:
                x_col, y_col = all_cols[0], all_cols[1]
                fig = px.scatter(results, x=x_col, y=y_col, title=f"Dispersión de {x_col} vs {y_col}")
            else:
                fig.add_annotation(text="No hay suficientes columnas para dispersión", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

        else:  # auto (modo actual) o tipo no reconocido
            if len(numeric_cols) >= 2:
                # Usar Mapa de Calor (heatmap) para correlación
                corr = results[numeric_cols].corr().reset_index()
                corr_melted = corr.melt(id_vars='index', var_name='Variable_2', value_name='Correlacion')
                fig = px.density_heatmap(corr_melted, x='index', y='Variable_2', z='Correlacion', 
                                        color_continuous_scale='RdBu', title="Mapa de Correlación Numérica")
            elif len(categorical_cols) >= 1:
                # Por defecto, si hay datos categóricos, mostrar un gráfico de barras horizontales de frecuencia
                col = categorical_cols[0]
                vc = results[col].value_counts().head(10).reset_index()
                vc.columns = [col, 'Conteo']
                fig = px.bar(vc, x='Conteo', y=col, orientation='h', title=f"Frecuencia de {col}")
            else:
                fig.add_annotation(text="No hay datos visualizables", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

        # 📌 Aplicar el layout CLARO forzado
        fig.update_layout(light_theme_layout)

        # Estilos generales de la figura (eliminados o ajustados para evitar conflicto con light_theme_layout)
        fig.update_layout(
            margin=dict(l=20, r=20, t=50, b=20)
        )

        # Generar el HTML incrustable del gráfico Plotly
        # 'full_html=False' y 'include_plotlyjs='cdn'' son esenciales
        plot_html = pio.to_html(
            fig, 
            full_html=False, 
            include_plotlyjs='cdn',
            default_height='100%',
            default_width='100%'
        )

        # Envolver el HTML del gráfico en un contenedor con estilos para la app
        return f'<div class="aigr-card aigr-plotly-container" style="padding: 10px 0;"><strong>Visualización Interactiva</strong>{plot_html}</div>'


    except Exception as e:
        return f"<p class='error-message'>⚠️ No se pudo generar el gráfico: {e}</p>"


# ===========================
# 4️⃣ INTERFAZ PRINCIPAL
# ===========================
def generate_report(results: pd.DataFrame, user_input: str, chart_type: str = None) -> str:
    
    report_text = synthesize_from_results(results, user_input)
    table_html = generate_table_html(results, user_input)
    
    # 🎯 PUNTO CLAVE: Pasar el tipo de gráfico recordado (chart_type) a la visualización
    visual_html = generate_visualization(results, user_input, chart_type) 

    # convertir markdown a HTML 
    report_html = markdown(report_text)

    # Devolver el informe narrativo, la tabla (si se solicita) y el gráfico interactivo
    return f"<div class='bot-report'>{report_html}</div>{table_html}{visual_html}"
