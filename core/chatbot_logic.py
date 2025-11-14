import os
import sys
import uuid
import re 
import pandas as pd
import psycopg2
import pypdf
from dotenv import load_dotenv
from openai import OpenAI
from db.connection import connect_db 
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


# Importar módulos del núcleo de IA'
from core.ai_core.nlp_embeddings import search_similar_embeddings
from core.ai_core.nlp_embeddings import extract_entities
from core.ai_core.intention_analyzer import detect_intention
from core.ai_core.dynamic_planner import plan_actions
from core.ai_core.nlp_embeddings import analyze_text
from core.ai_core.query_generator import generate_query_from_plan
from core.ai_core.report_synthesizer import generate_report
from core.ai_core.schema_loader import load_schema as schema_loader
from core.ai_core.nlp_embeddings import get_last_action_plan, store_action_plan

import logging # Para logging de errores
from core.exceptions import LogicalError, InvalidOperationError, InvalidVisualizationError

# ===============================
# CONFIGURACIÓN
# ===============================
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not DEEPSEEK_API_KEY or not OPENAI_API_KEY:
    print("❌ Falta API Key en .env")
    sys.exit(1)

OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY)
DEEPSEEK_CLIENT = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")

# ===============================
# MEMORIA DE SESIÓN GLOBAL 🧠 (Para almacenar el estado de la conversación)
# ===============================
# Estructura: {'session_id': {'last_chart_type': 'bar' | 'pie' | 'line' | None, ...}}
SESSION_STATES = {}


# ===============================
# UTILIDAD DE EXTRACCIÓN
# ===============================
def extract_chart_type(text):
    """Extrae el tipo de gráfico solicitado del texto."""
    text_lower = text.lower()
    
    # Mapeo de términos a tipos de gráfico estándar
    chart_mapping = {
        "barras": "bar",
        "bar": "bar",
        "pastel": "pie",
        "circular": "pie",
        "dona": "pie",
        "líneas": "line",
        "linea": "line",
        "dispersión": "scatter",
        "puntos": "scatter",
        "histograma": "histogram",
        "histo": "histogram"
    }
    
    # Buscar cualquier término de gráfico en el texto
    for term, chart_type in chart_mapping.items():
        # Usar \b para coincidencia de palabra completa
        if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
            return chart_type
            
    return None # Si no se encuentra un tipo específico


# ===============================
# CHAT & SESIONES
# ===============================
# ... (Funciones save_message, get_history, list_sessions, select_or_create_session SIN CAMBIOS) ...
def save_message(session_id, role, msg):
    conn = connect_db()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            db_role = "bot" if role == "assistant" else role
            cur.execute("INSERT INTO chatbot_logs (session_id, role, message) VALUES (%s,%s,%s)", (session_id, db_role, msg))
            conn.commit()
    finally:
        conn.close()


def get_history(session_id):
    conn = connect_db()
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT role, message FROM chatbot_logs WHERE session_id=%s ORDER BY timestamp", (session_id,))
            return [{"role": "assistant" if r[0]=="bot" else r[0], "content": r[1]} for r in cur.fetchall()]
    finally:
        conn.close()


def list_sessions():
    conn = connect_db()
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT session_id, COUNT(*) FROM chatbot_logs GROUP BY session_id ORDER BY MAX(timestamp) DESC")
            return cur.fetchall()
    finally:
        conn.close()


def select_or_create_session():
    sessions = list_sessions()
    if sessions:
        print("\n📂 Sesiones previas:")
        for i, (sid, count) in enumerate(sessions):
            print(f"{i+1}. {sid} ({count} mensajes)")
        print(f"{len(sessions)+1}. Nueva sesión")
        ch = input("Elige: ")
        if ch.isdigit() and 1 <= int(ch) <= len(sessions):
            return sessions[int(ch)-1][0]
    return str(uuid.uuid4())

# ===============================
# MODELO DE CHAT
# ===============================
def deepseek_chat(session_id, question, context=None, history=None): 
    global SESSION_STATES
    
    # 1. 🔄 Lógica de sesión (SIN CAMBIOS)
    if session_id not in SESSION_STATES:
        SESSION_STATES[session_id] = {'last_chart_type': None}
    session_state = SESSION_STATES[session_id]
    current_chart_type = extract_chart_type(question)
    if current_chart_type:
        session_state['last_chart_type'] = current_chart_type
        print(f"✅ Tipo de gráfico en memoria actualizado: {current_chart_type}")
    chart_type_to_use = session_state['last_chart_type']
    
    # --- PROMPT DEL SISTEMA (SIN CAMBIOS) ---
    system_prompt = (
        "Eres un asistente inteligente y amable. "
        # ... (resto del prompt sin cambios) ...
        "Si el usuario hace una petición técnica o analítica, responde de forma estructurada."
    )
    
    # === 1️⃣ Análisis semántico profundo (SIN CAMBIOS) ===
    semantic_data = analyze_text(question)

    # === 2️⃣ Detección de intención cognitiva (SIN CAMBIOS) ===
    intent_data = detect_intention(semantic_data["texto"])
    tipo = intent_data.get("tipo", "conversacion")
    confianza = intent_data.get("confianza", 1.0)

    if not intent_data.get("entidades"):
        intent_data["entidades"] = semantic_data.get("entidades", {})

    # === Manejo de Ambigüedad (SIN CAMBIOS) ===
    CONFIDENCE_THRESHOLD = 0.5 
    if tipo in ["evaluar", "generar_informe", "consultar_datos"] and confianza < CONFIDENCE_THRESHOLD:
        logging.warning(f"Confianza baja ({confianza}) para intención '{tipo}'. Pidiendo clarificación.")
        return (
            "No estoy completamente seguro de tu solicitud. "
            f"Detecté que podrías querer '{tipo}', pero la petición es ambigua. "
            "¿Podrías reformular tu pregunta de forma más específica?"
        )

    # === 3️⃣ Si es una intención cognitiva, activar modo AIGR ===
    if tipo in ["evaluar", "generar_informe", "consultar_datos", "guardar_resultado"]:
        print(f"🧠 Modo AIGR activo (intención: {tipo}, confianza: {confianza})")
        
        # ❗️ Envolvemos todo el flujo AIGR en un try/except general
        try:
            # 🔹 Cargar esquemas correctamente
            schema_semantic, schema_embeddings = schema_loader()
            
            # 3.0b Recuperar el plan anterior de la BD (si existe)
            last_plan = get_last_action_plan(session_id) 

            # 3.1 Planificación autónoma
            plan = plan_actions(intent_data, schema_semantic, schema_embeddings, last_plan)
            
            # 3.2 Generar SQL dinámico
            sql_result = generate_query_from_plan(plan)
            
            # --- Manejo de error de validación (SIN CAMBIOS) ---
            if sql_result.get("error_type") == "LogicalError":
                error_msg = sql_result.get("descripcion", "Error lógico en la solicitud.")
                suggestion = sql_result.get("suggested_action")
                response = f"❌ **Error en la operación:** {error_msg}"
                if suggestion:
                    response += f"\n\n**Sugerencia:** {suggestion}"
                return response
                
            sql = sql_result.get("sql")

            if not sql:
                return "No pude generar una consulta válida basada en tu solicitud."

            print(f"📜 SQL generado:\n{sql}")

            # 3.3 Ejecutar SQL
            df = pd.DataFrame()
            try:
                connection_url = URL.create(
                    drivername="postgresql+psycopg2",
                    username="postgres",
                    password="postgre",
                    host="localhost",
                    port=5432,
                    database="bdgestion_agente",
                    query={"client_encoding": "UTF8"} 
                )

                engine = create_engine(connection_url, connect_args={"options": "-c client_encoding=WIN1252"})
                with engine.connect() as connection:
                    df = pd.read_sql_query(text(sql), connection)

            # 👇 ==================================================
            # 👇 INICIO DE LA CORRECCIÓN 2 (chat_logic.py)
            # 👇 ==================================================
            except Exception as e:
                print(f"❌ Error ejecutando SQL dinámico: {e}")
                
                # Interpretar el error de BD para dar una respuesta semántica
                # NUNCA mostrar 'e' o 'sql' al usuario.
                error_str = str(e).lower() 
                user_message = "❌ Hubo un error al procesar tu consulta."

                if "undefined function" in error_str and "sum(character varying)" in error_str:
                    # Error "Suma de Nombres" (si la validación previa fallara)
                    user_message = ("❌ **Error en la operación:** No es posible realizar una operación matemática (como 'suma') sobre una columna de texto (como 'nombres').")
                    user_message += "\n\n**Sugerencia:** ¿Quizás quisiste 'contar' (count) los registros?"
                
                elif "undefined table" in error_str or "falta una entrada para la tabla" in error_str:
                    # Error "Facultad" (alucinación de tabla)
                    user_message = ("❌ **Error en la consulta:** No pude encontrar todos los datos que mencionaste (por ejemplo, 'facultad'). Parece que esa tabla o categoría no existe en mi base de datos.")
                    user_message += "\n\n**Sugerencia:** ¿Podrías verificar que los términos que usas (como áreas, departamentos o tablas) sean correctos?"

                elif "undefined column" in error_str:
                    col_match = re.search(r'columna "([^"]+)" no existe', error_str)
                    col = col_match.group(1) if col_match else "desconocida"
                    user_message = (f"❌ **Error en la consulta:** La columna '{col}' no parece existir en el contexto de tu solicitud.")
                
                else:
                    # Fallback genérico pero limpio
                    user_message = "❌ No pude procesar tu solicitud. La consulta generada no fue válida para la base de datos y causó un error."
                
                return user_message
            # 👆 ==================================================
            # 👆 FIN DE LA CORRECCIÓN 2
            # 👆 ==================================================


            # 3.4 Generar informe o acción autónoma
            # --- Este bloque try/except ya estaba bien (SIN CAMBIOS) ---
            try:
                if plan["accion"] == "guardar_resultado":
                    store_action_plan(session_id, plan, executed_sql=sql)
                    return "✅ Los datos fueron procesados y almacenados correctamente."
                
                store_action_plan(session_id, plan, executed_sql=sql)
                
                # ... (Lógica para decidir 'final_chart_type' sin cambios) ...
                plan_vis_type = plan.get("meta", {}).get("visualization")
                final_chart_type = chart_type_to_use or plan_vis_type or 'bar'
                
                if final_chart_type or "gráfico" in question.lower():
                    if final_chart_type != session_state['last_chart_type']:
                        session_state['last_chart_type'] = final_chart_type
                        print(f"✅ Tipo de gráfico final seleccionado: {final_chart_type}")

                    report = generate_report(df, question, final_chart_type) 
                    return f"🧩 Resultado de la consulta (mostrado como **gráfico de {final_chart_type}**):\n\n{report}"

                else:
                    report = generate_report(df, question) # Sin gráfico
                return f"🧩 Resultado de la consulta:\n\n{report}"

            # 🚀 MANEJO DE ERROR LÓGICO (Gráfico inválido - SIN CAMBIOS)
            except InvalidVisualizationError as e:
                print(f"🎨 Error de visualización detectado: {e}")
                report_text_only = generate_report(df, question, chart_type=None, force_text_only=True)
                error_msg = str(e)
                suggestion = e.suggested_action
                
                response = f"{report_text_only}\n\n"
                response += f"⚠️ **Nota sobre el gráfico:** {error_msg}"
                if suggestion:
                    response += f"\n\n**Sugerencia:** {suggestion}"
                return response
            
            except Exception as e:
                # Captura de error genérico en la generación del reporte
                print(f"❌ Error generando el reporte (post-SQL): {e}")
                return f"❌ Error al sintetizar el reporte: {e}"

        # Captura de error genérico en el flujo AIGR (planificación)
        # --- Este bloque ya estaba bien (SIN CAMBIOS) ---
        except Exception as e:
             # Si el error es una de nuestras excepciones lógicas (ej. de plan_actions)
             if isinstance(e, LogicalError):
                 logging.error(f"Error lógico (inesperado) detectado: {e}")
                 response = f"❌ **Error en la operación:** {str(e)}"
                 if e.suggested_action:
                     response += f"\n\n**Sugerencia:** {e.suggested_action}"
                 return response
                 
             logging.exception(f"❌ Error crítico en el flujo AIGR: {e}")
             return f"❌ Ocurrió un error inesperado al procesar tu solicitud: {e}"

    # === 4️⃣ Si no hay intención cognitiva, usar modo conversación (IDÉNTICO) ===
    semantic_contexts = search_similar_embeddings(question, top_k=3)

    user_prompt = question
    if semantic_contexts:
        print(f"📚 Se encontraron {len(semantic_contexts)} contextos relevantes.")
        context_text = "\n\n".join(
            [f"Contexto {i+1} (similitud {round(c['similaridad'], 3)}): {c['texto']}" 
             for i, c in enumerate(semantic_contexts)]
        )
        user_prompt += f"\n\nUsa el siguiente contexto para responder:\n{context_text}"
        if context:
            user_prompt += f"\n\nContexto adicional:\n{context}"

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages += history[-6:]
    messages.append({"role": "user", "content": user_prompt})

    try:
        res = OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=800,
            temperature=0.7
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Error en chat: {e}")
        return "No pude generar una respuesta en este momento."