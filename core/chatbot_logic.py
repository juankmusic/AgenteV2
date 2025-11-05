# core/chat_logic.py
import os
import sys
import uuid
import re # <-- Añadido
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
# 👉 NUEVAS IMPORTACIONES
from core.ai_core.nlp_embeddings import get_last_action_plan, store_action_plan

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
# 👉 Se ha modificado la firma para incluir session_id
def deepseek_chat(session_id, question, context=None, history=None): 
    global SESSION_STATES
    
    # 1. 🔄 Inicializar o recuperar el estado de la sesión
    if session_id not in SESSION_STATES:
        SESSION_STATES[session_id] = {'last_chart_type': None}
        
    session_state = SESSION_STATES[session_id]
    
    # 1.1 📊 Intentar extraer el tipo de gráfico del mensaje actual
    current_chart_type = extract_chart_type(question)
    
    # 1.2 💡 Lógica de memoria/actualización
    if current_chart_type:
        # El usuario ha especificado un nuevo tipo de gráfico, ¡actualizar la memoria!
        session_state['last_chart_type'] = current_chart_type
        print(f"✅ Tipo de gráfico en memoria actualizado: {current_chart_type}")
    
    # Obtener el tipo de gráfico a USAR (el recordado)
    chart_type_to_use = session_state['last_chart_type']
    
    # --- PROMPT DEL SISTEMA (IDÉNTICO) ---
    system_prompt = (
        "Eres un asistente inteligente y amable. "
        "Puedes mantener una conversación general o analizar documentos PDF/CSV cargados. "
        "Eres un asistente inteligente y autónomo (AIGR). "
        "consultar bases de datos, generar informes y decidir cuándo graficar o sintetizar texto. "
        "Sé preciso, no inventes información, y explica tus resultados de forma clara. "
        "Si hay contexto, úsalo para responder basándote en el documento. "
        "Si no, responde de forma natural como un chatbot general. "
        "Sé claro, conciso y no inventes información. "
        "Eres un agente inteligente con memoria y capacidad de análisis contextual. "
        "Puedes mantener conversaciones generales, generar informes automáticos, "
        "consultar datos o analizar información compleja. "
        "Si el usuario hace una petición técnica o analítica, responde de forma estructurada."
    )
    
    # === 1️⃣ Análisis semántico profundo ===
    semantic_data = analyze_text(question)

    # === 2️⃣ Detección de intención cognitiva ===
    intent_data = detect_intention(semantic_data["texto"])
    tipo = intent_data.get("tipo", "conversacion")

    # Combinar entidades si el detector no las incluye
    if not intent_data.get("entidades"):
        intent_data["entidades"] = semantic_data.get("entidades", {})

    # === 3️⃣ Si es una intención cognitiva, activar modo AIGR ===
    if tipo in ["evaluar", "generar_informe", "consultar_datos", "guardar_resultado"]:
        print(f"🧠 Modo AIGR activo (intención: {tipo})")

        # 🔹 Cargar esquemas correctamente
        schema_semantic, schema_embeddings = schema_loader()
        
        # 3.0b Recuperar el plan anterior de la BD (si existe)
        last_plan = get_last_action_plan(session_id) # 👈 Recupera el plan anterior

        # 3.1 Planificación autónoma
        # Se pasa el last_plan para que el dynamic_planner decida si reutilizar el contexto
        plan = plan_actions(intent_data, schema_semantic, schema_embeddings, last_plan)
        
        # 3.2 Generar SQL dinámico
        # El generate_query_from_plan debe ahora verificar plan["accion"] == "reutilizar_consulta"
        sql_result = generate_query_from_plan(plan)
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
                password="pgadmin4",
                host="localhost",
                port=5432,
                database="bdgestion_octubre",
                query={"client_encoding": "UTF8"} # 👈 forzar encoding
            )

            engine = create_engine(connection_url, connect_args={"options": "-c client_encoding=WIN1252"})
            with engine.connect() as connection:
                df = pd.read_sql_query(text(sql), connection)

        except Exception as e:
            print(f"❌ Error ejecutando SQL dinámico: {e}")
            # Si hay un error de SQL, el flujo se detiene aquí, no persiste el plan.
            # Podríamos implementar un fallback de conversación aquí si es necesario.
            # Por ahora, simplemente retornamos el error.
            return f"❌ Error ejecutando SQL dinámico. Posiblemente la consulta generada no es válida para la base de datos: {e}"

        # 3.4 Generar informe o acción autónoma
        if plan["accion"] == "guardar_resultado":
            # 3.5 PERSISTIR EL NUEVO PLAN DE ACCIÓN Y EL SQL EJECUTADO
            store_action_plan(session_id, plan, executed_sql=sql)
            return "✅ Los datos fueron procesados y almacenados correctamente."
        
        # 3.5 PERSISTIR EL NUEVO PLAN DE ACCIÓN Y EL SQL EJECUTADO
        store_action_plan(session_id, plan, executed_sql=sql) # 👈 Guardamos el plan y el SQL
        
        # 3.6 Decidir si graficar o sintetizar
        
        # Obtener la sugerencia de visualización del plan (puede ser 'line')
        plan_vis_type = plan.get("meta", {}).get("visualization")
        
        # ----------------------------------------------------------------------
        # 💥 CORRECCIÓN DE PRIORIDAD: Priorizar la detección del mensaje actual (pie)
        # 1. Detección del mensaje actual (chart_type_to_use)
        # 2. Sugerencia del Plan (plan_vis_type)
        # 3. Default 'bar'
        # ----------------------------------------------------------------------

        final_chart_type = chart_type_to_use # 👈 Prioridad MÁXIMA a la detección explícita ('pie')
        
        if not final_chart_type: 
            # Si el mensaje no tenía gráfico, usar la sugerencia del plan ('line')
            final_chart_type = plan_vis_type 
            
        if not final_chart_type: 
            # Si ni el mensaje ni el plan lo tenían, usar el default
            final_chart_type = 'bar'

        # Si el plan sugirió una visualización, o hay una en memoria, o se preguntó explícitamente por un gráfico
        if final_chart_type or "gráfico" in question.lower():
            
            # Actualizamos la memoria local con el tipo de gráfico final
            if final_chart_type != session_state['last_chart_type']:
                session_state['last_chart_type'] = final_chart_type
                # Usamos un mensaje de debug más claro para ver qué se decidió
                print(f"✅ Tipo de gráfico final seleccionado: {final_chart_type}")

            # 👉 Llama a la función con el tipo de gráfico
            report = generate_report(df, question, final_chart_type) 
            
            # Generar mensaje de respuesta
            return f"🧩 Resultado de la consulta (mostrado como **gráfico de {final_chart_type}**):\n\n{report}"

        else:
            # Llama a la función SIN tipo de gráfico (sólo síntesis de texto/reporte)
            report = generate_report(df, question)

        return f"🧩 Resultado de la consulta:\n\n{report}"

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
