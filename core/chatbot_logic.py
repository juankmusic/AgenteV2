# core/chat_logic.py
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
from core.ai_core.nlp_embeddings import analyze_text
from core.ai_core.intention_analyzer import detect_intention
from core.ai_core.dynamic_planner import plan_actions
from core.ai_core.nlp_embeddings import analyze_text
from core.ai_core.query_generator import generate_query_from_plan
from core.ai_core.report_synthesizer import generate_report

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

# --- (El resto de tu código se deja idéntico) ---
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
def deepseek_chat(question, context=None, history=None):
    system_prompt = (
        "Eres un asistente inteligente y amable. "
        "Puedes mantener una conversación general o analizar documentos PDF/CSV cargados. "
        "Eres un asistente inteligente y autónomo (AIGR). "
        "consultar bases de datos, generar informes y decidir cuándo graficar o sintetizar texto. "
        "Sé preciso, no inventes información, y explica tus resultados de forma clara."
        "Si hay contexto, úsalo para responder basándote en el documento. "
        "Si no, responde de forma natural como un chatbot general. "
        "Sé claro, conciso y no inventes información."
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

        # 3.1 Planificación autónoma
        plan = plan_actions(intent_data)

        # 3.2 Generar SQL dinámico
        sql_result = generate_query_from_plan(plan)
        sql = sql_result.get("sql")

        if not sql:
            return "No pude generar una consulta válida basada en tu solicitud."

        print(f"📜 SQL generado:\n{sql}")

        # 3.3 Ejecutar SQL
        try:
            # ✅ Construimos URL con codificación explícita
            connection_url = URL.create(
                drivername="postgresql+psycopg2",
                username="postgres",
                password="postgre",
                host="localhost",
                port=5432,
                database="bdgestionactual",
                query={"client_encoding": "WIN1252"}  # 👈 clave: forzar encoding
            )

            engine = create_engine(connection_url, connect_args={"options": "-c client_encoding=WIN1252"})
            with engine.connect() as connection:
                df = pd.read_sql_query(text(sql), connection)

        except Exception as e:
            print(f"❌ Error ejecutando SQL dinámico: {e}")
            df = pd.DataFrame()
        finally:
            connection.close()

        # 3.4 Generar informe o acción autónoma
        if plan["accion"] == "guardar_resultado":
            return "✅ Los datos fueron procesados y almacenados correctamente."
        
        # Decidir si graficar o no
        if "gráfico" in question.lower() or plan["accion"] == "visualizar_datos":
            report = generate_report(df, question + " gráfico")
        else:
            report = generate_report(df, question)

        return f"🧩 Resultado de la consulta:\n\n{report}"

    # === 4️⃣ Si no hay intención cognitiva, usar modo conversación ===
    semantic_contexts = search_similar_embeddings(question, top_k=3)

    if semantic_contexts:
        print(f"📚 Se encontraron {len(semantic_contexts)} contextos relevantes.")
        context_text = "\n\n".join(
            [f"Contexto {i+1} (similitud {round(c['similaridad'], 3)}): {c['texto']}" 
            for i, c in enumerate(semantic_contexts)]
        )
        question += f"\n\nUsa el siguiente contexto para responder:\n{context_text}"
        user_prompt = f"Pregunta: {question}"
        if context:
            user_prompt += f"\n\nContexto:\n{context}"

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