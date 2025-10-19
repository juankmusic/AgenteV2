# core/chat_logic.py
import os
import sys
import uuid
import re
import pandas as pd
import pypdf
from dotenv import load_dotenv
from openai import OpenAI
from db.connection import connect_db 

# Importar módulos del núcleo de IA
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
        "Si hay contexto, úsalo para responder basándote en el documento. "
        "Si no, responde de forma natural como un chatbot general. "
        "Sé claro, conciso y no inventes información."
        "Eres un agente inteligente con memoria y capacidad de análisis contextual. "
        "Puedes mantener conversaciones generales, generar informes automáticos, "
        "consultar datos o analizar información compleja. "
        "Si el usuario hace una petición técnica o analítica, responde de forma estructurada."
    )
    # === 1️⃣ Análisis de intención ===
    intent_data = detect_intention(question)
    tipo = intent_data.get("tipo")

    # Si el usuario pide una acción (no simple chat)
    if tipo in ["evaluar", "generar_informe", "consultar_datos", "guardar_resultado"]:
        print(f"🧠 Activando modo AIGR (intención: {tipo})")

        # === 2️⃣ Planificación dinámica ===
        plan = plan_actions(intent_data)

        # === 3️⃣ Generar SQL con base en el plan ===
        sql_result = generate_query_from_plan(plan)
        sql = sql_result.get("sql")

        if not sql:
            return "No pude generar una consulta válida basada en tu solicitud."

        print(f"📜 SQL generado:\n{sql}")

        # === 4️⃣ Ejecutar SQL ===
        conn = connect_db()
        try:
            df = pd.read_sql(sql, conn)
        except Exception as e:
            print(f"❌ Error ejecutando SQL: {e}")
            df = pd.DataFrame()
        finally:
            conn.close()

        # === 5️⃣ Generar informe cognitivo ===
        report = generate_report(df, question)
        return f"🧩 Consulta realizada con éxito.\n\n{report}"

    # === 6️⃣ Si no hay intención cognitiva → modo conversación normal ===
    user_prompt = f"Pregunta: {question}"
    if context:
        user_prompt += f"\n\nContexto:\n{context}"

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages += history[-6:]
    messages.append({"role": "user", "content": user_prompt})

    try:
        res = DEEPSEEK_CLIENT.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            max_tokens=800,
            temperature=0.7
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Error DeepSeek: {e}")
        return "No pude generar respuesta ahora."
