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
from core.errors.semantic_validator import validate_sql_semantics
from core.errors.error_types import ErrorType, ClassifiedError
from core.errors.error_manager import error_log

# === NUEVOS IMPORTS PARA INSERCIÓN SEGURA ===
from core.ai_core.structured_action_extractor import extract_structured_action
from core.ai_core.pending_actions_manager import PendingActionsManager
from core.ai_core.preview_builder import build_competencias_preview


# <<< MODIFICACIÓN >>>
# Importación correcta del nuevo manejador de errores
from core.errors.error_manager import handle_error
# <<< FIN MODIFICACIÓN >>>

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

# Para acciones pendientes de inserción
PENDING_MANAGER = PendingActionsManager(ttl_seconds=600)

# ===============================
# UTILIDAD DE EXTRACCIÓN
# ===============================
def extract_chart_type(text):
    """Extrae el tipo de gráfico SOLO si el usuario realmente lo pide."""
    text_lower = text.lower()

    # Señales claras de que está pidiendo graficar
    triggers = ["grafica", "gráfico", "grafico", "visualiza", "visualización", "haz un", "haz una"]

    if not any(t in text_lower for t in triggers):
        return None  # <- Solo activa detección si realmente pide un gráfico

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

    for term, chart_type in chart_mapping.items():
        if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
            return chart_type

    return None# Si no se encuentra un tipo específico


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
        # El usuario está pidiendo explícitamente un tipo de gráfico o cambiarlo
        session_state['last_chart_type'] = current_chart_type
        print(f"✅ Tipo de gráfico en memoria actualizado: {current_chart_type}")
    
    # Obtener el tipo de gráfico a USAR (el recordado)
    chart_type_to_use = session_state['last_chart_type']
    
    # --- PROMPT DEL SISTEMA ---
        # =======================================================
    # 🔥 NUEVO FLUJO: MANEJO DE ACCIONES PENDIENTES DE INSERCIÓN
    # =======================================================
    pending = PENDING_MANAGER.get(session_id)

    if pending:
        user_msg = question.strip().lower()

        # CONFIRMAR
        if user_msg in ["si", "sí", "yes", "confirmar", "confirmo"]:
            try:
                payload = pending
                conn = connect_db()
                if not conn:
                    PENDING_MANAGER.clear(session_id)
                    return "❌ No pude conectar a la base de datos para guardar la evaluación."

                target_table = payload["target_table"]
                usuario_id = int(payload["usuario_db_id"])
                competencias = payload["competencias"]

                if target_table == "competencia_transversales":
                    insert_table = "respuesta_competencia_transversales"
                    fk = "id_competencia_transversal"
                elif target_table == "competencia_docente":
                    insert_table = "respuesta_competencia_docente"
                    fk = "id_competencia_docente"
                else:
                    PENDING_MANAGER.clear(session_id)
                    return f"❌ Tabla objetivo no soportada: {target_table}"

                with conn.cursor() as cur:
                    sql = f"""
                        INSERT INTO {insert_table} (id_usuario, {fk}, fecha_respuesta)
                        VALUES (%s, %s, CURRENT_DATE)
                    """
                    for comp in competencias:
                        cur.execute(sql, (usuario_id, comp["id"]))
                    conn.commit()
                
                PENDING_MANAGER.clear(session_id)
                return f"✅ Se han registrado {len(competencias)} evaluaciones en {insert_table}."

            except Exception as e:
                print("❌ Error al insertar evaluación:", e)
                return "❌ Ocurrió un error al guardar la evaluación. Revisa los logs."

        # CANCELAR
        elif user_msg in ["no", "cancelar", "n"]:
            PENDING_MANAGER.clear(session_id)
            return "❌ Inserción cancelada. No se realizaron cambios."

        # Espera respuesta válida
        else:
            return "Tienes una acción pendiente. Responde **Sí** para confirmar o **No** para cancelar."

    # =======================================================
    # 🔥 NUEVO FLUJO: DETECCIÓN DE SOLICITUD ESTRUCTURADA DE INSERCIÓN
    # =======================================================
    structured = extract_structured_action(question)

    if structured:
        target_table = structured["target_table"]
        usuario_identifier = structured["usuario_identificador"]
        iluo_level = structured["iluo"]

        conn = connect_db()
        if not conn:
            return "❌ No pude conectar a la base de datos."

        try:
            with conn.cursor() as cur:

                # ---- Buscar usuario ----
                if "@" in usuario_identifier:
                    cur.execute("SELECT id, nombre, correo FROM usuario WHERE correo ILIKE %s LIMIT 1;",
                                (f"%{usuario_identifier}%",))
                    u = cur.fetchone()
                elif usuario_identifier.isdigit():
                    cur.execute("SELECT id, nombre, correo FROM usuario WHERE id = %s;",
                                (int(usuario_identifier),))
                    u = cur.fetchone()
                else:
                    cur.execute("SELECT id, nombre, correo FROM usuario WHERE nombre ILIKE %s LIMIT 5;",
                                (f"%{usuario_identifier}%",))
                    usuarios = cur.fetchall()

                    if len(usuarios) == 0:
                        return f"❌ No encontré ningún usuario llamado '{usuario_identifier}'."

                    if len(usuarios) > 1:
                        listado = "\n".join([f"- ID {r[0]} | {r[1]} | {r[2]}" for r in usuarios])
                        return (
                            "Encontré varias coincidencias, especifica con correo o ID:\n" +
                            listado
                        )

                    u = usuarios[0]

                if not u:
                    return f"❌ No se encontró el usuario '{usuario_identifier}'."

                usuario_info = {
                    "id": u[0],
                    "nombre": u[1],
                    "correo": u[2]
                }

                # ---- Buscar competencias por ILUO ----
                sql_comp = f"""
                    SELECT c.id,
                        p.texto AS pregunta_nombre,
                        r.texto AS respuesta_nombre,
                        c.id_iluo
                    FROM {target_table} c
                    INNER JOIN pregunta p ON p.id = c.id_pregunta
                    INNER JOIN respuesta r ON r.id = c.id_respuesta
                    WHERE c.id_iluo = %s
                    ORDER BY c.id;
                """
                cur.execute(sql_comp, (iluo_level,))
                rows = cur.fetchall()

                competencias = [
                    {
                        "id": r[0],                        # Oculto en previsualización, pero necesario para insertar
                        "pregunta": r[1],
                        "respuesta": r[2],
                        "id_iluo": r[3]
                    } for r in rows
                ]

                # ---- Crear previsualización ----
                preview = build_competencias_preview(usuario_info, competencias, target_table, iluo_level)

                # ---- Guardar acción pendiente ----
                PENDING_MANAGER.save(session_id, {
                    "target_table": target_table,
                    "usuario_db_id": usuario_info["id"],
                    "usuario_info": usuario_info,
                    "iluo": iluo_level,
                    "competencias": competencias
                })

                return preview

        except Exception as e:
            print("❌ Error en inserción estructurada:", e)
            return "❌ Hubo un error procesando tu solicitud."
        finally:
            conn.close()

    
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
    if tipo in ["evaluar", "generar_informe", "consultar", "guardar_resultado"]:
        print(f"🧠 Modo AIGR activo (intención: {tipo})")

        # 🔹 Cargar esquemas correctamente
        schema_semantic, schema_embeddings = schema_loader()
        
        # 3.0b Recuperar el plan anterior de la BD (si existe)
        last_plan = get_last_action_plan(session_id) 

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
        
        # 🔍 VALIDACIÓN SEMÁNTICA PREVIA
        validation = validate_sql_semantics(sql, schema_semantic)  # <<< CAMBIO AQUÍ
        if not validation.is_valid:
            print(f"❌ Validación semántica falló: {validation.error_message}")
            
            # Construir mensaje amigable
            user_message = f"⚠️ {validation.error_message}"
            if validation.suggestion:
                user_message += f"\n\n💡 **Sugerencia**: {validation.suggestion}"
            
            return user_message
        
        print(f"✅ Validación semántica pasó. Ejecutando query...")
        try:
            connection_url = URL.create(
                drivername="postgresql+psycopg2",
                username="postgres",
                password="pgadmin4",
                host="localhost",
                port=5432,
                database="bdgestion_octubre",
                query={"client_encoding": "UTF8"} 
            )

            engine = create_engine(connection_url, connect_args={"options": "-c client_encoding=WIN1252"})
            with engine.connect() as connection:
                df = pd.read_sql_query(text(sql), connection)
            
            wants_chart = False

            plan_vis_type = plan.get("meta", {}).get("visualization")
            final_chart_type = chart_type_to_use
            
            if not final_chart_type: 
                final_chart_type = plan_vis_type 
                
            if not final_chart_type: 
                final_chart_type = 'bar'
            if df.empty:

                
                empty_error = ClassifiedError(
                    type=ErrorType.EMPTY_DATASET,
                    technical_message="Query returned empty DataFrame",
                    original_exception=ValueError("Empty dataset")
                )
                error_log.add(empty_error)
                
                return "📊 La consulta se ejecutó correctamente, pero no se encontraron datos que coincidan con los criterios. Intenta ajustar los filtros o parámetros de búsqueda."
            
            # Validar datos insuficientes para gráficos
            if wants_chart:

                # Si no hay tipo explícito de gráfico, usar el último recordado
                if not final_chart_type:
                    final_chart_type = session_state.get("last_chart_type")

                # Si aún así no hay tipo, NO graficamos
                if not final_chart_type:
                    wants_chart = False
                else:
                    # Validar datos insuficientes
                    if len(df) == 1:
                        return "📊 Se encontró solo 1 registro. Los gráficos requieren al menos 2 puntos de datos."

                    # Validar columnas numéricas
                    if final_chart_type in ['line', 'scatter'] and len(df.select_dtypes(include=['number']).columns) == 0:
                        return f"📊 No se puede crear un gráfico de {final_chart_type} porque no hay columnas numéricas."
            
            # Validar columnas numéricas para gráficos que las requieren
            if final_chart_type in ['line', 'scatter'] and len(df.select_dtypes(include=['int64', 'float64', 'number']).columns) == 0:
                from core.errors.error_types import ErrorType, ClassifiedError
                from core.errors.error_manager import error_log
                
                invalid_chart = ClassifiedError(
                    type=ErrorType.INVALID_CHART_DATA,
                    technical_message=f"No numeric columns for {final_chart_type} chart",
                    original_exception=ValueError("Invalid chart data")
                )
                error_log.add(invalid_chart)
                
                return f"📊 No se puede crear un gráfico de {final_chart_type} porque los datos no contienen columnas numéricas. ¿Quieres un reporte en texto o una tabla en su lugar?"
        except Exception as e:
            print(f"❌ Error en la ejecución de SQL: {e}") # Log técnico para la consola
            user_message = handle_error(e) # Llama al nuevo manejador
            return user_message # Devuelve el mensaje amigable

        # 3.4 Generar informe o acción autónoma
        if plan["accion"] == "guardar_resultado":
            # 3.5 PERSISTIR EL NUEVO PLAN DE ACCIÓN Y EL SQL EJECUTADO
            store_action_plan(session_id, plan, executed_sql=sql)
            return "✅ Los datos fueron procesados y almacenados correctamente."
        
        # 3.5 PERSISTIR EL NUEVO PLAN DE ACCIÓN Y EL SQL EJECUTADO
        store_action_plan(session_id, plan, executed_sql=sql) 
        
        # 3.6 Decidir si graficar o sintetizar
        
        # Obtener la sugerencia de visualización del plan (puede ser 'line')
        plan_vis_type = plan.get("meta", {}).get("visualization")
        
       
        final_chart_type = chart_type_to_use
            
        if not final_chart_type:
            final_chart_type = None

        # Si el plan sugirió una visualización, o hay una en memoria, o se preguntó explícitamente por un gráfico
        wants_chart = (
            current_chart_type is not None or
            "grafica" in question.lower() or
            "gráfico" in question.lower() or
            "grafico" in question.lower()
        )

        if wants_chart:
            
            # Actualizamos la memoria local con el tipo de gráfico final
            if final_chart_type != session_state['last_chart_type']:
                session_state['last_chart_type'] = final_chart_type
                # Usamos un mensaje de debug más claro para ver qué se decidió
                print(f"✅ Tipo de gráfico final seleccionado: {final_chart_type}")


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