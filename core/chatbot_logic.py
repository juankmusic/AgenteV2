# core/chat_logic.py

# ===============================
# IMPORTACIONES DEL SISTEMA
# ===============================
# Estas librerías son herramientas que necesitamos para que el programa funcione
import os                                  # Para manejar archivos y carpetas del sistema operativo
import sys                                 # Para controlar el programa (como cerrarlo si falta algo importante)
import uuid                                # Para crear identificadores únicos (como números de serie)
import re                                  # Para buscar patrones en textos (como encontrar palabras específicas)
import pandas as pd                        # Para trabajar con tablas de datos (como Excel pero en Python)
import psycopg2                            # Para conectarse a la base de datos PostgreSQL
import pypdf                               # Para leer archivos PDF
from dotenv import load_dotenv             # Para cargar configuraciones secretas desde un archivo .env
from openai import OpenAI                  # Para usar inteligencia artificial de OpenAI
from db.connection import connect_db       # Nuestra función para conectar con la base de datos
from sqlalchemy import create_engine, text # Herramientas para hablar con bases de datos
from sqlalchemy.engine import URL          # Para crear la dirección de conexión a la base de datos

# ===============================
# IMPORTACIONES DEL NÚCLEO DE IA (nuestros módulos personalizados)
# ===============================
# Estos son los "cerebros" del agente inteligente que creamos nosotros

from core.ai_core.nlp_embeddings import search_similar_embeddings  # Busca textos parecidos
from core.ai_core.nlp_embeddings import extract_entities           # Saca información importante del texto
from core.ai_core.nlp_embeddings import get_last_action_plan, store_action_plan  # Lee y guarda planes de acción
from core.ai_core.nlp_embeddings import analyze_text               # Analiza el significado de un texto
from core.ai_core.intention_analyzer import detect_intention       # Detecta qué quiere hacer el usuario
from core.ai_core.dynamic_planner import plan_actions              # Crea un plan de acción automático
from core.ai_core.query_generator import generate_query_from_plan  # Convierte el plan en código SQL
from core.ai_core.report_synthesizer import generate_report        # Crea reportes con los datos
from core.ai_core.schema_loader import load_schema as schema_loader  # Carga la estructura de la base de datos
from core.errors.semantic_validator import validate_sql_semantics  # Verifica que el SQL tenga sentido
from core.errors.error_types import ErrorType, ClassifiedError     # Define tipos de errores
from core.errors.error_manager import error_log                    # Guarda registro de errores
from core.errors.error_manager import handle_error  # Maneja errores de forma amigable para el usuario
from core.ai_core.structured_action_extractor import extract_structured_action  # Extrae instrucciones estructuradas
from core.ai_core.pending_actions_manager import PendingActionsManager          # Guarda acciones que esperan confirmación
from core.ai_core.preview_builder import build_competencias_preview             # Muestra vista previa de competencias
from core.ai_core.preview_builder import build_multiuser_summary_item, build_multiuser_preview  # Vistas previas múltiples


# ===============================
# CONFIGURACIÓN INICIAL
# ===============================
# Aquí cargamos las "llaves secretas" para usar los servicios de IA

load_dotenv() # Lee el archivo .env que tiene las contraseñas

# Obtenemos las claves API (son como contraseñas para usar los servicios)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # Clave para el modelo DeepSeek
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")      # Clave para OpenAI (GPT)

# Si falta alguna clave, el programa se detiene porque no puede funcionar
if not DEEPSEEK_API_KEY or not OPENAI_API_KEY:
    print("❌ Falta API Key en .env")  # Mensaje de error
    sys.exit(1)  # Cierra el programa con código de error

# Creamos los "clientes" que nos permiten hablar con los servicios de IA
OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY)
DEEPSEEK_CLIENT = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")

# ===============================
# MEMORIA DE SESIÓN GLOBAL 
# ===============================
# Esto funciona como la "memoria a corto plazo" del agente
# Guarda información importante mientras dura una conversación

# Diccionario que almacena el estado de cada conversación (sesión)
# Estructura: {'id_de_sesion': {'last_chart_type': 'bar' | 'pie' | 'line' | None, ...}}
# Ejemplo: {'abc123': {'last_chart_type': 'bar'}} significa que en la sesión abc123
# el usuario pidió un gráfico de barras
SESSION_STATES = {}

# Manejador de acciones pendientes (cuando el usuario debe confirmar algo)
# ttl_seconds=600 significa que la acción expira después de 10 minutos (600 segundos)
PENDING_MANAGER = PendingActionsManager(ttl_seconds=600)

# ===============================
# FUNCIÓN: Extraer tipo de gráfico
# ===============================
def extract_chart_type(text):
    """Extrae el tipo de gráfico SOLO si el usuario realmente lo pide."""

    # Convertimos todo a minúsculas para facilitar la búsqueda
    text_lower = text.lower()

    # Palabras que indican que el usuario REALMENTE quiere un gráfico
    triggers = ["grafica", "gráfico", "grafico", "visualiza", "visualización", "haz un", "haz una"]

     # Si no encuentra ninguna palabra clave, NO es una petición de gráfico
    if not any(t in text_lower for t in triggers):
        return None  # <- Solo activa detección si realmente pide un gráfico

    # Diccionario que traduce las palabras del usuario al tipo técnico de gráfico
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

    # Buscamos cada término en el texto del usuario
    for term, chart_type in chart_mapping.items():
        # Usamos regex para buscar la palabra completa (no parte de otra palabra)
        if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
            return chart_type  # Devolvemos el tipo de gráfico encontrado

    return None #Si no se encuentra un tipo específico


# ===============================
# CHAT & SESIONES
# ===============================
def save_message(session_id, role, msg):
    """Guarda un mensaje en la base de datos para tener historial de conversaciones."""

    # Conectamos con la base de datos
    conn = connect_db()
    if not conn:  # Si no se pudo conectar
        return    # Salimos de la función
    try:
        # Usamos un cursor (como un puntero) para ejecutar comandos SQL
        with conn.cursor() as cur:
            # Si el role es "assistant" lo cambiamos a "bot" para la base de datos
            db_role = "bot" if role == "assistant" else role
            
            # Insertamos el mensaje en la tabla chatbot_logs
            cur.execute(
                "INSERT INTO chatbot_logs (session_id, role, message) VALUES (%s,%s,%s)", 
                (session_id, db_role, msg)
            )
            conn.commit()  # Guardamos los cambios permanentemente
    finally:
        conn.close()  # Siempre cerramos la conexión al terminar


def get_history(session_id):
    """Recupera el historial de mensajes de una sesión específica."""
    conn = connect_db()
    if not conn:
        return [] # Si no se pudo conectar, devolvemos lista vacía
    try:
        with conn.cursor() as cur:
            # Seleccionamos los mensajes de la sesión ordenados por fecha
            cur.execute("SELECT role, message FROM chatbot_logs WHERE session_id=%s ORDER BY timestamp", (session_id,))
            return [{"role": "assistant" if r[0]=="bot" else r[0], "content": r[1]} for r in cur.fetchall()]
    finally:
        conn.close()


def list_sessions():
    """Lista las sesiones previas con su cantidad de mensajes."""
    conn = connect_db()
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            # Agrupamos por session_id y contamos mensajes
            # Ordenamos por el mensaje más reciente primero
            cur.execute("SELECT session_id, COUNT(*) FROM chatbot_logs GROUP BY session_id ORDER BY MAX(timestamp) DESC")
            return cur.fetchall()
    finally:
        conn.close()


def select_or_create_session():
    """Permite al usuario elegir una sesión previa o crear una nueva."""
    # Listamos las sesiones existentes
    sessions = list_sessions()
    if sessions: # Si hay sesiones previas
        print("\n📂 Sesiones previas:")
        # Mostramos cada sesión con su número de mensajes
        for i, (sid, count) in enumerate(sessions):
            print(f"{i+1}. {sid} ({count} mensajes)")
        # Opción para nueva sesión
        print(f"{len(sessions)+1}. Nueva sesión")
        ch = input("Elige: ")
        if ch.isdigit() and 1 <= int(ch) <= len(sessions):
            return sessions[int(ch)-1][0]
    return str(uuid.uuid4())


# ===============================
# MODELO DE CHAT
# ===============================

def deepseek_chat(session_id, question, context=None, history=None): 
    """Función principal de chat que maneja la lógica del agente inteligente."""

    global SESSION_STATES # Usamos la memoria global de sesiones
    
   # ===============================
    # 1. INICIALIZAR MEMORIA DE SESIÓN
    # ===============================

    # Si es la primera vez que se usa esta sesión, creamos su espacio en memoria
    if session_id not in SESSION_STATES:
        SESSION_STATES[session_id] = {'last_chart_type': None}

    # Obtenemos el estado actual de esta sesión    
    session_state = SESSION_STATES[session_id]
    
    # ===============================
    # 1.1  DETECTAR TIPO DE GRÁFICO
    # ===============================
    # Intentamos saber si el usuario está pidiendo un gráfico específico
    current_chart_type = extract_chart_type(question)
    
    # ===============================
    # 1.2  ACTUALIZAR MEMORIA DE GRÁFICOS
    # ===============================
    # Si el usuario pidió un tipo de gráfico, lo guardamos en memoria
    if current_chart_type:
        # El usuario está pidiendo explícitamente un tipo de gráfico o cambiarlo
        session_state['last_chart_type'] = current_chart_type
        print(f"✅ Tipo de gráfico en memoria actualizado: {current_chart_type}")
    
    # Obtenemos el tipo de gráfico que vamos a usar (el último que pidió)
    chart_type_to_use = session_state['last_chart_type']
    
    # ===============================
    # 2. VERIFICAR SI HAY ACCIONES PENDIENTES DE CONFIRMACION
    # ===============================
    # Cuando el usuario pide insertar datos, primero mostramos una vista previa
    # y esperamos que confirme con "si" o cancele con "no"
    
    # Buscamos si hay alguna accion pendiente para esta sesion
    pending = PENDING_MANAGER.get(session_id)

    if pending:
        # Hay una accion esperando confirmacion del usuario
        
        # Limpiamos y convertimos a minusculas la respuesta del usuario
        user_msg = question.strip().lower()

        # ===============================
        # 2.1 CASO: USUARIO CANCELA LA INSERCION
        # ===============================
        if user_msg in ["no", "cancelar", "n"]:
            PENDING_MANAGER.clear(session_id)
            return "❌ Inserción cancelada. No se realizaron cambios."

       # ===============================
        # 2.2 CASO: USUARIO CONFIRMA LA INSERCION
        # ===============================
        if user_msg in ["si", "sí", "yes", "confirmo", "confirmar"]:

            # Conectamos con la base de datos
            conn = connect_db()
            if not conn:
                # Si no podemos conectar, limpiamos la accion pendiente
                PENDING_MANAGER.clear(session_id)
                return "❌ No pude conectar a la base de datos para guardar la evaluación."

            try:
                #creamos un cursor para ejecutar comandos SQL
                with conn.cursor() as cur:

                   # ===============================
                    # CASO A: INSERCION PARA UN SOLO USUARIO
                    # ===============================
                    if pending.get("is_multi") is False:

                        # Extraemos la informacion de la accion pendiente
                        target_table = pending["target_table"]  # Tabla de origen (competencia_transversales o competencia_docente)
                        usuario_id = pending["usuario_db_id"]   # ID del usuario en la base de datos
                        competencias = pending["competencias"]   # Lista de competencias a insertar

                       # Determinamos en que tabla vamos a insertar
                        if target_table == "competencia_transversales":
                            insert_table = "respuesta_competencia_transversales"
                            fk = "id_competencia_transversal"  # Nombre de la columna que relaciona
                        else:
                            insert_table = "respuesta_competencia_docente"
                            fk = "id_competencia_docente"  # Nombre de la columna que relaciona

                        # Preparamos la consulta SQL para insertar
                        # %s son marcadores de posicion que se reemplazan con valores reales
                        sql = f"""
                            INSERT INTO {insert_table} (id_usuario, {fk}, fecha_respuesta)
                            VALUES (%s, %s, CURRENT_DATE)
                        """
                        # Insertamos cada competencia una por una
                        for comp in competencias:
                            cur.execute(sql, (usuario_id, comp["id"]))

                        # Guardamos todos los cambios en la base de datos
                        conn.commit()
                        # Limpiamos la accion pendiente
                        PENDING_MANAGER.clear(session_id)

                        # Devolvemos mensaje de exito
                        return (
                            f"✅ Se registraron {len(competencias)} "
                            f"competencias para el usuario **{pending['usuario_info']['nombre']}**."
                        )

                    # ===============================
                    # CASO B: INSERCION PARA MULTIPLES USUARIOS
                    # ===============================
                    if pending.get("is_multi") is True:
                        
                        # Extraemos informacion similar al caso anterior    
                        target_table = pending["target_table"]

                        # Determinamos la tabla de insercion
                        if target_table == "competencia_transversales":
                            insert_table = "respuesta_competencia_transversales"
                            fk = "id_competencia_transversal"  # Nombre de la columna que relaciona
                        else:
                            insert_table = "respuesta_competencia_docente"
                            fk = "id_competencia_docente"  # Nombre de la columna que relaciona
                        # Preparamos la consulta SQL de insercion
                        sql = f"""
                            INSERT INTO {insert_table} (id_usuario, {fk}, fecha_respuesta)
                            VALUES (%s, %s, CURRENT_DATE)
                        """
                        # Variables para llevar el conteo
                        total_insertadas = 0
                        resumen = "<h3>✅ Inserciones completadas:</h3><ul>"

                        # Recorremos cada usuario en la lista
                        for item in pending["usuarios"]:

                            # Extraemos datos de cada usuario
                            u_id = item["usuario_db_id"]
                            u_nombre = item["usuario_info"]["nombre"]
                            comps = item["competencias"]

                            # Insertamos las competencias de este usuario
                            for comp in comps:
                                cur.execute(sql, (u_id, comp["id"]))
                                total_insertadas += 1

                            # Agregamos al resumen HTML
                            resumen += f"<li><strong>{u_nombre}</strong>: {len(comps)} competencias</li>"

                        resumen += "</ul>"

                        # Guardamos todos los cambios
                        conn.commit()
                        # Limpiamos la accion pendiente
                        PENDING_MANAGER.clear(session_id)

                        return (
                            resumen +
                            f"<p><strong>Total de competencias registradas:</strong> {total_insertadas}</p>"
                        )

                # Si todo sale bien
            except Exception as e:
                # Capturamos cualquier error durante la inserción
                print(" Error al insertar evaluación:", e)
                return " Ocurrió un error al guardar la evaluación."

       
        # ===============================
        # 2.3 CASO: RESPUESTA INVALIDA
        # ===============================
        # Si el usuario no respondio "si" ni "no", le recordamos que debe elegir
        return "Tienes una acción pendiente. Responde **Sí** para confirmar o **No** para cancelar."
    # ===============================
    # 3. DETECTAR SI EL USUARIO QUIERE INSERTAR DATOS
    # ===============================
    # Esta seccion analiza si el mensaje del usuario es una orden para guardar evaluaciones
    # Ejemplos: "evalua a juan con iluo 3", "registra a maria y pedro con iluo 2"
    
    # Intentamos extraer una accion estructurada del mensaje
    structured = extract_structured_action(question)

   # ===============================
    # 3.1 VALIDACION: VERIFICAR QUE LA INSERCION SEA VALIDA
    # ===============================
    # No queremos activar el modo insercion si no hay usuarios reales mencionados
    valid_insert = False

    if structured:
        # Si la accion es insertar evaluacion de UN usuario
        if structured.get("action") == "insert_evaluation":
            # Verificamos que tenga identificador de usuario Y nivel ILUO
            if structured.get("usuario_identificador") and structured.get("iluo") is not None:
                valid_insert = True

        # Si la accion es insertar evaluaciones de MULTIPLES usuarios
        elif structured.get("action") == "insert_evaluation_multi":
            usuarios = structured.get("usuarios", [])
            # válido solo si existe al menos un usuario con ILUO real
            if any(u.get("iluo") is not None for u in usuarios):
                valid_insert = True

     # Si NO es una insercion valida, eliminamos la estructura
    if not valid_insert:
        structured = None

    # ===============================
    # 4. PROCESAR INSERCION ESTRUCTURADA VALIDA
    # ===============================
    if structured:

        # ===============================
        # CASO A: INSERCION DE UN SOLO USUARIO
        # ===============================
        if structured["action"] == "insert_evaluation":
            # Extraemos los datos de la estructura
            target_table = structured["target_table"]                 # competencia_transversales o competencia_docente
            usuario_identifier = structured["usuario_identificador"]  # Puede ser nombre, correo o ID
            iluo_level = structured["iluo"]                           # Nivel ILUO (1, 2, 3 o 4)

            # Conectamos con la base de datos
            conn = connect_db()
            if not conn:
                return "❌ No pude conectar a la base de datos."

            try:
                with conn.cursor() as cur:

                    # ===============================
                    # 4.1 BUSCAR AL USUARIO EN LA BASE DE DATOS
                    # ===============================

                    # Si el identificador contiene @, es un correo electronico
                    if "@" in usuario_identifier:
                        cur.execute("SELECT id, nombre, correo FROM usuario WHERE correo ILIKE %s LIMIT 1;",
                                    (f"%{usuario_identifier}%",))
                        u = cur.fetchone()

                    # Si el identificador es solo numeros, es un ID
                    elif usuario_identifier.isdigit():
                        cur.execute("SELECT id, nombre, correo FROM usuario WHERE id = %s;",
                                    (int(usuario_identifier),))
                        u = cur.fetchone()

                    # Si no, asumimos que es un nombre
                    else:
                        cur.execute("SELECT id, nombre, correo FROM usuario WHERE nombre ILIKE %s LIMIT 5;",
                                    (f"%{usuario_identifier}%",))
                        usuarios = cur.fetchall()

                        # Si no encontramos ningun usuario    
                        if len(usuarios) == 0:
                            return f"❌ No encontré ningún usuario llamado '{usuario_identifier}'."

                        # Si encontramos varios usuarios con ese nombre
                        if len(usuarios) > 1:
                            # Listar usuarios encontrados
                            listado = "\n".join([f"- ID {r[0]} | {r[1]} | {r[2]}" for r in usuarios])
                            return (
                                "Encontré varias coincidencias, especifica con correo o ID:\n" +
                                listado
                            )
                        u = usuarios[0]
                    # Si no encontramos al usuario, informamos
                    if not u:
                        return f"❌ No se encontró el usuario '{usuario_identifier}'."
                    
                    # Guardamos la informacion del usuario encontrado
                    usuario_info = {"id": u[0], "nombre": u[1], "correo": u[2]}

                    # ===============================
                    # 4.2 BUSCAR LAS COMPETENCIAS SEGUN EL NIVEL ILUO
                    # ===============================
                    # Buscamos todas las competencias que corresponden al nivel ILUO especificado
                    
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

                    # Convertimos los resultados en una lista de diccionarios
                    competencias = [
                        {
                            "id": r[0],           # ID de la competencia
                            "pregunta": r[1],     # Texto de la pregunta
                            "respuesta": r[2],    # Texto de la respuesta
                            "id_iluo": r[3]       # Nivel ILUO
                        } for r in rows
                    ]

                    # ===============================
                    # 4.3 GENERAR VISTA PREVIA
                    # ===============================
                    # Creamos un HTML con la vista previa de lo que se va a insertar
                    preview = build_competencias_preview(usuario_info, competencias, target_table, iluo_level)

                    # ===============================
                    # 4.4 GUARDAR COMO ACCION PENDIENTE
                    # ===============================
                    # No insertamos todavia, esperamos confirmacion del usuario
                    PENDING_MANAGER.save(session_id, {
                        "is_multi": False,              # Es un solo usuario
                        "target_table": target_table,   # Tabla de origen
                        "usuario_db_id": usuario_info["id"],  # ID del usuario
                        "usuario_info": usuario_info,   # Informacion completa del usuario
                        "iluo": iluo_level,             # Nivel ILUO
                        "competencias": competencias    # Lista de competencias
                    })
                    # Devolvemos la vista previa al usuario   
                    return preview

            except Exception as e:
                print("❌ Error en inserción estructurada:", e)
                return "❌ Hubo un error procesando tu solicitud."
            finally:
                conn.close()

        # ===============================
        # CASO B: INSERCION PARA MULTIPLES USUARIOS
        # ===============================
        if structured["action"] == "insert_evaluation_multi":

            # Extraemos los datos de la accion estructurada
            target_table = structured["target_table"]  # competencia_transversales o competencia_docente
            usuarios_list = structured["usuarios"]     # Lista de usuarios con sus respectivos ILUO
           
            # Conectamos con la base de datos
            conn = connect_db()
            if not conn:
                return "❌ No pude conectar a la base de datos."

            # Listas para almacenar los resultados
            usuarios_preview = []  # Para la vista previa HTML
            pending_payload = {    # Para guardar como accion pendiente
                "is_multi": True,
                "target_table": target_table,
                "usuarios": []
            }

            try:
                with conn.cursor() as cur:
                    # Recorremos cada usuario en la lista
                    for uitem in usuarios_list:

                        # Extraemos datos de cada usuario
                        ident = uitem["identificador"]  # Nombre, correo o ID
                        iluo_level = uitem["iluo"]      # Nivel ILUO para este usuario

                        # ===============================
                        # 4.5 RESOLVER IDENTIDAD DE CADA USUARIO
                        # ===============================
                        # Mismo proceso que en el caso A, pero para cada usuario
                        
                        # Si tiene @, es un correo
                        if "@" in ident:
                            cur.execute("SELECT id, nombre, correo FROM usuario WHERE correo ILIKE %s LIMIT 1;",
                                        (f"%{ident}%",))
                            u = cur.fetchone()
                        #Si es solo un numero, es un ID
                        elif ident.isdigit():
                            cur.execute("SELECT id, nombre, correo FROM usuario WHERE id = %s;",
                                        (int(ident),))
                            u = cur.fetchone()

                        #Si no, es un nombre
                        else:
                            cur.execute("SELECT id, nombre, correo FROM usuario WHERE nombre ILIKE %s LIMIT 5;",
                                        (f"%{ident}%",))
                            matches = cur.fetchall()
                            # Si no encontramos usuarios, saltamos este
                            if len(matches) == 0:
                                continue
                            # Si hay varios con el mismo nombre, saltamos (es ambiguo)
                            if len(matches) > 1:
                                continue
                            u = matches[0]

                        # Si no encontramos al usuario, saltamos al siguient
                        if not u:
                            continue  # seguridad

                        # Guardamos informacion del usuario encontrado
                        usuario_info = {"id": u[0], "nombre": u[1], "correo": u[2]}

                        # ===============================
                        # 4.6 BUSCAR COMPETENCIAS POR ILUO PARA ESTE USUARIO
                        # ===============================
                        sql_comp = f"""
                            SELECT c.id, p.texto, r.texto, c.id_iluo
                            FROM {target_table} c
                            INNER JOIN pregunta p ON p.id = c.id_pregunta
                            INNER JOIN respuesta r ON r.id = c.id_respuesta
                            WHERE c.id_iluo = %s
                            ORDER BY c.id;
                        """
                        cur.execute(sql_comp, (iluo_level,))
                        rows = cur.fetchall()

                        # Convertimos las competencias en diccionarios
                        competencias = [
                            {"id": r[0], "pregunta": r[1], "respuesta": r[2], "id_iluo": r[3]}
                            for r in rows
                        ]

                        # ===============================
                        # 4.7 AGREGAR A LISTA DE VISTA PREVIA
                        # ===============================
                        usuarios_preview.append({
                            "usuario_info": usuario_info,
                            "competencias": competencias,
                            "iluo": iluo_level
                        })

                        # ===============================
                        # 4.8 AGREGAR A ACCION PENDIENTE
                        # ===============================
                        pending_payload["usuarios"].append({
                            "usuario_db_id": usuario_info["id"],
                            "usuario_info": usuario_info,
                            "iluo": iluo_level,
                            "competencias": competencias
                        })

                # ===============================
                # 4.9 GENERAR VISTA PREVIA MULTIUSUARIO
                # ===============================
                # Creamos un HTML con la vista previa de todos los usuarios
                previews_html = build_multiuser_preview(usuarios_preview, target_table)

                # ===============================
                # 4.10 GUARDAR COMO ACCION PENDIENTE
                # ===============================
                # Guardamos toda la informacion para cuando el usuario confirme
                PENDING_MANAGER.save(session_id, pending_payload)

                # Devolvemos la vista previa al usuario
                return previews_html

            except Exception as e:
                print("❌ Error multiusuario:", e)
                return "❌ Hubo un error procesando la solicitud múltiple."
            finally:
                conn.close()

    
    # ===============================
    # 5. DEFINIR EL PROMPT DEL SISTEMA
    # ===============================
    # Este es el "manual de instrucciones" que le damos a la IA
    # Le decimos quien es, que puede hacer y como debe comportarse
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
    
    # ===============================
    # 6. ANALISIS SEMANTICO PROFUNDO
    # ===============================
    # Analizamos el significado profundo del texto del usuario
    # Extraemos: entidades (nombres, lugares), verbos, sustantivos, etc.
    semantic_data = analyze_text(question)

    # ===============================
    # 7. DETECCIÓN DE INTENCIÓN COGNITIVA
    # ===============================
    # Intentamos entender QUE quiere hacer el usuario
    # Tipos posibles: "evaluar", "generar_informe", "consultar", "guardar_resultado", "conversacion"
    intent_data = detect_intention(semantic_data["texto"])
    tipo = intent_data.get("tipo", "conversacion")

    # ===============================
    # 7.1 COMBINAR ENTIDADES
    # ===============================
    # Si el detector de intencion no extrajo entidades, usamos las del analisis semantico
    if not intent_data.get("entidades"):
        intent_data["entidades"] = semantic_data.get("entidades", {})

    # ===============================
    # 8. ACTIVAR MODO AIGR (AGENTE INTELIGENTE)
    # ===============================
    # Si el usuario quiere hacer algo tecnico (consultas, reportes, evaluaciones)
    # activamos el modo de agente inteligente

    if tipo in ["evaluar", "generar_informe", "consultar", "guardar_resultado"]:
        print(f"🧠 Modo AIGR activo (intención: {tipo})")

        # ===============================
        # 8.1 CARGAR ESQUEMAS DE LA BASE DE DATOS
        # ===============================
        # Necesitamos saber que tablas, columnas y relaciones existen en la BD
        schema_semantic, schema_embeddings = schema_loader()
        
        # ===============================
        # 8.2 RECUPERAR EL PLAN ANTERIOR (si existe)
        # ===============================
        # Si el usuario ya hizo una consulta similar antes, podemos reutilizarla
        last_plan = get_last_action_plan(session_id) 

        # ===============================
        # 8.3 PLANIFICACION AUTONOMA
        # ===============================
        # El planificador decide:
        # - Que tablas consultar
        # - Que columnas necesitamos
        # - Que filtros aplicar
        # - Si debe reutilizar una consulta anterior
        plan = plan_actions(intent_data, schema_semantic, schema_embeddings, last_plan)
        
        # ===============================
        # 8.4 GENERAR CONSULTA SQL DINAMICA
        # ===============================
        # Convertimos el plan en una consulta SQL real
        sql_result = generate_query_from_plan(plan)
        sql = sql_result.get("sql")

        # Si no se pudo generar SQL, informamos al usuario
        if not sql:
            return "No pude generar una consulta válida basada en tu solicitud."

        print(f"📜 SQL generado:\n{sql}")

        # ===============================
        # 8.5 PREPARAR PARA EJECUTAR SQL
        # ===============================
        # Creamos un DataFrame vacio (como una tabla de Excel)
        df = pd.DataFrame()
        
        # ===============================
        # 8.6 VALIDACION SEMANTICA PREVIA
        # ===============================
        # ANTES de ejecutar el SQL, verificamos que tenga sentido
        # Por ejemplo: que las tablas existan, que las columnas sean correctas, etc.
        validation = validate_sql_semantics(sql, schema_semantic)
        if not validation.is_valid:
            # Si la validacion falla, NO ejecutamos el SQL
            print(f"❌ Validación semántica falló: {validation.error_message}")
            
            # Construimos un mensaje amigable para el usuario
            user_message = f"⚠️ {validation.error_message}"
            if validation.suggestion:
                user_message += f"\n\n💡 **Sugerencia**: {validation.suggestion}"
            
            return user_message
        
        print(f"✅ Validación semántica pasó. Ejecutando query...")

        # ===============================
        # 8.7 EJECUTAR LA CONSULTA SQL
        # ===============================
        try:
            # Creamos la URL de conexion a la base de datos
            connection_url = URL.create(
                drivername="postgresql+psycopg2",  # Tipo de base de datos
                username="postgres",                # Usuario
                password="postgre",                 # Contrasena
                host="localhost",                   # Direccion del servidor
                port=5432,                          # Puerto
                database="bdgestion_agente",        # Nombre de la base de datos
                query={"client_encoding": "UTF8"}   # Codificacion de caracteres
            )

            # Creamos el motor de conexion
            engine = create_engine(connection_url, connect_args={"options": "-c client_encoding=WIN1252"})

            # Ejecutamos la consulta y cargamos el resultado en un DataFrame
            with engine.connect() as connection:
                df = pd.read_sql_query(text(sql), connection)
            
            # Variable para saber si el usuario quiere un grafico
            wants_chart = False

            # Obtenemos el tipo de visualizacion sugerido por el plan
            plan_vis_type = plan.get("meta", {}).get("visualization")
            final_chart_type = chart_type_to_use
            
            # Si no hay tipo de grafico definido, usamos el del plan
            if not final_chart_type: 
                final_chart_type = plan_vis_type 
                
            # Si aun no hay tipo, usamos barras por defecto
            if not final_chart_type: 
                final_chart_type = 'bar'

            # ===============================
            # 8.8 VALIDAR QUE HAYA DATOS
            # ===============================
            if df.empty:
                # Si la consulta no devolvio ningun resultado

                # Registramos el error en el log
                empty_error = ClassifiedError(
                    type=ErrorType.EMPTY_DATASET,
                    technical_message="Query returned empty DataFrame",
                    original_exception=ValueError("Empty dataset")
                )
                error_log.add(empty_error)
                
                return "📊 La consulta se ejecutó correctamente, pero no se encontraron datos que coincidan con los criterios. Intenta ajustar los filtros o parámetros de búsqueda."
            
            # ===============================
            # 8.9 VALIDAR DATOS SUFICIENTES PARA GRAFICOS
            # ===============================
            if wants_chart:

                # Si no hay tipo explícito de gráfico, usar el último recordado
                if not final_chart_type:
                    final_chart_type = session_state.get("last_chart_type")

                # Si aún así no hay tipo, NO graficamos
                if not final_chart_type:
                    wants_chart = False
                else:
                    # ===============================
                    # VALIDACION 1: Minimo 2 puntos de datos
                    # ===============================
                    if len(df) == 1:
                        return "📊 Se encontró solo 1 registro. Los gráficos requieren al menos 2 puntos de datos."

                   # ===============================
                    # VALIDACION 2: Graficos de lineas y dispersion necesitan columnas numericas
                    # ===============================
                    if final_chart_type in ['line', 'scatter'] and len(df.select_dtypes(include=['number']).columns) == 0:
                        return f"📊 No se puede crear un gráfico de {final_chart_type} porque no hay columnas numéricas."
            
            # ===============================
            # 8.10 VALIDACION ADICIONAL DE COLUMNAS NUMERICAS
            # ===============================
            # Verificacion final para graficos que requieren datos numericos
            if final_chart_type in ['line', 'scatter'] and len(df.select_dtypes(include=['int64', 'float64', 'number']).columns) == 0:
                
                # Registramos el error
                invalid_chart = ClassifiedError(
                    type=ErrorType.INVALID_CHART_DATA,
                    technical_message=f"No numeric columns for {final_chart_type} chart",
                    original_exception=ValueError("Invalid chart data")
                )
                error_log.add(invalid_chart)
                
                return f" No se puede crear un gráfico de {final_chart_type} porque los datos no contienen columnas numéricas. ¿Quieres un reporte en texto o una tabla en su lugar?"
            
        except Exception as e:
            # ===============================
            # 8.11 MANEJO DE ERRORES EN LA EJECUCION
            # ===============================
            # Si algo sale mal al ejecutar el SQL, manejamos el error
            print(f"Error en la ejecucion de SQL: {e}")  # Log tecnico para la consola
            user_message = handle_error(e)  # Convertimos el error tecnico en mensaje amigable
            return user_message  # Devolvemos el mensaje al usuario

        # ===============================
        # 9. GENERAR INFORME O ACCION AUTONOMA
        # ===============================
        # Una vez que tenemos los datos, decidimos que hacer con ellos
        
        # ===============================
        # 9.1 CASO ESPECIAL: GUARDAR RESULTADO
        # ===============================
        if plan["accion"] == "guardar_resultado":
            # Si el plan indica que solo debemos guardar el resultado (no mostrarlo)
            
            # Guardamos el plan y el SQL ejecutado en la base de datos
            store_action_plan(session_id, plan, executed_sql=sql)
            return "✅ Los datos fueron procesados y almacenados correctamente."
        
       # ===============================
        # 9.2 PERSISTIR EL PLAN DE ACCION
        # ===============================
        # Guardamos el plan ejecutado para futuras consultas similares
        # Esto permite que el agente "recuerde" como resolvio esta peticion
        store_action_plan(session_id, plan, executed_sql=sql) 
        
        # ===============================
        # 9.3 DECIDIR SI GRAFICAR O SINTETIZAR
        # ===============================
        
        # Obtenemos la sugerencia de visualizacion del plan
        plan_vis_type = plan.get("meta", {}).get("visualization")
        
       
        final_chart_type = chart_type_to_use

        # Usamos el tipo de grafico que el usuario pidio o recordamos 
        if not final_chart_type:
            final_chart_type = None

        # ===============================
        # 9.4 DETERMINAR SI EL USUARIO QUIERE UN GRAFICO
        # ===============================
        # Verificamos si el usuario explicitamente pidio un grafico
        wants_chart = (
            current_chart_type is not None or          # Pidio un tipo especifico
            "grafica" in question.lower() or           # Uso la palabra "grafica"
            "gráfico" in question.lower() or           # Uso la palabra "grafico" con acento
            "grafico" in question.lower()              # Uso la palabra "grafico" sin acento
        )

        # ===============================
        # 9.5 GENERAR REPORTE CON GRAFICO
        # ===============================
        if wants_chart:
            
            # Actualizamos la memoria local con el tipo de gráfico final
            if final_chart_type != session_state['last_chart_type']:
                session_state['last_chart_type'] = final_chart_type
                # Usamos un mensaje de debug más claro para ver qué se decidió
                print(f"✅ Tipo de gráfico final seleccionado: {final_chart_type}")

             # Generamos el reporte incluyendo el grafico del tipo especificado
            report = generate_report(df, question, final_chart_type) 
            
            # Devolvemos el mensaje con el reporte y el grafico
            return f"🧩 Resultado de la consulta (mostrado como **gráfico de {final_chart_type}**):\n\n{report}"

        else:
            # ===============================
            # 9.6 GENERAR REPORTE SIN GRAFICO
            # ===============================
            # Si no se pidio grafico, generamos solo texto o tabla
            report = generate_report(df, question)

        # Devolvemos el reporte final
        return f" Resultado de la consulta:\n\n{report}"

     # ===============================
    # 10. MODO CONVERSACION (si no hay intencion cognitiva)
    # ===============================
    # Si el usuario solo quiere charlar o hacer preguntas generales
    # (no es una consulta tecnica ni una peticion de datos)
    
    # ===============================
    # 10.1 BUSCAR CONTEXTOS SEMANTICOS RELEVANTES
    # ===============================
    # Buscamos en nuestra base de conocimiento si hay informacion relacionada
    # con la pregunta del usuario (usando embeddings para encontrar textos similares)
    semantic_contexts = search_similar_embeddings(question, top_k=3)

    # Comenzamos con la pregunta del usuario
    user_prompt = question

    # ===============================
    # 10.2 AGREGAR CONTEXTOS ENCONTRADOS
    # ===============================
    if semantic_contexts:
        # Si encontramos contextos relevantes, los mostramos en consola
        print(f" Se encontraron {len(semantic_contexts)} contextos relevantes.")

        # Construimos un texto con todos los contextos encontrados
        # Cada contexto incluye su puntuacion de similitud (que tan parecido es)
        context_text = "\n\n".join(
            [f"Contexto {i+1} (similitud {round(c['similaridad'], 3)}): {c['texto']}" 
             for i, c in enumerate(semantic_contexts)]
        )
        user_prompt += f"\n\nUsa el siguiente contexto para responder:\n{context_text}"
        if context:
            user_prompt += f"\n\nContexto adicional:\n{context}"
    
    # ===============================
    # 10.3 CONSTRUIR MENSAJES PARA LA IA
    # ===============================
    # Creamos la lista de mensajes que vamos a enviar al modelo de IA
    messages = [{"role": "system", "content": system_prompt}]  # Primero el prompt del sistema

    # Si hay historial de conversacion, agregamos los ultimos 6 mensajes
    # Esto le da "memoria" a la IA sobre lo que se ha hablado antes
    if history:
        messages += history[-6:]  # Solo los ultimos 6 para no saturar el contexto
    
    # Finalmente agregamos el mensaje actual del usuario
    messages.append({"role": "user", "content": user_prompt})

    # ===============================
    # 10.4 LLAMAR A LA IA Y OBTENER RESPUESTA
    # ===============================
    try:
        # Hacemos la peticion a OpenAI GPT-4o-mini
        res = OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o-mini",         # Modelo a usar (version ligera de GPT-4)
            messages=messages,            # Los mensajes que construimos arriba
            max_tokens=800,               # Maximo de palabras en la respuesta
            temperature=0.7               # Creatividad (0=robotico, 1=muy creativo)
        )
        
        # Extraemos y devolvemos el texto de la respuesta
        return res.choices[0].message.content.strip()
        
    except Exception as e:
        # ===============================
        # 10.5 MANEJO DE ERRORES EN LA IA
        # ===============================
        # Si algo sale mal al llamar a la IA, capturamos el error
        print(f"Error en chat: {e}")
        return "No pude generar una respuesta en este momento."