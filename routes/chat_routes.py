# routes/chat_routes.py

# ===============================
# IMPORTACIONES DEL SISTEMA
# ===============================
import os          # Para manejar archivos y carpetas del sistema operativo
import re          # Para buscar patrones en textos (expresiones regulares)
import uuid        # Para crear identificadores unicos de sesiones
import subprocess  # Para ejecutar comandos externos (como ffmpeg para convertir audio)

# Importaciones de Flask (framework web)
from flask import Blueprint, request, jsonify, session, render_template
# - Blueprint: Para organizar rutas en modulos separados
# - request: Para recibir datos del usuario (mensajes, archivos)
# - jsonify: Para convertir datos de Python a formato JSON
# - session: Para guardar informacion del usuario mientras navega
# - render_template: Para mostrar paginas HTML

from werkzeug.utils import secure_filename  # Para limpiar nombres de archivos (seguridad)

# Importaciones de nuestros modulos personalizados
from core.chatbot_logic import OPENAI_CLIENT  # Cliente para comunicarse con OpenAI
from io import BytesIO  # Para manejar datos en memoria (como archivos temporales)
from core.chatbot_logic import deepseek_chat, get_history  # Funciones del cerebro del chatbot
from core.file_processing import get_embeddings, process_pdf_and_save_chunks, process_csv_and_save_chunks, search_similar_chunks
# - get_embeddings: Convierte texto en numeros para compararlo
# - process_pdf_and_save_chunks: Lee PDFs y los divide en pedazos pequeños
# - process_csv_and_save_chunks: Lee archivos CSV (como Excel) y los guarda
# - search_similar_chunks: Busca textos similares a una pregunta

from db.connection import connect_db  # Funcion para conectar con la base de datos

# ===============================
# CONFIGURACION INICIAL
# ===============================

# Carpeta donde se guardaran los archivos que suba el usuario
UPLOAD_FOLDER = "uploads"
# Crear la carpeta si no existe
# exist_ok=True significa que no da error si ya existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===============================
# CREAR BLUEPRINT (modulo de rutas)
# ===============================
# Un Blueprint es como un "mini-aplicacion" dentro de Flask
# Nos permite organizar las rutas en archivos separados
chat_bp = Blueprint(
    "chat_bp",                          # Nombre interno del blueprint
    __name__,                           # Nombre del modulo actual
    template_folder="../templates"      # Donde estan los archivos HTML
)                                       # "../" significa subir una carpeta

# ============================================================
# RUTA: Página principal del chat
# ============================================================
@chat_bp.route("/")
def index():
    """Muestra la interfaz de chat cuando el usuario entra a la pagina principal"""
    return render_template("chat.html")


# ============================================================
# RUTA: Subir Archivo (PDF / CSV)
# ============================================================
@chat_bp.route("/upload", methods=["POST"])
def upload_file():
    """Permite subir un archivo PDF o CSV, procesarlo y almacenarlo."""

    # ===============================
    # 1. VERIFICAR QUE SE ENVIO UN ARCHIVO
    # ===============================
    if "file" not in request.files:
        # Si no hay archivo en la peticion, devolvemos error
        return jsonify({"error": "No se envió ningún archivo"}), 400

    # Obtenemos el archivo del request
    file = request.files["file"]

    # Verificamos que el archivo tenga nombre
    if file.filename == "":
        return jsonify({"error": "Nombre de archivo vacío"}), 400

    # ===============================
    # 2. GUARDAR EL ARCHIVO DE FORMA SEGURA
    # ===============================
    # secure_filename limpia el nombre del archivo para evitar problemas de seguridad
    # Ejemplo: "../../etc/passwd" se convierte en "etc_passwd"
    filename = secure_filename(file.filename)

    # Obtenemos la extension del archivo (.pdf, .csv, etc.)
    file_ext = os.path.splitext(filename)[1].lower()

    # Creamos la ruta completa donde guardar el archivo
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    # Guardamos el archivo en el disco
    file.save(filepath)

     # ===============================
    # 3. PROCESAR SEGUN EL TIPO DE ARCHIVO
    # ===============================
    if file_ext == ".pdf":
        # Si es PDF, lo dividimos en pedazos y guardamos en la BD
        process_pdf_and_save_chunks(filepath)
        
    elif file_ext == ".csv":
        # Si es CSV, lo procesamos y guardamos en la BD
        process_csv_and_save_chunks(filepath)
        
    else:
        # Si no es PDF ni CSV, devolvemos error
        return jsonify({"error": "Tipo de archivo no soportado"}), 400

    # ===============================
    # 4. GUARDAR EN LA SESION DEL USUARIO
    # ===============================
    # Guardamos el nombre del archivo en la sesion para recordarlo
    # mientras el usuario navega por el sitio
    session["uploaded_file"] = filename

    # Devolvemos mensaje de exito
    return jsonify({"message": f"Archivo '{filename}' procesado correctamente."})


# ============================================================
# RUTA: Enviar mensaje al chatbot
# ============================================================
@chat_bp.route("/chat", methods=["POST"])
def chat_with_bot():
    """Esta es la ruta MAS IMPORTANTE. Recibe los mensajes del usuario,
    busca contexto relevante si hay archivos cargados y genera respuestas."""

    # ===============================
    # 1. OBTENER EL MENSAJE DEL USUARIO
    # ===============================
    data = request.get_json()  # Obtenemos los datos JSON enviados
    user_input = data.get("message", "").strip()  # Extraemos el mensaje y quitamos espacios
    
    # Si el mensaje esta vacio, devolvemos error
    if not user_input:
        return jsonify({"error": "Mensaje vacio"}), 400

    # ===============================
    # 2. OBTENER O CREAR ID DE SESION
    # ===============================
    # Cada conversacion tiene un ID unico para identificarla
    # Primero intentamos obtenerlo del request, luego de la sesion de Flask
    session_id = data.get("session_id") or session.get("id")

    # Si no existe, creamos uno nuevo
    if not session_id:
        session_id = str(uuid.uuid4())  # Genera algo como "a1b2c3d4-e5f6-..."
        session["id"] = session_id      # Lo guardamos en la sesion


    # ===============================
    # 3. RECUPERAR HISTORIAL DE LA CONVERSACION
    # ===============================
    conn = connect_db()  # Conectamos con la base de datos
    history = []         # Lista vacia por defecto
    if conn:
        with conn.cursor() as cur:
            # Consultamos todos los mensajes de esta sesion
            cur.execute("""
                SELECT role, message FROM chatbot_logs
                WHERE session_id = %s
                ORDER BY timestamp
            """, (session_id,))
            rows = cur.fetchall()

            # Convertimos cada mensaje a un diccionario
            # Si el role es "bot" lo cambiamos a "assistant"
            history = [{"role": "assistant" if r[0]=="bot" else r[0], "content": r[1]} for r in rows]

   # ===============================
    # 4. BUSCAR CONTEXTO EN ARCHIVOS CARGADOS
    # ===============================
    context_text = ""  # Texto de contexto vacio por defecto
    active_file = session.get("uploaded_file")  # Archivo activo en esta sesion

    if active_file:
         # Si hay un archivo cargado, buscamos informacion relevante
        
        # Convertimos la pregunta en un embedding (representacion numerica)
        query_embedding = get_embeddings([user_input])[0]
        print("Archivo activo:", active_file)
        print("Pregunta del usuario:", user_input)

        # Buscamos pedazos de texto similares a la pregunta
        results = search_similar_chunks(
            query_embedding,
            source_filename=os.path.basename(active_file)
        )

         # ===============================
        # 4.1 DETECTAR PREGUNTAS GENERALES
        # ===============================
        # Algunas preguntas piden un resumen general del documento
        general_pdf_questions = ["de que trata", "resumen del pdf", "resumen del documento", "de que se trata"]
        is_general_question = any(q in user_input.lower() for q in general_pdf_questions)

        # Si es pregunta general y no encontramos resultados especificos
        # usamos los primeros pedazos del documento
        if is_general_question and not results:
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT text_chunk FROM document_embeddings
                        WHERE source_filename = %s
                        ORDER BY page_number
                        LIMIT 10
                    """, (os.path.basename(active_file),))
                    rows = cur.fetchall()

                # Convertimos los resultados al formato esperado
                # (texto, archivo, None, similitud=1.0)
                results = [(r[0], active_file, None, 1.0) for r in rows]
                
            except Exception as e:
                print(f"❌ Error al obtener primeros chunks: {e}")
                results = []

        # Si hay resultados, construimos el texto de contexto
        if results:
            # Juntamos todos los textos con viñetas
            context_text = "\n".join([f"• {t}" for t, _, _, _ in results])

        print("Resultados de búsqueda:", results)

    # ===============================
    # 5. GENERAR RESPUESTA DEL CHATBOT
    # ===============================
    # Llamamos a la funcion principal del cerebro del chatbot
    response_text = deepseek_chat(
        session_id,      # ID de la sesion
        user_input,      # Pregunta del usuario
        context=context_text,  # Contexto del archivo (si hay)
        history=history  # Historial de mensajes anteriores
    )


   # ===============================
    # 6. GUARDAR EN LA BASE DE DATOS
    # ===============================
    if conn:
        with conn.cursor() as cur:
            # Guardamos el mensaje del usuario
            cur.execute("""
                INSERT INTO chatbot_logs (session_id, role, message)
                VALUES (%s, %s, %s)
            """, (session_id, "user", user_input))
            
            # Guardamos la respuesta del bot
            cur.execute("""
                INSERT INTO chatbot_logs (session_id, role, message)
                VALUES (%s, %s, %s)
            """, (session_id, "bot", response_text))
            
            conn.commit()  # Confirmamos los cambios
        conn.close()  # Cerramos la conexion
    # ===============================

    # ===============================
    # 7. DEVOLVER RESPUESTA AL USUARIO
    # ===============================
    return jsonify({
        "response": response_text,            # La respuesta del bot
        "context_used": bool(context_text),   # True si se uso contexto de archivos
        "source_file": active_file or None    # Nombre del archivo usado (si hay)
    })

# ============================================================
# RUTA: Obtener historial de la sesión
# ============================================================
@chat_bp.route("/history", methods=["GET"])
def get_history():
    """Obtiene el historial de chat guardado de la sesión indicada."""

    # Obtener el ID de sesión del request o de la sesión de Flask
    sid = request.args.get("session_id") or session.get("id")

    # Si no hay ID, devolvemos historial vacio
    if not sid:
        return jsonify({"history": []})
    
    # Conectamos con la base de datos
    conn = connect_db()
    with conn.cursor() as cur:
        # Consultamos todos los mensajes de esta sesion
        cur.execute("""
            SELECT role, message FROM chatbot_logs
            WHERE session_id = %s
            ORDER BY timestamp
        """, (sid,))
        rows = cur.fetchall()
    conn.close()

    # Convertimos los resultados a lista de diccionarios
    history = [{"role": r[0], "message": r[1]} for r in rows]

    return jsonify({"history": history})

# ============================================================
# RUTA: Listar sesiones anteriores
# ============================================================
@chat_bp.route("/sessions", methods=["GET"])
def list_sessions():
    """Devuelve todas las sesiones con conteo, fecha y el primer mensaje del usuario."""
    conn = connect_db()
    # Si no se pudo conectar, devolvemos lista vacia
    if not conn:
        return jsonify({"sessions": []})
    try:
        with conn.cursor() as cur:
            # Consulta para obtener sesiones con conteo, ultima fecha y primer mensaje del usuario
            cur.execute("""
                SELECT 
                    T1.session_id, 
                    COUNT(T1.id) as msg_count, 
                    MAX(T1.timestamp) as last_msg,
                    -- Subconsulta para obtener el mensaje (message) y la hora (timestamp) 
                    -- del primer mensaje del usuario (role='user').
                    (SELECT T2.message
                     FROM chatbot_logs AS T2
                     WHERE T2.session_id = T1.session_id AND T2.role = 'user'
                     ORDER BY T2.timestamp ASC
                     LIMIT 1) AS first_user_msg
                FROM chatbot_logs AS T1
                WHERE T1.session_id IS NOT NULL 
                GROUP BY T1.session_id
                ORDER BY last_msg DESC
            """)
            rows = cur.fetchall()
        # Procesar resultados
        sessions = []
        for r in rows:
            # Si no hay primer mensaje, usar "Sesión sin nombre"
            first_msg = r[3] if r[3] else "Sesión sin nombre"
            
            # Agregar diccionario de sesión a la lista
            sessions.append({
                "session_id": r[0],
                "msg_count": r[1],
                "last_updated": r[2].isoformat(),
                "name": first_msg # <- ¡Nuevo campo de nombre!
            })
        
        # Devolver lista de sesiones procesada
        return jsonify({"sessions": sessions})
    
    # Manejo de errores
    except Exception as e:
        print(f"Error al listar sesiones: {e}")
        return jsonify({"sessions": []})
    
    finally:
        conn.close()


# ============================================================
# RUTA: Crear nueva sesión
# ============================================================
@chat_bp.route("/new_session", methods=["GET"])
def new_session():
    """Crea una nueva sesión y limpia la session actual"""
    new_id = str(uuid.uuid4())
    session["id"] = new_id
    session.pop("uploaded_file", None)  # opcional, limpia archivo cargado
    return jsonify({"session_id": new_id})

# ============================================================
# RUTA: Entrada de voz / transcripción
# ============================================================
@chat_bp.route("/voice", methods=["POST"])
def voice_to_text():
    if "file" not in request.files:
        return jsonify({"error": "No se envió ningún archivo"}), 400

    file = request.files["file"]
    original_filename = file.filename
    # Guardar archivo temporal
    temp_input_path = os.path.join("uploads", original_filename)
    file.save(temp_input_path)

    # Generar ruta de salida WAV
    temp_output_path = os.path.splitext(temp_input_path)[0] + "_converted.wav"

    # Convertir a WAV usando ffmpeg
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", temp_input_path, "-ar", "44100", "-ac", "1", temp_output_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Error al convertir audio: {e.stderr.decode()}"}), 500

    # Transcribir con OpenAI
    try:
        with open(temp_output_path, "rb") as f:
            transcript = OPENAI_CLIENT.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=f,
                language="es"
            )
        text = transcript.text
    except Exception as e:
        return jsonify({"error": f"No se pudo transcribir: {e}"}), 500
    finally:
        # Opcional: eliminar archivos temporales
        os.remove(temp_input_path)
        os.remove(temp_output_path)

    return jsonify({"transcription": text})