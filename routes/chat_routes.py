import os
import re
import uuid
import subprocess
from flask import Blueprint, request, jsonify, session, render_template
from werkzeug.utils import secure_filename
from core.chatbot_logic import OPENAI_CLIENT
from io import BytesIO
from core.chatbot_logic import deepseek_chat, get_history
from core.file_processing import get_embeddings, process_pdf_and_save_chunks, process_csv_and_save_chunks, search_similar_chunks
from db.connection import connect_db

# --- Configuración ---
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# blueprint con carpeta de templates
chat_bp = Blueprint(
    "chat_bp",
    __name__,
    template_folder="../templates"  # "../" porque estamos en /routes
)

# ============================================================
# RUTA: Página principal del chat
# ============================================================
@chat_bp.route("/")
def index():
    """Muestra la interfaz de chat"""
    return render_template("chat.html")


# ============================================================
# RUTA: Subir Archivo (PDF / CSV)
# ============================================================
@chat_bp.route("/upload", methods=["POST"])
def upload_file():
    """Permite subir un archivo PDF o CSV, procesarlo y almacenarlo."""
    if "file" not in request.files:
        return jsonify({"error": "No se envió ningún archivo"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nombre de archivo vacío"}), 400

    filename = secure_filename(file.filename)
    file_ext = os.path.splitext(filename)[1].lower()
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Procesar según el tipo
    if file_ext == ".pdf":
        process_pdf_and_save_chunks(filepath)
    elif file_ext == ".csv":
        process_csv_and_save_chunks(filepath)
    else:
        return jsonify({"error": "Tipo de archivo no soportado"}), 400

    session["uploaded_file"] = filename
    return jsonify({"message": f"Archivo '{filename}' procesado correctamente."})


# ============================================================
# RUTA: Enviar mensaje al chatbot
# ============================================================
@chat_bp.route("/chat", methods=["POST"])
def chat_with_bot():
    """Permite al usuario enviar preguntas al chatbot (con contexto si hay un archivo cargado)."""
    data = request.get_json()
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"error": "Mensaje vacío"}), 400

    # --------------------------
    # Session ID
    # --------------------------
    session_id = data.get("session_id") or session.get("id")
    if not session_id:
        session_id = str(uuid.uuid4())
        session["id"] = session_id

    # --------------------------
    # Historial de la sesión
    # --------------------------
    conn = connect_db()
    history = []
    if conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT role, message FROM chatbot_logs
                WHERE session_id = %s
                ORDER BY timestamp
            """, (session_id,))
            rows = cur.fetchall()
            history = [{"role": "assistant" if r[0]=="bot" else r[0], "content": r[1]} for r in rows]

    # --------------------------
    # Contexto de archivos (PDF / CSV)
    # --------------------------
    context_text = ""
    active_file = session.get("uploaded_file")
    if active_file:
        query_embedding = get_embeddings([user_input])[0]
        print("Archivo activo:", active_file)
        print("Pregunta del usuario:", user_input)

        results = search_similar_chunks(
            query_embedding,
            source_filename=os.path.basename(active_file)
        )

        # Detectar preguntas generales sobre el PDF
        general_pdf_questions = ["de que trata", "resumen del pdf", "resumen del documento", "de que se trata"]
        is_general_question = any(q in user_input.lower() for q in general_pdf_questions)

        # Si es pregunta general y no hay resultados, usar primeros N chunks
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
                results = [(r[0], active_file, None, 1.0) for r in rows]  # sim=1.0 por default
            except Exception as e:
                print(f"❌ Error al obtener primeros chunks: {e}")
                results = []

        if results:
            context_text = "\n".join([f"• {t}" for t, _, _, _ in results])

        print("Resultados de búsqueda:", results)

    # --------------------------
    # Llamar modelo DeepSeek
    # --------------------------
    response_text = deepseek_chat(user_input, context=context_text, history=history)

    # --------------------------
    # Guardar en BD
    # --------------------------
    if conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO chatbot_logs (session_id, role, message)
                VALUES (%s, %s, %s)
            """, (session_id, "user", user_input))
            cur.execute("""
                INSERT INTO chatbot_logs (session_id, role, message)
                VALUES (%s, %s, %s)
            """, (session_id, "bot", response_text))
            conn.commit()
        conn.close()

    return jsonify({
        "response": response_text,
        "context_used": bool(context_text),
        "source_file": active_file or None
    })

# ============================================================
# RUTA: Obtener historial de la sesión
# ============================================================
@chat_bp.route("/history", methods=["GET"])
def get_history():
    """Obtiene el historial de chat guardado de la sesión indicada."""
    sid = request.args.get("session_id") or session.get("id")
    if not sid:
        return jsonify({"history": []})

    conn = connect_db()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT role, message FROM chatbot_logs
            WHERE session_id = %s
            ORDER BY timestamp
        """, (sid,))
        rows = cur.fetchall()
    conn.close()

    history = [{"role": r[0], "message": r[1]} for r in rows]
    return jsonify({"history": history})


# ============================================================
# RUTA: Listar sesiones anteriores
# ============================================================
@chat_bp.route("/sessions", methods=["GET"])
def list_sessions():
    """Devuelve todas las sesiones existentes con cantidad de mensajes."""
    conn = connect_db()
    if not conn:
        return jsonify({"sessions": []})
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT session_id, COUNT(*) as msg_count, MAX(timestamp) as last_msg
                FROM chatbot_logs
                GROUP BY session_id
                ORDER BY last_msg DESC
            """)
            rows = cur.fetchall()
        sessions = [{"session_id": r[0], "msg_count": r[1]} for r in rows]
        return jsonify({"sessions": sessions})
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