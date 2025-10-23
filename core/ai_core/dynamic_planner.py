# core/ai_core/dynamic_planner.py
import json
import re

def plan_actions(intent_data):
    """
    Genera un plan de acción estructurado para el AIGR (Agente Inteligente con Razonamiento)
    a partir de la intención detectada y las entidades extraídas del texto del usuario.
    """
    tipo = intent_data.get("tipo", "conversacion")
    entidades = intent_data.get("entidades", {})
    texto = intent_data.get("texto", "").lower()

    # --- 1️⃣ Inicializa plan base ---
    plan = {
        "accion": "chat_general",
        "objetivo": "mantener conversación",
        "entidades": {},
        "filtros": {},
        "texto": texto
    }

    # --- 2️⃣ Identificación semántica de posibles tablas ---
    tabla_candidates = {
        "colaborador": ["colaborador", "empleado", "persona", "usuario", "trabajador"],
        "evaluacion": ["evaluación", "evaluaciones", "desempeño", "performance"],
        "document_embeddings": ["documento", "texto", "embedding", "vector"],
        "chatbot_logs": ["historial", "conversación", "mensaje"]
    }

    tabla_detectada = None
    for tabla, keywords in tabla_candidates.items():
        if any(word in texto for word in keywords):
            tabla_detectada = tabla
            break

    # --- 3️⃣ Inferencia de atributos comunes ---
    atributos = []
    if any(word in texto for word in ["nombre", "usuario", "persona"]):
        atributos.append("nombre")
    if any(word in texto for word in ["nivel", "contribución", "puntaje", "evaluación"]):
        atributos.append("nivel_contribucion")
    if "cargo" in texto:
        atributos.append("cargo")
    if "correo" in texto:
        atributos.append("correo")
    if "rol" in texto or "cargo" in texto:
        plan["entidades"]["tablas_relacionadas"] = ["rol"]  # puede extenderse luego
    if "evaluación" in texto or "nivel" in texto or "desempeño" in texto:
        plan["entidades"]["tablas_relacionadas"] = plan["entidades"].get("tablas_relacionadas", []) + ["evaluacion"]


    # --- 4️⃣ Filtros adicionales ---
    periodo = re.search(r"(202\d|202\d\d?)", texto)
    equipo = None
    match_equipo = re.search(r"equipo\s+(\w+)", texto)
    if match_equipo:
        equipo = match_equipo.group(1)

    if periodo:
        plan["filtros"]["periodo"] = periodo.group(1)
    if equipo:
        plan["filtros"]["equipo"] = equipo

    # --- 5️⃣ Lógica de acción según intención ---
    if tipo in ["generar_informe", "consultar_datos"]:
        plan["accion"] = "generar_informe"
        plan["objetivo"] = "consultar y sintetizar información de la base de datos"
    elif tipo == "evaluar":
        plan["accion"] = "evaluar_colaborador"
        plan["objetivo"] = "analizar métricas de desempeño de un colaborador"
    elif tipo == "guardar_resultado":
        plan["accion"] = "guardar_resultado"
        plan["objetivo"] = "almacenar información procesada"
    else:
        plan["accion"] = "chat_general"
        plan["objetivo"] = "mantener conversación general"

    # --- 6️⃣ Asignar contexto semántico ---
    plan["entidades"]["tabla"] = tabla_detectada or "usuario"
    plan["entidades"]["atributos"] = atributos or ["*"]

    return plan
