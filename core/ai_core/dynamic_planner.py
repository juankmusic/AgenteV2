# core/ai_core/dynamic_planner.py
import json

# ===========================================
# PLANIFICADOR DINÁMICO DEL AGENTE INTELIGENTE
# ===========================================

# core/ai_core/dynamic_planner.py (agrega dentro de plan_actions)
def plan_actions(intent_data):
    tipo = intent_data.get("tipo")

    # extraer entidades detectadas
    entidades = intent_data.get("entidades", {})
    texto = intent_data.get("texto", "").lower()

    # inicializa el plan base
    plan = {"accion": "chat_general", "parametros": {}}

    # ejemplo: detectar si hay nombre propio u otros términos relevantes
    nombre = None
    for ent in entidades.values():
        if isinstance(ent, str) and ent.istitle():
            nombre = ent
            break
    if not nombre:
        import re
        match = re.search(r"\b(justin|maria|carlos|juan)\b", texto)
        if match:
            nombre = match.group(1)

    # ahora decidir acción según intención
    if tipo in ["generar_informe", "consultar"]:
        plan["accion"] = "consultar_datos"
        plan["parametros"] = {"nombre": nombre}

    elif tipo == "evaluar":
        plan["accion"] = "evaluar_colaborador"
        plan["parametros"] = entidades

    elif tipo == "guardar_resultado":
        plan["accion"] = "guardar_resultado"
        plan["parametros"] = entidades

    return plan
# ===========================================