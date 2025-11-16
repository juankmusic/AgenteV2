# core/ai_core/dynamic_planner.py
import json
import re
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level="INFO")


def _detect_aggregation_and_metric(text: str) -> List[Dict[str, str]]:
    aggs = []
    txt = text.lower()
    if any(w in txt for w in ["promedio", "media", "avg", "mean"]):
        if re.search(r"(puntaje|nivel|score|salario|ingreso|ventas?)", txt):
            aggs.append({"op": "avg", "col": "nivel_contribucion"})
    if any(w in txt for w in ["suma", "total", "sum"]):
        if re.search(r"(ventas|monto|cantidad|importe|score|puntaje)", txt):
            aggs.append({"op": "sum", "col": "monto"})
    if any(w in txt for w in ["contar", "cantidad", "numero", "cuantos", "count"]):
        aggs.append({"op": "count", "col": "*"})
    if any(w in txt for w in ["top", "mejores", "peores", "rank"]):
        aggs.append({"op": "rank_hint", "col": "nivel_contribucion"})
    return aggs


def _detect_visualization(text: str) -> Optional[str]:
    txt = text.lower()
    if any(w in txt for w in ["gráfico", "grafico", "plot", "curve", "serie temporal", "trend", "tendencia", "evolución"]):
        return "line"
    if any(w in txt for w in ["histograma", "distribución", "distribucion"]):
        return "histogram"
    if any(w in txt for w in ["barra", "bar", "barras", "bar chart"]):
        return "bar"
    if any(w in txt for w in ["pastel", "pie", "porcentaje", "circular"]):
        return "pie"
    if any(w in txt for w in ["tabla", "mostrar", "listar", "listado"]):
        return "table"
    return None


def _need_clarification_for_plan(plan: Dict[str, Any]) -> Optional[str]:
    accion = plan.get("accion")
    filtros = plan.get("filtros", {})
    if accion == "evaluar_colaborador" and not filtros.get("persona"):
        return "¿A qué colaborador te refieres? Indica nombre o identificador."
    if accion == "generar_informe":
        tabla = plan.get("entidades", {}).get("tabla", "")
        if tabla in ("usuario", "colaborador") and not filtros:
            return "¿Quieres un informe para toda la organización o para un equipo/periodo específico?"
    return None


def plan_actions(intent_data: Dict[str, Any],
                 schema_semantic: Dict[str, Any],
                 schema_embeddings: Dict[str, Any],
                 last_plan: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Genera un plan de acción, utilizando el plan anterior (last_plan) para dar contexto a solicitudes de continuación.
    """
    tipo = intent_data.get("tipo", "conversacion")
    entidades_raw = intent_data.get("entidades", {}) or {}
    texto = (intent_data.get("texto") or "").lower()

    persona = None

    plan: Dict[str, Any] = {
        "accion": "chat_general",
        "objetivo": "mantener conversación",
        "entidades": {},
        "filtros": {},
        "texto": texto,
        "meta": {} # Inicializar meta
    }
    
    # ----------------------------------------------------
    # 🎯 LÓGICA DE CONTINUACIÓN (Memoria de datos)
    # ----------------------------------------------------
    # Heurística: ¿El texto solo contiene una solicitud de visualización y NO es una nueva pregunta de datos?
    is_visualization_change_only = (_detect_visualization(texto) is not None) and \
                                 (tipo in ["conversacion", "consultar_datos", "generar_informe"]) and \
                                 (not any(w in texto for w in ["qué", "cuales", "cuantos", "dime", "saber"])) and \
                                 (not entidades_raw.get("area") and not entidades_raw.get("periodo")) and \
                                 (not _detect_aggregation_and_metric(texto))
    
    # Si tenemos un plan anterior Y la solicitud es una continuación

    if last_plan and (is_visualization_change_only or (tipo in ["generar_informe", "consultar_datos"] and not entidades_raw)):
        
        logger.info("Reutilizando contexto del plan anterior.")
        
        # Reutilizar entidades, filtros y meta base del plan anterior
        plan["entidades"] = last_plan.get("entidades", {})
        plan["filtros"] = last_plan.get("filtros", {})
        plan["meta"] = last_plan.get("meta", {})
        
        # 📌 LÓGICA DE REUTILIZACIÓN FORZADA (El cambio más importante)
        if is_visualization_change_only and last_plan.get("accion") in ["generar_informe", "evaluar_colaborador"]:
            
            # Forzar la acción a reutilizar la consulta anterior
            plan["accion"] = "reutilizar_consulta"
            plan["objetivo"] = f"actualizar visualización de la consulta anterior: {last_plan.get('objetivo', 'datos')}"
            
            # Pasar el SQL anterior para que el Query Generator lo use directamente
            previous_sql = last_plan.get("executed_sql")
            if previous_sql:
                plan["meta"]["previous_sql"] = previous_sql
            else:
                # Si no hay SQL anterior, no podemos reutilizar; volvemos al flujo normal
                plan["accion"] = "generar_informe"
            
            # Sobrescribir el tipo de visualización si se detectó uno nuevo
            vis_new = _detect_visualization(texto)
            if vis_new:
                plan["meta"]["visualization"] = vis_new
            
            # 🛑 Si la acción es reutilizar_consulta y tenemos el SQL, terminamos aquí.
            if plan["accion"] == "reutilizar_consulta":
                plan["confidence"] = 1.0 
                plan["confidence_reasons"] = ["reutilizacion_forzada_visualizacion"]
                return plan
        # Si no es un cambio de visualización forzado, la acción y el objetivo mantienen la del plan anterior
        elif last_plan.get("accion") in ["generar_informe", "evaluar_colaborador"]:
            plan["accion"] = last_plan["accion"]
            plan["objetivo"] = last_plan["objetivo"]


    # ----------------------------------------------------
    # 1. Detección de entidades y atributos (priorizar nuevo input)
    # ----------------------------------------------------
    
    # --- Detectar tabla usando esquema semántico ---
    tabla_detectada = None
    candidate_tables = list(schema_semantic.get("tables", {}).keys()) + ["document_embeddings"]
    # Buscar coincidencia en entidades o texto
    for candidate in candidate_tables:
        kws = schema_semantic.get("tables", {}).get(candidate, {}).get("keywords", [])
        if any(word in texto for word in kws):
            tabla_detectada = candidate
            break
    
    # Si se detectó una tabla, sobrescribir la del plan anterior
    if tabla_detectada:
        plan["entidades"]["tabla"] = tabla_detectada
    else:
        tabla_detectada = plan["entidades"].get("tabla", "usuario") # Usar tabla del plan o fallback

    # --- Detectar atributos validos según esquema semántico ---
    atributos = []
    columnas = schema_semantic.get("tables", {}).get(tabla_detectada, {}).get("columns", [])
    for col in columnas:
        if col.lower() in texto:
            atributos.append(col)
    
    # Si se detectaron atributos, sobrescribir. Si no, usar los del contexto o el fallback
    if atributos:
        plan["entidades"]["atributos"] = atributos
    elif not plan["entidades"].get("atributos"):
        plan["entidades"]["atributos"] = ["*"]
        
    # --- Filtros heurísticos (Sobreescribir o añadir a los del plan anterior) ---
    periodo_match = re.search(r"(20\d{2})(?:[-/](20\d{2}))?", texto)
    if periodo_match:
        plan["filtros"]["periodo"] = periodo_match.group(0)
    equipos_validos = ["omega", "alfa", "beta", "delta"]

    # Detectar: "equipo <nombre>"
    equipo_match = re.search(r"\bequipo\s+([a-zA-Z0-9_\-]+)\b", texto)
    if equipo_match:
        posible_equipo = equipo_match.group(1).lower()

        # Solo aceptar si está dentro de los equipos válidos
        if posible_equipo in equipos_validos:
            plan["filtros"]["equipo"] = posible_equipo
        else:
            # Ignorar coincidencias no válidas (score, seguimiento, etc.)
            logger.info(f"Ignorando coincidencia de equipo no válida: '{posible_equipo}'")
    if persona:
        if isinstance(persona, list):
            persona = persona[0]
        plan["filtros"]["persona"] = persona

    # --- Determinar acción (Sobrescribir si hay una intención fuerte) ---
    if tipo in ["generar_informe", "consultar_datos", "consultar"]:
        plan["accion"] = "generar_informe"
        plan["objetivo"] = "consultar y sintetizar información de la base de datos"
    elif tipo == "evaluar":
        plan["accion"] = "evaluar_colaborador"
        plan["objetivo"] = "analizar métricas de desempeño de un colaborador"
    elif tipo == "guardar_resultado":
        plan["accion"] = "guardar_resultado"
        plan["objetivo"] = "almacenar información procesada"
        
    # plan["entidades"]["tabla"] y plan["entidades"]["atributos"] ya están seteados arriba

    # --- Meta para query_generator ---
    meta: Dict[str, Any] = plan.get("meta", {}) # Reutilizar meta del plan si existe

    # Aggregations y group_by (Sobrescribir si se detecta nueva, sino mantener la anterior)
    aggregations = _detect_aggregation_and_metric(texto)
    if aggregations:
        # validar columnas con schema_semantic
        valid_aggs = [agg for agg in aggregations if agg["col"] in columnas or agg["col"] == "*"]
        if valid_aggs:
            meta["aggregations"] = valid_aggs

    # Group_by heurístico (Sobrescribir si se detecta nueva, sino mantener la anterior)
    group_by = []
    for col in columnas:
        if f"por {col.lower()}" in texto:
            group_by.append(col)
    if group_by:
        meta["group_by"] = group_by

    # Visualización (Siempre se sobrescribe, ya que es el cambio más probable)
    vis = _detect_visualization(texto)
    if vis:
        meta["visualization"] = vis

    # Límite
    if any(w in texto for w in ["todos", "completo", "todas"]):
        meta["limit"] = 1000
    # Si no se especifica, usa el límite anterior o el default 100
    elif "limit" not in meta:
        meta["limit"] = 100

    # --- Embeddings ---
    if tabla_detectada == "document_embeddings":
        meta["embedding_query"] = texto
        meta["top_k"] = 10
        embedding_fields = schema_embeddings.get("vector_fields", [])
        if embedding_fields:
            meta["embedding_field"] = embedding_fields[0]

    plan["meta"] = meta

    # --- Confianza heurística ---
    score = 0.0
    reasons: List[str] = []
    if persona:
        score += 0.25; reasons.append("persona_detectada")
    if periodo_match:
        score += 0.15; reasons.append("periodo_detectado")
    if tabla_detectada and tabla_detectada != "usuario":
        score += 0.2; reasons.append(f"tabla_detectada:{tabla_detectada}")
    if aggregations:
        score += 0.1; reasons.append("aggregation_hint")
    if vis:
        score += 0.1; reasons.append(f"visualization_suggested:{vis}")
        
    # ➕ Bonificación por reusar el contexto (Solo si no hubo detección de tabla nueva)
    if last_plan and not tabla_detectada:
        score += 0.15; reasons.append("contexto_reutilizado")
        
    confidence = min(round(score, 3), 1.0)
    plan["confidence"] = confidence
    plan["confidence_reasons"] = reasons

    # --- Clarificación si hace falta ---
    clarify_q = _need_clarification_for_plan(plan)
    if clarify_q:
        plan["clarify"] = True
        plan["clarify_question"] = clarify_q
    else:
        plan["clarify"] = False

    return plan