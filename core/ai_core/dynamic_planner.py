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
    if any(w in txt for w in ["pastel", "pie", "porcentaje"]):
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
                 schema_embeddings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Genera un plan de acción teniendo en cuenta:
      - Intención detectada (intent_data)
      - Esquema semántico (tablas, columnas, relaciones)
      - Esquema de embeddings (documentos, vector fields)
    """
    tipo = intent_data.get("tipo", "conversacion")
    entidades_raw = intent_data.get("entidades", {}) or {}
    texto = (intent_data.get("texto") or "").lower()

    plan: Dict[str, Any] = {
        "accion": "chat_general",
        "objetivo": "mantener conversación",
        "entidades": {},
        "filtros": {},
        "texto": texto
    }

    # --- Detectar tabla usando esquema semántico ---
    tabla_detectada = None
    candidate_tables = list(schema_semantic.get("tables", {}).keys()) + ["document_embeddings"]
    # Buscar coincidencia en entidades o texto
    for candidate in candidate_tables:
        kws = schema_semantic.get("tables", {}).get(candidate, {}).get("keywords", [])
        if any(word in texto for word in kws):
            tabla_detectada = candidate
            break
    if not tabla_detectada:
        tabla_detectada = "usuario"  # fallback

    # --- Detectar atributos validos según esquema semántico ---
    atributos = []
    columnas = schema_semantic.get("tables", {}).get(tabla_detectada, {}).get("columns", [])
    for col in columnas:
        if col.lower() in texto:
            atributos.append(col)
    if not atributos:
        atributos = ["*"]

    # --- Filtros heurísticos ---
    periodo_match = re.search(r"(20\d{2})(?:[-/](20\d{2}))?", texto)
    if periodo_match:
        plan["filtros"]["periodo"] = periodo_match.group(0)
    equipo_match = re.search(r"equipo\s+([a-zA-Z0-9_\-]+)", texto)
    if equipo_match:
        plan["filtros"]["equipo"] = equipo_match.group(1)
    persona = entidades_raw.get("persona") or entidades_raw.get("nombre")
    if persona:
        if isinstance(persona, list):
            persona = persona[0]
        plan["filtros"]["persona"] = persona

    # --- Determinar acción ---
    if tipo in ["generar_informe", "consultar_datos", "consultar"]:
        plan["accion"] = "generar_informe"
        plan["objetivo"] = "consultar y sintetizar información de la base de datos"
    elif tipo == "evaluar":
        plan["accion"] = "evaluar_colaborador"
        plan["objetivo"] = "analizar métricas de desempeño de un colaborador"
    elif tipo == "guardar_resultado":
        plan["accion"] = "guardar_resultado"
        plan["objetivo"] = "almacenar información procesada"

    plan["entidades"]["tabla"] = tabla_detectada
    plan["entidades"]["atributos"] = atributos

    # --- Meta para query_generator ---
    meta: Dict[str, Any] = {}

    # Aggregations y group_by
    aggregations = _detect_aggregation_and_metric(texto)
    if aggregations:
        # validar columnas con schema_semantic
        valid_aggs = [agg for agg in aggregations if agg["col"] in columnas or agg["col"] == "*"]
        if valid_aggs:
            meta["aggregations"] = valid_aggs

    # Group_by heurístico
    group_by = []
    for col in columnas:
        if f"por {col.lower()}" in texto:
            group_by.append(col)
    if group_by:
        meta["group_by"] = group_by

    # Visualización
    vis = _detect_visualization(texto)
    if vis:
        meta["visualization"] = vis

    # Límite
    meta["limit"] = 100
    if any(w in texto for w in ["todos", "completo", "todas"]):
        meta["limit"] = 1000

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
