import json
import re
import logging
from typing import Dict, Any, List, Optional
import numpy as np

# 👇 Importar la excepción de error lógico
from core.exceptions import LogicalError 
# 👇 Importar las herramientas semánticas que necesitamos
from core.ai_core.nlp_embeddings import generate_embedding

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level="INFO")

# ======================================================
# 1. FUNCIONES AUXILIARES SEMÁNTICAS (EL NUEVO CEREBRO)
# ======================================================

def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calcula la similitud coseno entre dos vectores (listas de floats).
    """
    try:
        if vec1 is None or vec2 is None:
            return 0.0
        
        v1 = np.array(vec1, dtype=float)
        v2 = np.array(vec2, dtype=float)
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return float(np.dot(v1, v2) / (norm1 * norm2))
    except Exception as e:
        logger.error(f"Error en cosine_similarity: {e}")
        return 0.0

def _find_semantic_match(term: str, 
                         schema_embeddings: Dict[str, Any], 
                         entity_type: str = "any", # "table", "column", o "any"
                         threshold: float = 0.75) -> Optional[Dict[str, Any]]:
    """
    Toma un término (ej. "facultad"), genera su embedding, y lo compara
    con todos los embeddings del esquema para encontrar la mejor coincidencia.
    """
    if not term:
        return None
        
    term_embedding = generate_embedding(term)
    if not term_embedding:
        logger.warning(f"No se pudo generar embedding para el término: {term}")
        return None
        
    best_match = None
    best_score = 0.0
    
    for key, data in schema_embeddings.items():
        # Filtrar por tipo si se especifica
        if entity_type != "any" and data.get("type") != entity_type:
            continue
            
        schema_embed = data.get("embedding")
        if not schema_embed:
            continue
            
        score = _cosine_similarity(term_embedding, schema_embed)
        
        if score > best_score and score >= threshold:
            best_score = score
            best_match = {
                "key": key, # ej. "equipo.nombre" o "evaluacion"
                "type": data.get("type"),
                "score": score
            }
            
    return best_match

def _extract_concepts(text: str) -> Dict[str, Optional[str]]:
    """
    Extrae los conceptos clave de la consulta del usuario para el mapeo semántico.
    """
    text_lower = text.lower()
    
    # 1. Detectar Operación
    operation = None
    if any(w in text_lower for w in ["promedio", "media", "avg", "mean"]):
        operation = "avg"
    elif any(w in text_lower for w in ["suma", "total", "sum"]):
        operation = "sum"
    elif any(w in text_lower for w in ["contar", "cantidad", "numero", "cuantos", "count"]):
        operation = "count"
        
    # 2. Detectar Métrica (lo que se está midiendo)
    metric_term = None
    metric_match = re.search(r"(?:promedio|suma|media|total)\s+(?:de|del|de los)\s+([a-zA-Z0-9_]+)", text_lower)
    if metric_match:
        metric_term = metric_match.group(1)
    elif "puntaje" in text_lower or "score" in text_lower:
        metric_term = "puntaje" # Fallback común

    # 3. Detectar Agrupador (el "por ...")
    group_by_term = None
    group_by_match = re.search(r"por\s+([a-zA-Z0-9_]+)", text_lower)
    if group_by_match:
        group_by_term = group_by_match.group(1)

    # 4. Detectar Tabla Principal (por palabras clave comunes)
    # (El mapeo semántico de tabla completa es más complejo, usamos heurística)
    main_table_term = None
    if "evaluacion" in text_lower or "evaluaciones" in text_lower:
        main_table_term = "evaluacion"
    elif "usuario" in text_lower or "usuarios" in text_lower:
        main_table_term = "usuario"
    elif "respuesta" in text_lower or "respuestas" in text_lower:
        main_table_term = "respuesta"

    return {
        "operation": operation,
        "metric_term": metric_term,
        "group_by_term": group_by_term,
        "main_table_term": main_table_term
    }

def _get_groupable_suggestions(schema_embeddings: Dict[str, Any]) -> List[str]:
    """
    (¡Arregla el bug de "como: ?"!)
    Sugiere columnas agrupables extrayendo sus descripciones del
    schema_embeddings.
    """
    suggestions = []
    for key, data in schema_embeddings.items():
        if data.get("type") == "column":
            # Asumimos que las columnas con "nombre", "tipo", "categoria" 
            # en su clave son agrupables.
            if "nombre" in key or "tipo" in key or "categoria" in key:
                # Extraer el "nombre" del texto, ej. "Columna: nombre en la tabla equipo..."
                match = re.search(r"Columna:\s*([^\s]+)", data.get("text", ""))
                if match:
                    suggestions.append(match.group(1))
    
    return list(set(suggestions))[:5] # 5 sugerencias únicas


def _detect_visualization(text: str) -> Optional[str]:
    # ... (Sin cambios) ...
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

# ======================================================
# 2. FUNCIÓN PRINCIPAL DE PLANIFICACIÓN (SEMÁNTICA)
# ======================================================

def plan_actions(intent_data: Dict[str, Any],
                 schema_semantic: Dict[str, Any],
                 schema_embeddings: Dict[str, Any], # <--- ¡AHORA USAMOS ESTO!
                 last_plan: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Genera un plan de acción basado 100% en el schema_embeddings,
    usando similitud semántica para mapear conceptos.
    """
    tipo = intent_data.get("tipo", "conversacion")
    entidades_raw = intent_data.get("entidades", {}) or {}
    texto = (intent_data.get("texto") or "").lower()

    plan: Dict[str, Any] = {
        "accion": "chat_general",
        "objetivo": "mantener conversación",
        "entidades": {},
        "filtros": {},
        "texto": texto,
        "meta": {} 
    }
    
    # ----------------------------------------------------
    # 🎯 LÓGICA DE CONTINUACIÓN (Sin cambios)
    # ----------------------------------------------------
    is_visualization_change_only = (_detect_visualization(texto) is not None) and \
                                   (tipo in ["conversacion", "consultar_datos", "generar_informe"]) and \
                                   (not any(w in texto for w in ["qué", "cuales", "cuantos", "dime", "saber"])) and \
                                   (not entidades_raw.get("area") and not entidades_raw.get("periodo"))
    
    if last_plan and (is_visualization_change_only or (tipo in ["generar_informe", "consultar_datos"] and not entidades_raw)):
        logger.info("Reutilizando contexto del plan anterior.")
        plan.update(last_plan) # Heredar todo el plan anterior
        plan["meta"] = last_plan.get("meta", {}) # Asegurar que meta exista
        
        if is_visualization_change_only and last_plan.get("accion") in ["generar_informe", "evaluar_colaborador"]:
            plan["accion"] = "reutilizar_consulta"
            plan["objetivo"] = f"actualizar visualización de la consulta anterior: {last_plan.get('objetivo', 'datos')}"
            previous_sql = last_plan.get("executed_sql")
            
            if previous_sql:
                plan["meta"]["previous_sql"] = previous_sql
            else:
                plan["accion"] = "generar_informe"
            
            vis_new = _detect_visualization(texto)
            if vis_new:
                plan["meta"]["visualization"] = vis_new
            
            if plan["accion"] == "reutilizar_consulta":
                plan["confidence"] = 1.0 
                plan["confidence_reasons"] = ["reutilizacion_forzada_visualizacion"]
                return plan
        
        elif last_plan.get("accion") in ["generar_informe", "evaluar_colaborador"]:
            plan["accion"] = last_plan["accion"]
            plan["objetivo"] = last_plan["objetivo"]

    # ----------------------------------------------------
    # 1. Mapeo de Entidades (El Cerebro Semántico)
    # ----------------------------------------------------
    
    # Si schema_embeddings está vacío, no podemos ser semánticos
    if not schema_embeddings:
        logger.error("¡Schema_embeddings está vacío! El planificador semántico no puede funcionar.")
        raise LogicalError(
            "Error de configuración interna: El cerebro semántico (schema_embeddings) está vacío.",
            "Por favor, reinicia la aplicación o verifica el `schema_loader.py`."
        )

    # Extraer los conceptos a mapear
    concepts = _extract_concepts(texto)
    operation = concepts["operation"]
    
    # --- Mapear la Tabla Principal ---
    # (Usamos la heurística simple por ahora, se puede mejorar a semántica)
    tabla_detectada = concepts["main_table_term"]
    
    # Si es una intención de datos, y no hay tabla, buscamos semánticamente
    if tipo in ["generar_informe", "evaluar"] and not tabla_detectada:
        # Busca la tabla más relevante para toda la consulta
        table_match = _find_semantic_match(texto, schema_embeddings, entity_type="table")
        if table_match:
            tabla_detectada = table_match["key"] # ej. "evaluacion"
    
    if tabla_detectada:
        plan["entidades"]["tabla"] = tabla_detectada
    elif tipo in ["generar_informe", "evaluar"]:
         # Fallo Rápido: No hay tabla
         logger.warning(f"No se pudo detectar la tabla principal para: {texto}")
         raise LogicalError(
            f"No pude identificar una entidad principal (como 'evaluacion' o 'usuario') en tu solicitud.",
            "Por favor, reformula tu pregunta."
         )

    # --- Mapear la Métrica ("puntaje") ---
    mapped_metric = None
    if concepts["metric_term"]:
        metric_match = _find_semantic_match(concepts["metric_term"], schema_embeddings, entity_type="column")
        if metric_match:
            # key es "tabla.columna", la dividimos
            parts = metric_match["key"].split('.')
            if len(parts) == 2:
                mapped_metric = parts[1] # ej. "valor_respuesta"
            
    # --- Mapear el Agrupador ("facultad") ---
    mapped_group_by = None
    if concepts["group_by_term"]:
        group_by_match = _find_semantic_match(concepts["group_by_term"], schema_embeddings, entity_type="column")
        if group_by_match:
            # key es "tabla.columna"
            parts = group_by_match["key"].split('.')
            if len(parts) == 2:
                mapped_group_by = {"table": parts[0], "column": parts[1]} # ej. {"table": "equipo", "column": "nombre"}
        
        # ❗️ EL "POR DIOS" - Fallo Rápido y Limpio
        elif tipo in ["generar_informe", "evaluar"]: # Solo si era una consulta importante
            logger.warning(f"Término de 'group by' no mapeado: {concepts['group_by_term']}")
            suggestions = _get_groupable_suggestions(schema_embeddings) # ¡Ahora da sugerencias!
            raise LogicalError(
                f"El término '{concepts['group_by_term']}' que mencionaste no existe o no es claro en mi base de datos.",
                f"¿Quizás quisiste agrupar por una entidad que sí conozco, como: {', '.join(suggestions)}?"
            )

    # ----------------------------------------------------
    # 3. Construcción del Plan (Basado en el Mapeo Semántico)
    # ----------------------------------------------------
    
    plan["accion"] = "generar_informe" if tipo in ["generar_informe", "evaluar"] else "chat_general"
    plan["objetivo"] = "consultar y sintetizar información de la base de datos"
         
    # --- Filtros (Sin cambios, aún heurísticos) ---
    periodo_match = re.search(r"(20\d{2})(?:[-/](20\d{2}))?", texto)
    if periodo_match:
        plan["filtros"]["periodo"] = periodo_match.group(0)
    
    persona = entidades_raw.get("persona") or entidades_raw.get("nombre")
    if persona:
        if isinstance(persona, list):
            persona = persona[0]
        plan["filtros"]["persona"] = persona

    # --- Meta para query_generator ---
    meta: Dict[str, Any] = plan.get("meta", {}) 

    # Aggregations (Semántico)
    if operation and mapped_metric:
        meta["aggregations"] = [{"op": operation, "col": mapped_metric}]
    elif operation == "count":
        meta["aggregations"] = [{"op": "count", "col": "*"}]

    # Group_by (Semántico)
    if mapped_group_by:
        meta["group_by"] = [mapped_group_by] 

    # Visualización (Sin cambios)
    vis = _detect_visualization(texto)
    if vis:
        meta["visualization"] = vis
    elif operation: # Default a tabla si hay agregación
        meta["visualization"] = "table"

    # Límite (Sin cambios)
    if any(w in texto for w in ["todos", "completo", "todas"]):
        meta["limit"] = 1000
    elif "limit" not in meta:
        meta["limit"] = 100

    plan["meta"] = meta
    
    # ... (Confianza y Clarificación se pueden omitir por ahora) ...

    logger.debug("Plan semántico final generado: %s", json.dumps(plan, indent=2))
    return plan