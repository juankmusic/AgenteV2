# core/ai_core/dynamic_planner.py

# ============================================
# IMPORTACIONES - Librerías que necesitamos
# ============================================
import json  # Para trabajar con datos en formato JSON
import re  # Para buscar patrones de texto (expresiones regulares)
import logging  # Para registrar mensajes de lo que hace el programa
from typing import Dict, Any, List, Optional  # Para indicar qué tipo de datos usamos

# ============================================
# CONFIGURACIÓN DEL LOGGER
# ============================================
# El logger es como un cuaderno donde el programa escribe lo que va haciendo
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level="INFO")  # Nivel INFO = registra información general


# ============================================
# FUNCIÓN: Detectar operaciones matemáticas y métricas
# ============================================
def _detect_aggregation_and_metric(text: str) -> List[Dict[str, str]]:
    """
    Esta función busca en el texto si el usuario quiere hacer operaciones como:
    - Promedios (calcular la media)
    - Sumas (sumar valores)
    - Contar (cuántos hay)
    - Rankings (los mejores o peores)
    
    Parámetros:
        text: El texto que escribió el usuario
        
    Retorna:
        Una lista de diccionarios con la operación y la columna a usar
        Ejemplo: [{"op": "avg", "col": "nivel_contribucion"}]
    """
    aggs = []  # Lista donde guardaremos las operaciones encontradas
    txt = text.lower()  # Convertimos todo a minúsculas para buscar mejor
    
    # ¿El usuario quiere un PROMEDIO?
    if any(w in txt for w in ["promedio", "media", "avg", "mean"]):
        # Buscamos si menciona cosas que se pueden promediar (puntaje, nivel, salario, etc.)
        if re.search(r"(puntaje|nivel|score|salario|ingreso|ventas?)", txt):
            aggs.append({"op": "avg", "col": "nivel_contribucion"})
    
    # ¿El usuario quiere una SUMA?
    if any(w in txt for w in ["suma", "total", "sum"]):
        # Buscamos si menciona cosas que se pueden sumar (ventas, montos, cantidades, etc.)
        if re.search(r"(ventas|monto|cantidad|importe|score|puntaje)", txt):
            aggs.append({"op": "sum", "col": "monto"})
    
    # ¿El usuario quiere CONTAR algo?
    if any(w in txt for w in ["contar", "cantidad", "numero", "cuantos", "count"]):
        aggs.append({"op": "count", "col": "*"})
    
    # ¿El usuario quiere un RANKING (los mejores o peores)?
    if any(w in txt for w in ["top", "mejores", "peores", "rank"]):
        aggs.append({"op": "rank_hint", "col": "nivel_contribucion"})
    
    return aggs  # Devolvemos la lista de operaciones encontradas


# ============================================
# FUNCIÓN: Detectar qué tipo de gráfico quiere el usuario
# ============================================
def _detect_visualization(text: str) -> Optional[str]:
    """
    Esta función detecta si el usuario quiere ver los datos en algún formato visual como:
    - Gráfico de línea
    - Histograma
    - Gráfico de barras
    - Gráfico de pastel
    - Tabla
    
    Parámetros:
        text: El texto que escribió el usuario
        
    Retorna:
        El tipo de visualización (line, histogram, bar, pie, table) o None si no detecta ninguna
    """
    txt = text.lower()  # Convertimos a minúsculas para buscar mejor
    
    # ¿Quiere un gráfico de LÍNEA?
    if any(w in txt for w in ["gráfico", "grafico", "plot", "curve", "serie temporal", "trend", "tendencia", "evolución"]):
        return "line"
    
    # ¿Quiere un HISTOGRAMA?
    if any(w in txt for w in ["histograma", "distribución", "distribucion"]):
        return "histogram"
    
    # ¿Quiere un gráfico de BARRAS?
    if any(w in txt for w in ["barra", "bar", "barras", "bar chart"]):
        return "bar"
    
    # ¿Quiere un gráfico de PASTEL (circular)?
    if any(w in txt for w in ["pastel", "pie", "porcentaje", "circular"]):
        return "pie"
    
    # ¿Quiere una TABLA simple?
    if any(w in txt for w in ["tabla", "mostrar", "listar", "listado"]):
        return "table"
    
    return None  # Si no encontramos nada, devolvemos None


# ============================================
# FUNCIÓN: Verificar si necesitamos más información del usuario
# ============================================
def _need_clarification_for_plan(plan: Dict[str, Any]) -> Optional[str]:
    """
    Esta función revisa si el plan tiene toda la información necesaria.
    Si falta algo importante, devuelve una pregunta para pedirle más detalles al usuario.
    
    Parámetros:
        plan: El plan de acción que estamos construyendo
        
    Retorna:
        Una pregunta para el usuario o None si no hace falta nada
    """
    accion = plan.get("accion")  # ¿Qué acción vamos a hacer?
    filtros = plan.get("filtros", {})  # ¿Qué filtros tiene el plan?
    
    # Si vamos a evaluar un colaborador pero no sabemos QUIÉN, pedimos su nombre
    if accion == "evaluar_colaborador" and not filtros.get("persona"):
        return "¿A qué colaborador te refieres? Indica nombre o identificador."
    
    # Si vamos a generar un informe de usuarios/colaboradores sin filtros, preguntamos
    if accion == "generar_informe":
        tabla = plan.get("entidades", {}).get("tabla", "")
        if tabla in ("usuario", "colaborador") and not filtros:
            return "¿Quieres un informe para toda la organización o para un equipo/periodo específico?"
    
    return None  # No necesitamos más información


# ============================================
# FUNCIÓN PRINCIPAL: Crear un plan de acción
# ============================================
def plan_actions(intent_data: Dict[str, Any],
                 schema_semantic: Dict[str, Any],
                 schema_embeddings: Dict[str, Any],
                 last_plan: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    ESTA ES LA FUNCIÓN MÁS IMPORTANTE DEL ARCHIVO.
    
    Su trabajo es recibir lo que el usuario quiere hacer y crear un "plan de acción"
    con todos los detalles de cómo hacerlo.
    
    Es como cuando alguien te pide algo y tú piensas:
    "Ok, para hacer esto necesito X, Y y Z"
    
    Parámetros:
        intent_data: Lo que el usuario quiere hacer (su intención)
        schema_semantic: La estructura de la base de datos (qué tablas y columnas hay)
        schema_embeddings: Información sobre embeddings (para búsqueda semántica)
        last_plan: El plan anterior (si existe) para darle contexto
        
    Retorna:
        Un diccionario con el plan completo de acción
    """
    
    # ============================================
    # PASO 1: Extraer información básica
    # ============================================
    tipo = intent_data.get("tipo", "conversacion")  # ¿Qué tipo de petición es?
    entidades_raw = intent_data.get("entidades", {}) or {}  # Entidades detectadas
    texto = (intent_data.get("texto") or "").lower()  # El texto del usuario en minúsculas
    
    persona = None  # Por ahora no tenemos persona identificada
    
    # Creamos la estructura básica del plan
    plan: Dict[str, Any] = {
        "accion": "chat_general",  # Por defecto, es una conversación normal
        "objetivo": "mantener conversación",  # El objetivo inicial
        "entidades": {},  # Aquí van las entidades (tablas, columnas, etc.)
        "filtros": {},  # Aquí van los filtros (persona, fecha, equipo, etc.)
        "texto": texto,  # Guardamos el texto original
        "meta": {}  # Información extra (visualización, límites, etc.)
    }
    
    # ============================================
    # PASO 2: LÓGICA DE CONTINUACIÓN (Memoria de datos)
    # ============================================
    # Esta parte es SÚPER IMPORTANTE: permite que el usuario cambie solo la visualización
    # sin tener que repetir toda la consulta.
    # 
    # Ejemplo: El usuario pidió "datos del equipo omega" y luego dice "muéstralo en gráfico"
    # En lugar de pedir de nuevo los datos, reutilizamos la consulta anterior.
    
    # Detectamos si SOLO está cambiando la visualización (y no pidiendo datos nuevos)
    is_visualization_change_only = (_detect_visualization(texto) is not None) and \
                                 (tipo in ["conversacion", "consultar_datos", "generar_informe"]) and \
                                 (not any(w in texto for w in ["qué", "cuales", "cuantos", "dime", "saber"])) and \
                                 (not entidades_raw.get("area") and not entidades_raw.get("periodo")) and \
                                 (not _detect_aggregation_and_metric(texto))
    
    # Si tenemos un plan anterior Y el usuario está continuando la conversación
    if last_plan and (is_visualization_change_only or (tipo in ["generar_informe", "consultar_datos"] and not entidades_raw)):
        
        logger.info("Reutilizando contexto del plan anterior.")
        
        # Copiamos las entidades, filtros y metadata del plan anterior
        plan["entidades"] = last_plan.get("entidades", {})
        plan["filtros"] = last_plan.get("filtros", {})
        plan["meta"] = last_plan.get("meta", {})
        
        # 📌 LÓGICA DE REUTILIZACIÓN FORZADA (El cambio más importante)
        # Si SOLO está cambiando la visualización y ya teníamos una consulta previa
        if is_visualization_change_only and last_plan.get("accion") in ["generar_informe", "evaluar_colaborador"]:
            
            # Marcamos que vamos a REUTILIZAR la consulta anterior
            plan["accion"] = "reutilizar_consulta"
            plan["objetivo"] = f"actualizar visualización de la consulta anterior: {last_plan.get('objetivo', 'datos')}"
            
            # Guardamos el SQL anterior para usarlo directamente
            previous_sql = last_plan.get("executed_sql")
            if previous_sql:
                plan["meta"]["previous_sql"] = previous_sql
            else:
                # Si no hay SQL anterior, no podemos reutilizar; volvemos al flujo normal
                plan["accion"] = "generar_informe"
            
            # Actualizamos el tipo de visualización si detectamos uno nuevo
            vis_new = _detect_visualization(texto)
            if vis_new:
                plan["meta"]["visualization"] = vis_new
            
            # 🛑 Si logramos reutilizar, terminamos aquí y devolvemos el plan
            if plan["accion"] == "reutilizar_consulta":
                plan["confidence"] = 1.0  # Estamos 100% seguros
                plan["confidence_reasons"] = ["reutilizacion_forzada_visualizacion"]
                return plan
        
        # Si no es un cambio de visualización, mantenemos la acción del plan anterior
        elif last_plan.get("accion") in ["generar_informe", "evaluar_colaborador"]:
            plan["accion"] = last_plan["accion"]
            plan["objetivo"] = last_plan["objetivo"]

    # ============================================
    # PASO 3: Detectar TABLA usando el esquema semántico
    # ============================================
    # Buscamos qué tabla de la base de datos necesitamos usar
    
    tabla_detectada = None
    # Listamos todas las tablas disponibles
    candidate_tables = list(schema_semantic.get("tables", {}).keys()) + ["document_embeddings"]
    
    # Buscamos coincidencias entre las palabras clave de cada tabla y el texto del usuario
    for candidate in candidate_tables:
        kws = schema_semantic.get("tables", {}).get(candidate, {}).get("keywords", [])
        if any(word in texto for word in kws):
            tabla_detectada = candidate
            break
    
    # Si detectamos una tabla, la guardamos; si no, usamos la del plan anterior o "usuario" por defecto
    if tabla_detectada:
        plan["entidades"]["tabla"] = tabla_detectada
    else:
        tabla_detectada = plan["entidades"].get("tabla", "usuario")
    
    # ============================================
    # PASO 4: Detectar ATRIBUTOS (columnas) válidos
    # ============================================
    # Buscamos qué columnas específicas de la tabla necesitamos
    
    atributos = []
    # Obtenemos la lista de columnas de la tabla detectada
    columnas = schema_semantic.get("tables", {}).get(tabla_detectada, {}).get("columns", [])
    
    # Buscamos si el usuario mencionó alguna columna específica
    for col in columnas:
        if col.lower() in texto:
            atributos.append(col)
    
    # Si encontramos atributos, los guardamos; si no, usamos "*" (todas las columnas)
    if atributos:
        plan["entidades"]["atributos"] = atributos
    elif not plan["entidades"].get("atributos"):
        plan["entidades"]["atributos"] = ["*"]
    
    # ============================================
    # PASO 5: Detectar FILTROS (periodo, equipo, persona)
    # ============================================
    # Los filtros son condiciones para reducir los resultados
    # Ejemplo: "solo del año 2024" o "del equipo omega"
    
    # Buscar PERIODO (años) en el texto usando expresiones regulares
    # Ejemplo: "2024" o "2023-2024"
    periodo_match = re.search(r"(20\d{2})(?:[-/](20\d{2}))?", texto)
    if periodo_match:
        plan["filtros"]["periodo"] = periodo_match.group(0)
    
    # Lista de equipos válidos que reconocemos
    equipos_validos = ["omega", "alfa", "beta", "delta"]
    
    # Buscar EQUIPO en el texto
    # Busca el patrón "equipo <nombre>"
    equipo_match = re.search(r"\bequipo\s+([a-zA-Z0-9_\-]+)\b", texto)
    if equipo_match:
        posible_equipo = equipo_match.group(1).lower()
        
        # Solo lo aceptamos si está en la lista de equipos válidos
        if posible_equipo in equipos_validos:
            plan["filtros"]["equipo"] = posible_equipo
        else:
            # Si no es válido, lo ignoramos y registramos un mensaje
            logger.info(f"Ignorando coincidencia de equipo no válida: '{posible_equipo}'")
    
    # Si tenemos una PERSONA identificada, la agregamos al filtro
    if persona:
        if isinstance(persona, list):  # Si viene como lista, tomamos el primer elemento
            persona = persona[0]
        plan["filtros"]["persona"] = persona
    
    # ============================================
    # PASO 6: Determinar la ACCIÓN principal
    # ============================================
    # Dependiendo del tipo de petición, definimos qué acción tomar
    
    if tipo in ["generar_informe", "consultar_datos", "consultar"]:
        plan["accion"] = "generar_informe"
        plan["objetivo"] = "consultar y sintetizar información de la base de datos"
    elif tipo == "evaluar":
        plan["accion"] = "evaluar_colaborador"
        plan["objetivo"] = "analizar métricas de desempeño de un colaborador"
    elif tipo == "guardar_resultado":
        plan["accion"] = "guardar_resultado"
        plan["objetivo"] = "almacenar información procesada"
    
    # ============================================
    # PASO 7: Configurar METADATA extra
    # ============================================
    # La metadata contiene información adicional sobre cómo ejecutar la consulta
    
    meta: Dict[str, Any] = plan.get("meta", {})  # Reutilizar meta si existe
    
    # Detectar AGREGACIONES (promedios, sumas, conteos)
    aggregations = _detect_aggregation_and_metric(texto)
    if aggregations:
        # Validamos que las columnas existan en el esquema
        valid_aggs = [agg for agg in aggregations if agg["col"] in columnas or agg["col"] == "*"]
        if valid_aggs:
            meta["aggregations"] = valid_aggs
    
    # Detectar GROUP BY (agrupar por algún campo)
    # Ejemplo: "por equipo", "por año"
    group_by = []
    for col in columnas:
        if f"por {col.lower()}" in texto:
            group_by.append(col)
    if group_by:
        meta["group_by"] = group_by
    
    # Detectar tipo de VISUALIZACIÓN (siempre se actualiza si se detecta una nueva)
    vis = _detect_visualization(texto)
    if vis:
        meta["visualization"] = vis
    
    # Detectar LÍMITE de resultados
    # Si el usuario dice "todos" o "completo", ponemos límite alto (1000)
    if any(w in texto for w in ["todos", "completo", "todas"]):
        meta["limit"] = 1000
    # Si no especifica, usamos el límite anterior o el default (100)
    elif "limit" not in meta:
        meta["limit"] = 100
    
    # ============================================
    # PASO 8: Configuración para EMBEDDINGS (búsqueda semántica)
    # ============================================
    # Si estamos buscando en documentos usando embeddings (vectores)
    
    if tabla_detectada == "document_embeddings":
        meta["embedding_query"] = texto  # El texto a buscar
        meta["top_k"] = 10  # Traer los 10 resultados más relevantes
        embedding_fields = schema_embeddings.get("vector_fields", [])
        if embedding_fields:
            meta["embedding_field"] = embedding_fields[0]  # Campo del vector
    
    # Guardamos toda la metadata en el plan
    plan["meta"] = meta
    
    # ============================================
    # PASO 9: Calcular CONFIANZA del plan
    # ============================================
    # Calculamos qué tan seguros estamos de que el plan es correcto
    # Esto ayuda a saber si necesitamos pedir aclaraciones al usuario
    
    score = 0.0  # Puntuación inicial
    reasons: List[str] = []  # Lista de razones por las que sumamos puntos
    
    # Sumamos puntos por cada cosa que detectamos correctamente
    if persona:
        score += 0.25
        reasons.append("persona_detectada")
    if periodo_match:
        score += 0.15
        reasons.append("periodo_detectado")
    if tabla_detectada and tabla_detectada != "usuario":
        score += 0.2
        reasons.append(f"tabla_detectada:{tabla_detectada}")
    if aggregations:
        score += 0.1
        reasons.append("aggregation_hint")
    if vis:
        score += 0.1
        reasons.append(f"visualization_suggested:{vis}")
    
    # Bonificación si reutilizamos contexto del plan anterior (solo si no detectamos tabla nueva)
    if last_plan and not tabla_detectada:
        score += 0.15
        reasons.append("contexto_reutilizado")
    
    # La confianza final es el score (máximo 1.0 = 100%)
    confidence = min(round(score, 3), 1.0)
    plan["confidence"] = confidence
    plan["confidence_reasons"] = reasons
    
    # ============================================
    # PASO 10: Verificar si necesitamos ACLARACIÓN
    # ============================================
    # Si falta información importante, marcamos que necesitamos preguntar
    
    clarify_q = _need_clarification_for_plan(plan)
    if clarify_q:
        plan["clarify"] = True  # Sí necesitamos aclaración
        plan["clarify_question"] = clarify_q  # La pregunta a hacer
    else:
        plan["clarify"] = False  # No necesitamos aclaración, tenemos todo
    
    # ============================================
    # FINAL: Devolver el plan completo
    # ============================================
    return plan