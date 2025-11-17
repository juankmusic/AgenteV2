# core/ai_core/structured_action_extractor.py
import re
from typing import Optional, Dict, Any, List

# Mapeo de keywords a tablas (extensible)
TABLE_KEYWORDS = {
    "transversal": "competencia_transversales",
    "transversales": "competencia_transversales",
    "docente": "competencia_docente",
    "docentes": "competencia_docente"
}

# Regex básicos
EMAIL_REGEX = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
ID_REGEX = r"\b\d{3,10}\b"
# Nombres: hasta 2 palabras, letras y vocales con tilde, aceptamos punto/guion opcional en nombres
NAME_WORD = r"[A-Za-zÁÉÍÓÚÑáéíóúñ]+(?:[.\-]?[A-Za-zÁÉÍÓÚÑáéíóúñ]+)?"
NAME_REGEX = rf"\b{NAME_WORD}(?:\s+{NAME_WORD})?\b"

# Separadores para dividir múltiples usuarios en la frase
SPLIT_SEPARATORS = r",|\band\b|\by\b|\btambién\b|\bademás\b|;"

# Patrones para buscar "usuario ... (con|nivel|iluo) N"
PAIR_PATTERN = re.compile(
    rf"(?P<user>{EMAIL_REGEX}|{ID_REGEX}|{NAME_REGEX})\s*(?:,|\(|\-|:)?\s*(?:con\s*)?(?:nivel|iluo)\s*[:\-]?\s*(?P<iluo>\d)\b",
    flags=re.IGNORECASE
)

# Patrones para detectar ILUO independiente si viene antes/ después
ILUO_PATTERN = re.compile(r"(?:iluo|nivel)\s*[:\-]?\s*(\d)\b", flags=re.IGNORECASE)


def _find_table(text: str) -> Optional[str]:
    t = text.lower()
    for kw, table in TABLE_KEYWORDS.items():
        if kw in t:
            return table
    return None


def _split_to_segments(text: str) -> List[str]:
    """
    Divide correctamente las frases con múltiples usuarios.
    Evita doble normalización y realiza un único split robusto.
    """
    # Usamos un único patrón que captura todos los conectores
    segments = re.split(r"\s*(?:,|;|\by\b|\band\b|\btambién\b|\bademás\b)\s*", text, flags=re.IGNORECASE)

    # Limpiar resultados vacíos o conectores solos
    cleaned = [s.strip() for s in segments if s.strip()]
    return cleaned



def _extract_from_segment(seg: str) -> Optional[Dict[str, Any]]:
    """
    Extrae (usuario_identificador, iluo) de un segmento.
    Prioridad de identificación:
      - email
      - id numérico
      - nombre (1 o 2 palabras)
    """
    seg = seg.strip()

    # 1) Buscar pares con pattern robusto (usuario seguido de 'nivel/iluo N')
    m_pair = PAIR_PATTERN.search(seg)
    if m_pair:
        user = m_pair.group("user").strip()
        iluo = int(m_pair.group("iluo"))
        return {"identificador": user, "iluo": iluo}

    # 2) Si no hay par directo, intentar extraer email primero
    m_email = re.search(EMAIL_REGEX, seg)
    if m_email:
        user = m_email.group(0)
        # buscar iluo en el segmento
        m_il = ILUO_PATTERN.search(seg)
        if m_il:
            return {"identificador": user, "iluo": int(m_il.group(1))}
        return {"identificador": user, "iluo": None}

    # 3) Buscar id numérico
    m_id = re.search(ID_REGEX, seg)
    if m_id:
        user = m_id.group(0)
        m_il = ILUO_PATTERN.search(seg)
        if m_il:
            return {"identificador": user, "iluo": int(m_il.group(1))}
        return {"identificador": user, "iluo": None}

    # 4) Buscar nombre (1 o 2 palabras)
    # Intentamos encontrar la primer match que no sea palabra reservada
    m_name = re.search(NAME_REGEX, seg)
    if m_name:
        # Evitar capturar palabras como "evaluar", "competencias", etc.
        candidate = m_name.group(0).strip()
        # si el candidato es muy genérico, descartar
        if candidate.lower() not in {"evaluar", "evaluar:", "competencias", "competencias:"}:
            m_il = ILUO_PATTERN.search(seg)
            if m_il:
                return {"identificador": candidate, "iluo": int(m_il.group(1))}
            return {"identificador": candidate, "iluo": None}

    return None


def _fill_missing_iluo_by_context(segments: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
    """
    Si algún segmento no incluye iluo explícito, tratamos de asignarle un iluo
    global de contexto si existe (por ejemplo 'con nivel 2' al final de la frase),
    o dejamos None para que el caller pida aclaración.
    """
    # buscar posibles iluo globales (ej: "con nivel 2 para todos")
    global_il_match = ILUO_PATTERN.search(text)
    global_il = int(global_il_match.group(1)) if global_il_match else None

    for seg in segments:
        if seg["iluo"] is None and global_il is not None:
            seg["iluo"] = global_il
    return segments


def extract_structured_action(text: str) -> Optional[Dict[str, Any]]:
    """
    Extrae acción estructurada. Soporta:
      - Un solo usuario con su ILUO
      - Múltiples usuarios separados por comas/ conectores con ILUO individual
    Salida:
    - Si se detecta 1 usuario: action = "insert_evaluation", keys: target_table, usuario_identificador, iluo
    - Si se detectan varios: action = "insert_evaluation_multi", keys: target_table, usuarios: [{identificador, iluo}, ...]
    """
    if not text or not text.strip():
        return None

    table = _find_table(text)
    if not table:
        return None

    # 1) Dividir la parte relevante de la oración en segmentos
    segments = _split_to_segments(text)

    extracted = []
    for seg in segments:
        res = _extract_from_segment(seg)
        if res:
            extracted.append(res)

    # Si no extrajimos nada con split, intentar extraer pares en toda la frase
    if not extracted:
        # intentar pares globales
        matches = PAIR_PATTERN.findall(text)
        for m in matches:
            user = m[0].strip()
            iluo = int(m[1])
            extracted.append({"identificador": user, "iluo": iluo})

    if not extracted:
        return None

    # Rellenar iluos faltantes por un iluo global si existe
    extracted = _fill_missing_iluo_by_context(extracted, text)

    # Normalizar: quitar duplicados por identificador (manteniendo el primero)
    seen = set()
    normalized = []
    for e in extracted:
        key = e["identificador"].lower()
        if key not in seen:
            seen.add(key)
            normalized.append({"identificador": e["identificador"], "iluo": e["iluo"]})
    extracted = normalized

    # Si solo hay uno, devolver la forma simple (retrocompatibilidad)
    if len(extracted) == 1:
        u = extracted[0]
        return {
            "action": "insert_evaluation",
            "target_table": table,
            "usuario_identificador": u["identificador"],
            "iluo": u["iluo"]
        }

    # Si hay varios, devolver list
    return {
        "action": "insert_evaluation_multi",
        "target_table": table,
        "usuarios": extracted  # lista de {identificador, iluo}
    }