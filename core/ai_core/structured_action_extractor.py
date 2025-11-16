# core/ai_core/structured_action_extractor.py
import re
from typing import Optional, Dict, Any

# Mapeo de keywords a tablas (extensible)
TABLE_KEYWORDS = {
    "transversal": "competencia_transversales",
    "transversales": "competencia_transversales",
    "docente": "competencia_docente",
    "docentes": "competencia_docente"
}

def _find_table(text: str) -> Optional[str]:
    t = text.lower()
    for kw, table in TABLE_KEYWORDS.items():
        if kw in t:
            return table
    return None

def _find_iluo(text: str) -> Optional[int]:
    t = text.lower()
    m = re.search(r"(iluo|nivel)\s*[:\-]?\s*(\d+)\b", t)
    if m:
        try:
            return int(m.group(2))
        except:
            return None
    # catch "con nivel 3", "nivel 3"
    m2 = re.search(r"\bnivel\s+(\d)\b", t)
    if m2:
        return int(m2.group(1))
    return None

def _find_user_identifier(text: str) -> Optional[str]:
    """
    Intenta capturar un correo, un id numérico o un nombre (lo que venga).
    Prioridad: correo > id numeric > nombre completo (hasta 'con'/'en'/'con nivel')
    """
    t = text.strip()
    # correo
    m_email = re.search(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", t)
    if m_email:
        return m_email.group(1).strip()

    # id numérico (ej: usuario 123, id 45)
    m_id = re.search(r"\b(?:id|usuario|uid)\s*[:#]?\s*(\d{1,8})\b", t.lower())
    if m_id:
        return m_id.group(1).strip()

    # nombre heurístico: "evaluar a Juan Perez" o "evaluar Juan Perez"
    m_name = re.search(r"(?:evaluar|calificar|asignar|registrar)\s+(?:a\s+)?([A-ZÁÉÍÓÚÑa-záéíóúñ0-9\.\_\- ]{3,80})", t, re.IGNORECASE)
    if m_name:
        name = m_name.group(1).strip()
        # truncate if trailing keywords
        name = re.split(r"\b(en|con|con nivel|nivel|en competencias|en competencia)\b", name, flags=re.IGNORECASE)[0].strip()
        return name

    return None

def extract_structured_action(text: str) -> Optional[Dict[str, Any]]:
    """
    Si detecta una intención clara de 'insertar evaluación', devuelve:
    {
        "action": "insert_evaluation",
        "target_table": "competencia_transversales" | "competencia_docente",
        "usuario_identificador": "correo|id|nombre",
        "iluo": 3
    }
    """
    table = _find_table(text)
    iluo = _find_iluo(text)
    user = _find_user_identifier(text)

    if table and iluo and user:
        return {
            "action": "insert_evaluation",
            "target_table": table,
            "usuario_identificador": user,
            "iluo": iluo
        }
    return None
