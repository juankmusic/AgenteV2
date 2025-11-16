# core/errors/error_types.py

from enum import Enum, auto
from dataclasses import dataclass
import re

class ErrorType(Enum):
    SQL_SYNTAX_ERROR = auto()
    MISSING_COLUMN = auto()
    MISSING_TABLE = auto()
    CONNECTION_ERROR = auto()
    TOO_LONG_QUERY = auto()
    SEMANTIC_ERROR = auto()
    TYPE_MISMATCH = auto()
    EMPTY_DATASET = auto()        # <<< NUEVO
    INVALID_CHART_DATA = auto()   # <<< NUEVO
    UNKNOWN_ERROR = auto()

@dataclass
class ClassifiedError:
    """Un objeto estandarizado para guardar la información del error."""
    type: ErrorType
    technical_message: str
    original_exception: Exception

def classify_error(exception: Exception) -> ClassifiedError:
    """
    Analiza una excepción y la clasifica en un ErrorType estandarizado.
    """
    str_exc = str(exception).lower()
    tech_msg = f"{type(exception).__name__}: {str(exception)}"

    # --- Lógica de Clasificación (ORDEN IMPORTA) ---

    # Errores de tipo incompatible (PRIMERO - más específico)
    if re.search(r'no existe la función|function.*does not exist|undefined function|cannot.*aggregate|type mismatch|invalid input syntax for type', str_exc):
        return ClassifiedError(ErrorType.TYPE_MISMATCH, tech_msg, exception)

    # Errores de Conexión
    if re.search(r'connection timed out|connection refused|no se pudo conectar|failed to connect|timed out', str_exc):
        return ClassifiedError(ErrorType.CONNECTION_ERROR, tech_msg, exception)

    # Errores de sintaxis SQL
    if re.search(r'syntax error|error de sintaxis', str_exc):
        return ClassifiedError(ErrorType.SQL_SYNTAX_ERROR, tech_msg, exception)
    
    # Columna inexistente
    if re.search(r'column(.)*does not exist|no existe la columna|unknown column', str_exc):
        return ClassifiedError(ErrorType.MISSING_COLUMN, tech_msg, exception)

    # Tabla inexistente
    if re.search(r'relation(.)*does not exist|no existe la relación|table or view not found', str_exc):
        return ClassifiedError(ErrorType.MISSING_TABLE, tech_msg, exception)

    # Consulta demasiado larga
    if re.search(r'query timeout|query took too long|statement timeout', str_exc):
        return ClassifiedError(ErrorType.TOO_LONG_QUERY, tech_msg, exception)
    
    # Errores semánticos generados por el propio validador
    if re.search(r'semantic|validation|join sin on|error semántico', str_exc):
        return ClassifiedError(ErrorType.SEMANTIC_ERROR, tech_msg, exception)

    # Error por defecto
    return ClassifiedError(ErrorType.UNKNOWN_ERROR, tech_msg, exception)