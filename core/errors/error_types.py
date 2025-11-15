# core/errors/error_types.py (ACTUALIZADO)

from enum import Enum, auto
from dataclasses import dataclass
import re # Usaremos regex para mejor detección

# 1. Definimos los tipos de errores estandarizados
class ErrorType(Enum):
    SQL_SYNTAX_ERROR = auto()
    MISSING_COLUMN = auto()
    MISSING_TABLE = auto()
    CONNECTION_ERROR = auto()
    TOO_LONG_QUERY = auto()
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
    Ahora incluye detección para mensajes en inglés y español de psycopg2.
    """
    # Convertimos el mensaje de error a minúsculas para facilitar la búsqueda
    str_exc = str(exception).lower()
    tech_msg = f"{type(exception).__name__}: {str(exception)}" # Guardamos el mensaje original

    # --- Lógica de Clasificación ---

    # Errores de Conexión
    # (connection refused, connection timed out, no se pudo conectar, etc.)
    if re.search(r'connection timed out|connection refused|no se pudo conectar|failed to connect|timed out', str_exc):
        return ClassifiedError(ErrorType.CONNECTION_ERROR, tech_msg, exception)

    # Errores de sintaxis SQL
    # (syntax error, error de sintaxis)
    if re.search(r'syntax error|error de sintaxis', str_exc):
        return ClassifiedError(ErrorType.SQL_SYNTAX_ERROR, tech_msg, exception)
    
    # Columna inexistente
    # (column "X" does not exist, no existe la columna «X»)
    if re.search(r'column(.)*does not exist|no existe la columna|unknown column', str_exc):
        return ClassifiedError(ErrorType.MISSING_COLUMN, tech_msg, exception)

    # Tabla inexistente
    # (relation "X" does not exist, no existe la relación «X»)
    if re.search(r'relation(.)*does not exist|no existe la relación|table or view not found', str_exc):
        return ClassifiedError(ErrorType.MISSING_TABLE, tech_msg, exception)

    # Consulta demasiado larga o timeout de ejecución
    if re.search(r'query timeout|query took too long|statement timeout', str_exc):
        return ClassifiedError(ErrorType.TOO_LONG_QUERY, tech_msg, exception)

    # --- Error por Defecto ---
    # Si no coincide con nada, es un error desconocido
    return ClassifiedError(ErrorType.UNKNOWN_ERROR, tech_msg, exception)