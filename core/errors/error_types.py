# core/errors/error_types.py

from enum import Enum, auto
from dataclasses import dataclass
import re

# ---------------------------------------------------------
# Esta clase define todos los tipos de errores que nuestro
# sistema puede reconocer. Es como una lista de "categorías".
# ---------------------------------------------------------
class ErrorType(Enum):
    SQL_SYNTAX_ERROR = auto()      # Error cuando la consulta SQL está mal escrita
    MISSING_COLUMN = auto()        # Falta una columna en la base de datos
    MISSING_TABLE = auto()         # Falta una tabla en la base de datos
    CONNECTION_ERROR = auto()      # No se pudo conectar a la base de datos
    TOO_LONG_QUERY = auto()        # La consulta tardó demasiado tiempo
    SEMANTIC_ERROR = auto()        # Hay un problema lógico en la consulta
    TYPE_MISMATCH = auto()         # Se usó un tipo de dato incorrecto
    EMPTY_DATASET = auto()         # La consulta no devolvió datos (NUEVO)
    INVALID_CHART_DATA = auto()    # No se puede hacer un gráfico con los datos (NUEVO)
    UNKNOWN_ERROR = auto()         # Cualquier error que no encaje en los anteriores

# ---------------------------------------------------------
# Este dataclass sirve para guardar todos los detalles
# importantes de un error en un solo “paquetico”.
# ---------------------------------------------------------
@dataclass
class ClassifiedError:
    """Un objeto que guarda:
       - el tipo de error,
       - el mensaje técnico,
       - y la excepción real que se produjo."""
    type: ErrorType                # Categoría del error
    technical_message: str         # Mensaje técnico del error
    original_exception: Exception  # La excepción original de Python

# ---------------------------------------------------------
# Esta función mira un error y decide qué tipo de error es.
# Analiza el texto del error y lo clasifica.
# ---------------------------------------------------------
def classify_error(exception: Exception) -> ClassifiedError:
    """
    Lee una excepción y decide a qué tipo pertenece.
    Luego devuelve un objeto ClassifiedError con la info.
    """

    # Convertimos el error en texto, todo en minúsculas para comparar más fácil
    str_exc = str(exception).lower()

    # Mensaje técnico completo (incluye el nombre de la excepción)
    tech_msg = f"{type(exception).__name__}: {str(exception)}"

    # ---------------------------------------------------------
    # LÓGICA DE CLASIFICACIÓN
    # (El orden de los if es importante)
    # ---------------------------------------------------------

    # 1. Errores de tipo de dato incompatible (los más específicos)
    # Se detectan con palabras que indican que no se puede hacer
    # cierta operación con ciertos tipos de datos.
    if re.search(r'no existe la función|function.*does not exist|undefined function|cannot.*aggregate|type mismatch|invalid input syntax for type', str_exc):
        return ClassifiedError(ErrorType.TYPE_MISMATCH, tech_msg, exception)

    # 2. Problemas de conexión con la base de datos
    if re.search(r'connection timed out|connection refused|no se pudo conectar|failed to connect|timed out', str_exc):
        return ClassifiedError(ErrorType.CONNECTION_ERROR, tech_msg, exception)

    # 3. Errores de sintaxis (cuando el SQL está mal escrito)
    if re.search(r'syntax error|error de sintaxis', str_exc):
        return ClassifiedError(ErrorType.SQL_SYNTAX_ERROR, tech_msg, exception)

    # 4. Error cuando una columna no existe
    if re.search(r'column(.)*does not exist|no existe la columna|unknown column', str_exc):
        return ClassifiedError(ErrorType.MISSING_COLUMN, tech_msg, exception)

    # 5. Error cuando una tabla no existe
    if re.search(r'relation(.)*does not exist|no existe la relación|table or view not found', str_exc):
        return ClassifiedError(ErrorType.MISSING_TABLE, tech_msg, exception)

    # 6. Consulta demasiado lenta o que se canceló por tiempo
    if re.search(r'query timeout|query took too long|statement timeout', str_exc):
        return ClassifiedError(ErrorType.TOO_LONG_QUERY, tech_msg, exception)

    # 7. Errores lógicos o de validación (problemas de significado)
    if re.search(r'semantic|validation|join sin on|error semántico', str_exc):
        return ClassifiedError(ErrorType.SEMANTIC_ERROR, tech_msg, exception)

    # 8. Si no coincide con nada, lo consideramos un error desconocido
    return ClassifiedError(ErrorType.UNKNOWN_ERROR, tech_msg, exception)
