from core.errors.error_types import classify_error, ErrorType, ClassifiedError
from core.errors.error_memory import ErrorMemory
from typing import List

# --- Instancia única de la Memoria ---
# Esta línea crea una "cajita" donde vamos a guardar los últimos errores.
# Solo guardamos un máximo de 50 para no llenar demasiado la memoria.
error_log = ErrorMemory(max_size=50)

# --- Diccionarios de Mensajes (ACTUALIZADOS) ---
# Estos diccionarios contienen mensajes que se muestran al usuario cuando ocurre un error.
# La idea es traducir errores técnicos complicados a explicaciones fáciles de entender.


# Explicación simple de cada error para el usuario
USER_EXPLANATIONS = {
    ErrorType.SQL_SYNTAX_ERROR: "Tuve un problema al estructurar la consulta a la base de datos.",
    ErrorType.MISSING_COLUMN: "Parece que intenté acceder a una columna o dato que no existe en la base de datos.",
    ErrorType.MISSING_TABLE: "Intenté consultar una tabla que no parece existir.",
    ErrorType.CONNECTION_ERROR: "Lo siento, no pude establecer conexión con la base de datos en este momento.",
    ErrorType.TOO_LONG_QUERY: "La solicitud que intenté ejecutar tardó demasiado tiempo y tuvo que ser cancelada.",
    ErrorType.SEMANTIC_ERROR: "Detecté un problema lógico en mi interpretación de tu pregunta.",
    ErrorType.TYPE_MISMATCH: "Intenté realizar una operación matemática sobre un tipo de dato incompatible (como sumar texto).",
    ErrorType.UNKNOWN_ERROR: "Ocurrió un error inesperado mientras procesaba tu solicitud.",
    ErrorType.EMPTY_DATASET: "La consulta se ejecutó correctamente, pero no devolvió ningún resultado.",
    ErrorType.INVALID_CHART_DATA: "Los datos obtenidos no son adecuados para generar el tipo de gráfico solicitado.",
    ErrorType.UNKNOWN_ERROR: "Ocurrió un error inesperado mientras procesaba tu solicitud."
}

# Posibles soluciones o pasos sugeridos para el usuario
USER_SOLUTIONS = {
    ErrorType.SQL_SYNTAX_ERROR: "Estoy aprendiendo de esto. ¿Podrías intentar reformular tu pregunta? A veces, usar palabras diferentes ayuda.",
    ErrorType.MISSING_COLUMN: "Estoy revisando mi conocimiento de la base de datos. ¿Podrías verificar si los nombres o filtros que usaste son correctos?",
    ErrorType.MISSING_TABLE: "Estoy actualizando mi mapa de la base de datos. Por favor, intenta de nuevo.",
    ErrorType.CONNECTION_ERROR: "Por favor, un administrador debería revisar que la base de datos esté funcionando correctamente. Puedes intentarlo de nuevo en unos minutos.",
    ErrorType.TOO_LONG_QUERY: "Mi lógica interna falló. ¿Podrías intentar hacer una pregunta más específica o acotar un poco más el rango de búsqueda?",
    ErrorType.SEMANTIC_ERROR: "La operación que intentaba realizar no tiene sentido con el tipo de datos. He incluido más detalles arriba.",
    ErrorType.TYPE_MISMATCH: "Por ejemplo, no se puede sumar nombres o promediar texto. ¿Querías contar registros en su lugar? (usa COUNT)",
    ErrorType.UNKNOWN_ERROR: "He guardado los detalles técnicos para mi revisión. Por favor, intenta la solicitud de nuevo, quizás de una forma un poco diferente.",
    ErrorType.EMPTY_DATASET: "Intenta ajustar los filtros o parámetros de búsqueda. Por ejemplo, verifica rangos de fechas, IDs o nombres.",
    ErrorType.INVALID_CHART_DATA: "Por ejemplo, los gráficos de líneas requieren datos temporales o secuenciales, y los gráficos circulares necesitan categorías con valores numéricos. ¿Quieres ver los datos en formato de tabla?",
    ErrorType.UNKNOWN_ERROR: "He guardado los detalles técnicos para mi revisión. Por favor, intenta la solicitud de nuevo, quizás de una forma un poco diferente."
}


def handle_error(exception: Exception) -> str:
    """
    Función principal para manejar una excepción.

    Esta función hace tres cosas:
    1. Clasifica el error (decide qué tipo de error fue).
    2. Guarda el error en la memoria para poder revisarlo después.
    3. Devuelve un mensaje amigable para el usuario.
    """

    # 1. Clasificar el error usando la función que sabe identificar tipos de errores.
    classified_error = classify_error(exception)

    # 2. Guardar ese error en la memoria global que creamos arriba.
    # Esto sirve para llevar un historial de fallas.
    error_log.add(classified_error)

    # 3. Elegir el mensaje de explicación según el tipo de error.
    explanation = USER_EXPLANATIONS.get(
        classified_error.type,
        USER_EXPLANATIONS[ErrorType.UNKNOWN_ERROR]  # si no existe, usamos "error desconocido"
    )

    # Elegir la posible solución o recomendación para el usuario.
    solution = USER_SOLUTIONS.get(
        classified_error.type,
        USER_SOLUTIONS[ErrorType.UNKNOWN_ERROR]
    )

    # Devolver ambos mensajes juntos (explicación + solución).
    return f"{explanation} {solution}"


# --- Funciones de utilidad ---
# Estas funciones permiten consultar la memoria de errores desde otras partes del sistema.

def get_recent_errors(n: int = 5) -> List[ClassifiedError]:
    """Devuelve los últimos N errores guardados."""
    return error_log.get_last(n)


def get_full_error_history() -> List[ClassifiedError]:
    """Devuelve la lista completa de todos los errores guardados."""
    return error_log.get_all()


# --- Función auxiliar para feedback de aprendizaje ---
def get_error_patterns_summary() -> dict:
    """
    Esta función analiza todos los errores guardados para encontrar patrones,
    como cuáles son los más comunes o qué palabras clave aparecen.

    Esto sirve para mejorar el sistema en el futuro.
    """
    # Obtener todos los errores guardados.
    all_errors = error_log.get_all()

    # Contador para saber cuántas veces ocurre cada tipo de error.
    error_counts = {}
    for error in all_errors:
        error_type = error.type.name
        error_counts[error_type] = error_counts.get(error_type, 0) + 1

    # Diccionario para detectar patrones dentro de los mensajes técnicos.
    common_patterns = {}
    for error in all_errors:
        msg_lower = error.technical_message.lower()  # poner mensaje en minúsculas para buscar palabras

        # Si el mensaje menciona tablas
        if 'tabla' in msg_lower or 'table' in msg_lower:
            key = "problemas_con_tablas"
            common_patterns[key] = common_patterns.get(key, 0) + 1

        # Si menciona columnas
        if 'columna' in msg_lower or 'column' in msg_lower:
            key = "problemas_con_columnas"
            common_patterns[key] = common_patterns.get(key, 0) + 1

    # Devolver el resumen completo
    return {
        "total_errors": len(all_errors),
        "error_counts": error_counts,
        "common_patterns": common_patterns,
        # Si no hay errores, devolvemos None
        "most_common_error": max(error_counts.items(), key=lambda x: x[1])[0] if error_counts else None
    }
