from core.errors.error_types import classify_error, ErrorType, ClassifiedError
from core.errors.error_memory import ErrorMemory
from typing import List

# --- Instancia única de la Memoria ---
error_log = ErrorMemory(max_size=50)

# --- Diccionarios de Mensajes (ACTUALIZADOS) ---

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
    1. Clasifica el error.
    2. Lo guarda en la memoria.
    3. Devuelve un mensaje amigable para el usuario.
    """
    
    # 1. Clasificar el error
    classified_error = classify_error(exception)
    
    # 2. Guardar en la memoria (usando la instancia global)
    error_log.add(classified_error)
    
    # 3. Generar el mensaje para el usuario
    explanation = USER_EXPLANATIONS.get(
        classified_error.type, 
        USER_EXPLANATIONS[ErrorType.UNKNOWN_ERROR]
    )
    
    solution = USER_SOLUTIONS.get(
        classified_error.type,
        USER_SOLUTIONS[ErrorType.UNKNOWN_ERROR]
    )
    
    # Combinamos la explicación y la alternativa
    return f"{explanation} {solution}"


# --- Funciones de utilidad ---

def get_recent_errors(n: int = 5) -> List[ClassifiedError]:
    """Expone los errores recientes al resto de la aplicación."""
    return error_log.get_last(n)

def get_full_error_history() -> List[ClassifiedError]:
    """Expone todo el historial de errores."""
    return error_log.get_all()


# --- Función auxiliar para feedback de aprendizaje ---
def get_error_patterns_summary() -> dict:
    """
    Analiza los errores recurrentes para mejorar el sistema.
    Útil para implementar "aprendizaje" del agente.
    """
    all_errors = error_log.get_all()
    
    # Contar frecuencia de cada tipo
    error_counts = {}
    for error in all_errors:
        error_type = error.type.name
        error_counts[error_type] = error_counts.get(error_type, 0) + 1
    
    # Identificar patrones comunes en mensajes técnicos
    common_patterns = {}
    for error in all_errors:
        # Extraer palabras clave del mensaje técnico
        msg_lower = error.technical_message.lower()
        
        # Buscar tablas mencionadas
        if 'tabla' in msg_lower or 'table' in msg_lower:
            key = "problemas_con_tablas"
            common_patterns[key] = common_patterns.get(key, 0) + 1
        
        # Buscar columnas mencionadas
        if 'columna' in msg_lower or 'column' in msg_lower:
            key = "problemas_con_columnas"
            common_patterns[key] = common_patterns.get(key, 0) + 1
    
    return {
        "total_errors": len(all_errors),
        "error_counts": error_counts,
        "common_patterns": common_patterns,
        "most_common_error": max(error_counts.items(), key=lambda x: x[1])[0] if error_counts else None
    }