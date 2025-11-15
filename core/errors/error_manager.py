from core.errors.error_types import classify_error, ErrorType, ClassifiedError
from core.errors.error_memory import ErrorMemory
from typing import List

# --- Instancia única de la Memoria ---
# Creamos una instancia global (Singleton en la práctica) que será
# compartida por todos los módulos que importen error_manager.
error_log = ErrorMemory(max_size=50)

# --- Diccionarios de Mensajes ---
# Separamos la "Explicación" (Por qué pasó) de la "Solución" (Qué hacer)
# Esto hace que sea más fácil de mantener.

USER_EXPLANATIONS = {
    ErrorType.SQL_SYNTAX_ERROR: "Tuve un problema al estructurar la consulta a la base de datos.",
    ErrorType.MISSING_COLUMN: "Parece que intenté acceder a una columna o dato que no existe en la base de datos.",
    ErrorType.MISSING_TABLE: "Intenté consultar una tabla que no parece existir.", # <<< AÑADIDO
    ErrorType.CONNECTION_ERROR: "Lo siento, no pude establecer conexión con la base de datos en este momento.",
    ErrorType.TOO_LONG_QUERY: "La solicitud que intenté ejecutar tardó demasiado tiempo y tuvo que ser cancelada.",
    ErrorType.UNKNOWN_ERROR: "Ocurrió un error inesperado mientras procesaba tu solicitud."
}

USER_SOLUTIONS = {
    ErrorType.SQL_SYNTAX_ERROR: "Estoy aprendiendo de esto. ¿Podrías intentar reformular tu pregunta? A veces, usar palabras diferentes ayuda.",
    ErrorType.MISSING_COLUMN: "Estoy revisando mi conocimiento de la base de datos. ¿Podrías verificar si los nombres o filtros que usaste son correctos?",
    ErrorType.MISSING_TABLE: "Estoy actualizando mi mapa de la base de datos. Por favor, intenta de nuevo.", # <<< AÑADIDO
    ErrorType.CONNECTION_ERROR: "Por favor, un administrador debería revisar que la base de datos esté funcionando correctamente. Puedes intentarlo de nuevo en unos minutos.",
    ErrorType.TOO_LONG_QUERY: "Mi lógica interna falló. ¿Podrías intentar hacer una pregunta más específica o acotar un poco más el rango de búsqueda?",
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
# Estas funciones permiten que otros módulos "miren" la memoria
# para implementar el "aprendizaje".

def get_recent_errors(n: int = 5) -> List[ClassifiedError]:
    """Expone los errores recientes al resto de la aplicación."""
    return error_log.get_last(n)

def get_full_error_history() -> List[ClassifiedError]:
    """Expone todo el historial de errores."""
    return error_log.get_all()