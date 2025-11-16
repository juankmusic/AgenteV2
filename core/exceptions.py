# ======================================================
# Archivo: core/exceptions.py (NUEVO ARCHIVO)
# ======================================================
"""
Excepciones personalizadas para el Agente Inteligente.

Permiten un manejo de errores más semántico y específico 
que las excepciones genéricas de Python.
"""

class LogicalError(Exception):
    """
    Excepción base para errores de lógica de negocio o de usuario.
    
    Atributos:
        message (str): La descripción del error para el usuario.
        suggested_action (str): (Opcional) Una alternativa lógica que el 
                                agente puede proponer.
    """
    def __init__(self, message, suggested_action=None):
        super().__init__(message)
        self.suggested_action = suggested_action

class InvalidOperationError(LogicalError):
    """
    Se lanza cuando se intenta una operación en un tipo de dato incorrecto.
    Ej: SUM() en una columna de texto.
    """
    pass

class InvalidVisualizationError(LogicalError):
    """
    Se lanza cuando se solicita una visualización incompatible con los datos.
    Ej: Gráfico de líneas sin datos numéricos.
    """
    pass

class DataValidationError(LogicalError):
    """
    Se lanza cuando los datos de entrada para una operación fallan 
    una validación de negocio.
    """
    pass