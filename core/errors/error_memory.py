from collections import deque
from typing import List
from .error_types import ErrorType  # Importación relativa
from .error_types import ClassifiedError  # Importación relativa

class ErrorMemory:
    """
    Almacena un historial reciente de los últimos N errores clasificados.
    """
    def __init__(self, max_size: int = 20):
        """
        Inicializa la memoria con un tamaño máximo.
        :param max_size: El número de errores a recordar.
        """
        # maxlen=max_size se encarga automáticamente de eliminar
        # el elemento más antiguo cuando se agrega uno nuevo y la cola está llena.
        self.memory = deque(maxlen=max_size)

    def add(self, error_info: ClassifiedError):
        """
        Agrega un nuevo error clasificado a la memoria.
        """
        print(f"[ErrorMemory] Agregando error: {error_info.type}")
        self.memory.append(error_info)

    def get_last(self, n: int = 5) -> List[ClassifiedError]:
        """
        Recupera los últimos 'n' errores.
        """
        # Convierte el deque a lista y devuelve los últimos n elementos
        return list(self.memory)[-n:]

    def get_all(self) -> List[ClassifiedError]:
        """
        Recupera todos los errores actualmente en memoria.
        """
        return list(self.memory)

    def get_last_error_type(self) -> ErrorType | None:
        """
        Devuelve el tipo del último error, si existe.
        """
        if self.memory:
            return self.memory[-1].type
        return None