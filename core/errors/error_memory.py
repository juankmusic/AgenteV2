from collections import deque
from typing import List
from .error_types import ErrorType  # Importación relativa
from .error_types import ClassifiedError  # Importación relativa

class ErrorMemory:
    """
    Esta clase funciona como una pequeña "caja de recuerdos".
    Aquí guardamos los últimos errores que han ocurrido.
    La caja tiene un límite, así que si se llena, el error más viejo se borra solo.
    """

    def __init__(self, max_size: int = 20):
        """
        Crea la caja donde se guardarán los errores.
        :param max_size: Cantidad máxima de errores que se van a recordar.
        """
        # Usamos deque porque maneja automáticamente el tamaño máximo.
        # Si se llena, borra el error más antiguo cuando se agrega uno nuevo.
        self.memory = deque(maxlen=max_size)

    def add(self, error_info: ClassifiedError):
        """
        Guarda un nuevo error dentro de la memoria.
        """
        # Este print sirve para que el desarrollador vea qué error se agregó.
        print(f"[ErrorMemory] Agregando error: {error_info.type}")

        # Añadimos el error a la cola.
        self.memory.append(error_info)

    def get_last(self, n: int = 5) -> List[ClassifiedError]:
        """
        Devuelve una lista con los últimos 'n' errores.
        """
        # Convertimos la cola completa a lista
        # y luego regresamos solo los últimos n elementos.
        return list(self.memory)[-n:]

    def get_all(self) -> List[ClassifiedError]:
        """
        Devuelve todos los errores guardados en la memoria.
        """
        return list(self.memory)

    def get_last_error_type(self) -> ErrorType | None:
        """
        Devuelve el tipo del error más reciente.
        Si no hay errores guardados aún, devuelve None.
        """
        if self.memory:
            # El último error está al final de la cola
            return self.memory[-1].type

        # Si no hay nada aún, devolvemos None
        return None