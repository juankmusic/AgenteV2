# core/ai_core/pending_actions_manager.py
import time
from typing import Any, Dict, Optional

class PendingActionsManager:
    """
    Manager simple en memoria para guardar acciones pendientes por session_id.
    Está pensado para ser ligero: los items expiran a las N segundos (default 10 min).
    """
    def __init__(self, ttl_seconds: int = 600):
        self._pending: Dict[str, Dict[str, Any]] = {}
        self.ttl = ttl_seconds

    def save(self, session_id: str, payload: Dict[str, Any]):
        entry = {
            "payload": payload,
            "ts": time.time()
        }
        self._pending[session_id] = entry

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        e = self._pending.get(session_id)
        if not e:
            return None
        if time.time() - e["ts"] > self.ttl:
            # Expiró
            del self._pending[session_id]
            return None
        return e["payload"]

    def clear(self, session_id: str):
        if session_id in self._pending:
            del self._pending[session_id]
