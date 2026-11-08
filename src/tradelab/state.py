"""Stockage d'état atomique, repris de la logique de ``bot.py`` (``load_state`` / ``save_state``).

Un bot qui redémarre avec un état VIDE alors que le compte contient des positions rachète en boucle :
c'est l'incident décrit dans ``fix_after_incident.py``. Les règles ici :

- écriture dans un fichier temporaire complet, ``fsync``, puis remplacement atomique : une coupure ne
  laisse jamais un fichier à moitié écrit ;
- l'ancien état est conservé en ``.bak`` ;
- lecture : fichier principal, sinon ``.bak`` ; si des fichiers existent mais qu'AUCUN n'est lisible,
  on lève ``StateUnreadableError`` au lieu de repartir à vide.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class StateUnreadableError(RuntimeError):
    """Des fichiers d'état existent mais aucun n'est lisible : ne pas repartir à vide."""


class StateStore:
    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.backup = Path(str(path) + ".bak")
        self.tmp = Path(str(path) + ".tmp")

    def load(self) -> dict[str, Any] | None:
        """Renvoie l'état, ou None si c'est un premier démarrage (aucun fichier)."""
        existing = [p for p in (self.path, self.backup) if p.exists()]
        for p in existing:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(data, dict):
                return data
        if existing:
            raise StateUnreadableError(
                f"État illisible ({', '.join(str(p) for p in existing)}). Arrêt volontaire : ne relance pas à vide."
            )
        return None

    def save(self, state: dict[str, Any]) -> None:
        payload = json.dumps(state, indent=2, default=str)
        with open(self.tmp, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        if self.path.exists() and self.path.stat().st_size > 0:
            os.replace(self.path, self.backup)
        os.replace(self.tmp, self.path)
