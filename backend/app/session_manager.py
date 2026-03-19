"""Per-session state management for multi-user support.

Each browser tab gets its own Session with an independent DRREngine and
RegistrationRunner. Sessions are created on WebSocket connect and destroyed
on disconnect (or by the stale-session reaper).
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field

from app.drr_engine import DRREngine
from app.registration import RegistrationRunner

logger = logging.getLogger(__name__)


@dataclass
class Session:
    id: str
    engine: DRREngine | None = None
    runner: RegistrationRunner | None = None
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_active = time.time()


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create_session(self) -> Session:
        session = Session(id=uuid.uuid4().hex)
        with self._lock:
            self._sessions[session.id] = session
        logger.info("Session created: %s", session.id)
        return session

    def get_session(self, session_id: str) -> Session | None:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is not None:
            session.touch()
        return session

    def destroy_session(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return
        # Cancel any active registration
        if session.runner and session.runner.is_running:
            session.runner.cancel()
            logger.info("Session %s: cancelled active registration", session_id)
        # Free engine resources
        if session.engine is not None:
            session.engine.clear_volume()
            logger.info("Session %s: cleared engine volume", session_id)
        logger.info("Session destroyed: %s", session_id)

    def destroy_all(self) -> None:
        with self._lock:
            ids = list(self._sessions.keys())
        for sid in ids:
            self.destroy_session(sid)

    def cleanup_stale(self, max_idle_seconds: float = 3600) -> None:
        now = time.time()
        with self._lock:
            stale = [
                sid for sid, s in self._sessions.items()
                if now - s.last_active > max_idle_seconds
            ]
        for sid in stale:
            logger.warning("Reaping stale session: %s", sid)
            self.destroy_session(sid)

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._sessions)
