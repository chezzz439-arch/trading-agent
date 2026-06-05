"""Cross-process shared state for the monitoring layers.

The agent process and the three dashboards (Streamlit, Telegram, terminal) run
independently, so they communicate through two small JSON files under ``logs/``:

* ``agent_state.json`` — a snapshot the agent writes each scan and the
  dashboards read (equity, positions, scores, regime, ...).
* ``control.json``     — a control channel the dashboards write and the agent
  polls (currently: a HALT request from the Streamlit kill-switch button).

Writes are atomic (temp file + ``os.replace``) so a reader never sees a partial
file. All operations are best-effort and never raise.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

_MAX_EQUITY_POINTS = 2880   # ~10 days of 5-min scans


class StateStore:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.state_path = os.path.join(log_dir, "agent_state.json")
        self.control_path = os.path.join(log_dir, "control.json")
        os.makedirs(log_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Atomic JSON helpers
    # ------------------------------------------------------------------ #
    def _write_json(self, path: str, data: dict) -> None:
        try:
            d = os.path.dirname(path) or "."
            fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, default=str)
            os.replace(tmp, path)
        except Exception:
            logger.exception("StateStore: write failed for %s", path)

    @staticmethod
    def _read_json(path: str) -> dict:
        try:
            with open(path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        except Exception:
            logger.exception("StateStore: read failed for %s", path)
            return {}

    # ------------------------------------------------------------------ #
    # Agent state
    # ------------------------------------------------------------------ #
    def write_state(self, state: dict) -> None:
        state = dict(state)
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        # Maintain a rolling equity history for the live chart.
        prev = self.read_state()
        history = prev.get("equity_history", [])
        eq = state.get("equity")
        if eq is not None:
            history.append({"t": state["updated_at"], "equity": eq})
            history = history[-_MAX_EQUITY_POINTS:]
        state["equity_history"] = history
        self._write_json(self.state_path, state)

    def read_state(self) -> dict:
        return self._read_json(self.state_path)

    def is_fresh(self, max_age_seconds: int = 120) -> bool:
        """True if the agent has updated state recently (used for health checks)."""
        st = self.read_state()
        ts = st.get("updated_at")
        if not ts:
            return False
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).total_seconds()
            return age <= max_age_seconds
        except ValueError:
            return False

    # ------------------------------------------------------------------ #
    # Control channel (dashboard -> agent)
    # ------------------------------------------------------------------ #
    def request_halt(self, reason: str = "manual halt from dashboard") -> None:
        self._write_json(self.control_path, {
            "halt": True, "reason": reason,
            "requested_at": datetime.now(timezone.utc).isoformat(),
        })

    def halt_requested(self) -> Optional[str]:
        ctl = self._read_json(self.control_path)
        return ctl.get("reason") if ctl.get("halt") else None

    def clear_halt(self) -> None:
        self._write_json(self.control_path, {"halt": False})
