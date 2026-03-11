from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class ModelHealthState:
    consecutive_failures: int = 0
    cooldown_until: float = 0.0
    last_error: str | None = None


class ModelHealthTracker:
    def __init__(self, cooldown_seconds: float):
        self.cooldown_seconds = cooldown_seconds
        self._states: dict[str, ModelHealthState] = {}

    def is_available(self, model: str, now: float | None = None) -> bool:
        state = self._states.get(model)
        if state is None:
            return True
        current = time.monotonic() if now is None else now
        return current >= state.cooldown_until

    def record_success(self, model: str) -> None:
        self._states[model] = ModelHealthState()

    def record_failure(self, model: str, error: str, now: float | None = None) -> None:
        current = time.monotonic() if now is None else now
        state = self._states.setdefault(model, ModelHealthState())
        state.consecutive_failures += 1
        state.last_error = error
        state.cooldown_until = current + self.cooldown_seconds * state.consecutive_failures

    def snapshot(self) -> dict[str, dict[str, float | int | str | None]]:
        return {
            model: {
                "consecutive_failures": state.consecutive_failures,
                "cooldown_until": state.cooldown_until,
                "last_error": state.last_error,
            }
            for model, state in self._states.items()
        }
