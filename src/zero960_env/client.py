from __future__ import annotations

from typing import Any, Dict

from openenv.core.env_client import EnvClient, StepResult
from openenv.core.env_server.types import State

from zero960_env.models import Zero960Action, Zero960Observation


class Zero960Client(EnvClient[Zero960Action, Zero960Observation, State]):
    def _step_payload(self, action: Zero960Action) -> Dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[Zero960Observation]:
        obs_data = payload.get("observation", payload)
        observation = Zero960Observation.model_validate(obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State.model_validate(payload)
