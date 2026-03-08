from __future__ import annotations

from openenv.core.client import HTTPEnvClient

from zero960_env.models import Zero960Action, Zero960Observation


class Zero960Client(HTTPEnvClient[Zero960Action, Zero960Observation]):
    def __init__(self, base_url: str) -> None:
        super().__init__(
            base_url=base_url,
            action_type=Zero960Action,
            observation_type=Zero960Observation,
        )

