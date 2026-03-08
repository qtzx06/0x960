from __future__ import annotations

from uuid import uuid4

from openenv.core.env_server import Environment
from openenv.core.env_server.types import State

from zero960.runtime.episode import EpisodeConfig, Zero960EpisodeRuntime
from zero960.runtime.types import RuntimeAction
from zero960_env.models import Zero960Action, Zero960Observation


class Zero960Environment(Environment[Zero960Action, Zero960Observation]):
    def __init__(self) -> None:
        self.runtime = Zero960EpisodeRuntime(EpisodeConfig())
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(self) -> Zero960Observation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        observation = self.runtime.reset()
        return Zero960Observation(**observation.__dict__)

    def step(self, action: Zero960Action) -> Zero960Observation:
        result = self.runtime.step(
            RuntimeAction(
                action_type=action.action_type,
                path=action.path,
                content=action.content,
            )
        )
        self._state.step_count += 1
        return Zero960Observation(**result.observation.__dict__)

    @property
    def state(self) -> State:
        return self._state

