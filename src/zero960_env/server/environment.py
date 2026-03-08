from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server import Environment
from openenv.core.env_server.types import State

from zero960.runtime.episode import EpisodeConfig, Zero960EpisodeRuntime
from zero960.runtime.types import RuntimeAction
from zero960_env.models import Zero960Action, Zero960Observation


class Zero960Environment(Environment[Zero960Action, Zero960Observation, State]):
    def __init__(self) -> None:
        super().__init__()
        self.runtime = Zero960EpisodeRuntime(EpisodeConfig())
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Zero960Observation:
        eid = episode_id or str(uuid4())
        self._state = State(episode_id=eid, step_count=0)
        observation = self.runtime.reset(chess960_index=seed)
        return Zero960Observation(
            task=observation.task,
            status_message=observation.status_message,
            file_contents=observation.file_contents,
            start_position=observation.start_position,
            history=observation.history,
            remaining_steps=observation.remaining_steps,
            last_match_score=observation.last_match_score,
            invalid_edit_count=observation.invalid_edit_count,
            workflow_hint=observation.workflow_hint,
            suggested_actions=observation.suggested_actions,
            has_valid_edit=observation.has_valid_edit,
            has_run_match=observation.has_run_match,
        )

    def step(
        self,
        action: Zero960Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Zero960Observation:
        result = self.runtime.step(
            RuntimeAction(
                action_type=action.action_type,
                path=action.path,
                content=action.content,
            )
        )
        self._state.step_count += 1
        obs = result.observation
        return Zero960Observation(
            task=obs.task,
            status_message=obs.status_message,
            file_contents=obs.file_contents,
            start_position=obs.start_position,
            history=obs.history,
            remaining_steps=obs.remaining_steps,
            last_match_score=obs.last_match_score,
            invalid_edit_count=obs.invalid_edit_count,
            workflow_hint=obs.workflow_hint,
            suggested_actions=obs.suggested_actions,
            has_valid_edit=obs.has_valid_edit,
            has_run_match=obs.has_run_match,
            reward=obs.reward,
            done=obs.done,
        )

    @property
    def state(self) -> State:
        return self._state
