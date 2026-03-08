from __future__ import annotations

from typing import Literal

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class Zero960Action(Action):
    action_type: Literal["read_file", "write_file", "run_static_eval", "run_match", "finish"]
    path: str | None = None
    content: str | None = None


class Zero960Observation(Observation):
    task: str = ""
    status_message: str = ""
    file_contents: dict[str, str] = Field(default_factory=dict)
    start_position: int = 0
    history: list[str] = Field(default_factory=list)
    remaining_steps: int = 0
    last_match_score: float | None = None
    invalid_edit_count: int = 0
