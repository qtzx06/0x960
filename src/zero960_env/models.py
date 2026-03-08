from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Zero960Action(BaseModel):
    action_type: Literal["read_file", "write_file", "run_static_eval", "run_match", "finish"]
    path: str | None = None
    content: str | None = None


class Zero960Observation(BaseModel):
    task: str
    status_message: str
    file_contents: dict[str, str] = Field(default_factory=dict)
    start_position: int
    history: list[str] = Field(default_factory=list)
    remaining_steps: int
    last_match_score: float | None = None
    invalid_edit_count: int = 0
    reward: float | None = None
    done: bool = False

