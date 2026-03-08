from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ActionType = Literal["read_file", "write_file", "run_static_eval", "run_match", "finish"]


@dataclass(slots=True)
class RuntimeAction:
    action_type: ActionType
    path: str | None = None
    content: str | None = None


@dataclass(slots=True)
class RuntimeObservation:
    task: str
    status_message: str
    file_contents: dict[str, str]
    start_position: int
    history: list[str]
    remaining_steps: int
    last_match_score: float | None
    invalid_edit_count: int
    workflow_hint: str
    suggested_actions: list[str]
    has_valid_edit: bool
    has_run_match: bool
    reward: float | None = None
    done: bool = False


@dataclass(slots=True)
class RuntimeStepResult:
    observation: RuntimeObservation
    reward: float | None
    done: bool
    info: dict[str, Any] = field(default_factory=dict)
