from __future__ import annotations

import random
from dataclasses import dataclass

import chess

from zero960.engine.default_eval import evaluate as baseline_evaluate
from zero960.engine.match import play_match_series
from zero960.runtime.types import RuntimeAction, RuntimeObservation, RuntimeStepResult
from zero960.runtime.workspace import WorkspaceManager


@dataclass(slots=True)
class EpisodeConfig:
    max_steps: int = 6
    search_depth: int = 2
    training_games: int = 2
    crash_penalty: float = 0.25


class Zero960EpisodeRuntime:
    def __init__(self, config: EpisodeConfig | None = None) -> None:
        self.config = config or EpisodeConfig()
        self.workspace: WorkspaceManager | None = None
        self.start_position = 0
        self.history: list[str] = []
        self.steps_taken = 0
        self.invalid_edit_count = 0
        self.last_match_score: float | None = None

    def reset(self, chess960_index: int | None = None) -> RuntimeObservation:
        self.close()
        self.workspace = WorkspaceManager()
        self.start_position = chess960_index if chess960_index is not None else random.randrange(960)
        self.history = []
        self.steps_taken = 0
        self.invalid_edit_count = 0
        self.last_match_score = None
        return self._observation("episode reset")

    def close(self) -> None:
        if self.workspace is not None:
            self.workspace.cleanup()
            self.workspace = None

    def step(self, action: RuntimeAction) -> RuntimeStepResult:
        if self.workspace is None:
            raise RuntimeError("episode not initialized; call reset() first")

        done = False
        reward: float | None = None
        status_message = ""
        info: dict[str, object] = {}

        try:
            if action.action_type == "read_file":
                if action.path is None:
                    raise ValueError("read_file requires path")
                content = self.workspace.read_file(action.path)
                status_message = f"read {action.path} ({len(content)} bytes)"
            elif action.action_type == "write_file":
                if action.path is None or action.content is None:
                    raise ValueError("write_file requires path and content")
                self.workspace.write_file(action.path, action.content)
                status_message = f"wrote {action.path}"
            elif action.action_type == "run_static_eval":
                eval_fn = self.workspace.load_eval_function()
                board = chess.Board.from_chess960_pos(self.start_position)
                board.chess960 = True
                score = eval_fn(board)
                status_message = f"static eval score={score}"
                info["static_eval_score"] = score
            elif action.action_type == "run_match":
                self.last_match_score = self._run_training_match()
                status_message = f"match score={self.last_match_score:.3f}"
                info["match_score"] = self.last_match_score
            elif action.action_type == "finish":
                reward = self._final_reward()
                done = True
                status_message = f"episode finished with reward={reward:.3f}"
            else:
                raise ValueError(f"unsupported action_type={action.action_type}")
        except Exception as exc:
            self.invalid_edit_count += 1
            status_message = f"action failed: {exc}"
            info["error"] = str(exc)

        self.history.append(f"{action.action_type}: {status_message}")
        self.steps_taken += 1

        if not done and self.steps_taken >= self.config.max_steps:
            reward = self._final_reward()
            done = True
            status_message = f"{status_message}; step budget exhausted with reward={reward:.3f}"

        observation = self._observation(status_message, reward=reward, done=done)
        return RuntimeStepResult(observation=observation, reward=reward, done=done, info=info)

    def _run_training_match(self) -> float:
        if self.workspace is None:
            raise RuntimeError("workspace unavailable")
        candidate_eval = self.workspace.load_eval_function()
        start_positions = [(self.start_position + offset) % 960 for offset in range(self.config.training_games)]
        result = play_match_series(
            candidate_eval=candidate_eval,
            baseline_eval=baseline_evaluate,
            start_positions=start_positions,
            depth=self.config.search_depth,
        )
        return result.candidate_points / result.total_games

    def _final_reward(self) -> float:
        if self.last_match_score is None:
            self.last_match_score = self._run_training_match()
        penalty = self.invalid_edit_count * self.config.crash_penalty
        return self.last_match_score - penalty

    def _observation(
        self,
        status_message: str,
        reward: float | None = None,
        done: bool = False,
    ) -> RuntimeObservation:
        if self.workspace is None:
            raise RuntimeError("workspace unavailable")
        return RuntimeObservation(
            task=(
                "Improve eval.py for the current Chess960 engine. "
                "Use bounded file edits and finish when ready for scoring."
            ),
            status_message=status_message,
            file_contents={"eval.py": self.workspace.read_file("eval.py")},
            start_position=self.start_position,
            history=list(self.history),
            remaining_steps=max(self.config.max_steps - self.steps_taken, 0),
            last_match_score=self.last_match_score,
            invalid_edit_count=self.invalid_edit_count,
            reward=reward,
            done=done,
        )

