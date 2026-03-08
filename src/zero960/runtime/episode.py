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
    search_depth: int = 1
    training_games: int = 1
    crash_penalty: float = 0.25
    valid_write_bonus: float = 0.20
    changed_write_bonus: float = 0.10
    unchanged_write_penalty: float = 0.10
    explicit_match_bonus: float = 0.15
    finish_after_match_bonus: float = 0.05
    repeated_static_eval_penalty: float = 0.15
    static_eval_before_write_penalty: float = 0.20
    redundant_read_penalty: float = 0.25
    match_without_edit_penalty: float = 0.15
    finish_without_edit_penalty: float = 0.45
    finish_without_match_penalty: float = 0.20
    finish_without_retest_penalty: float = 0.08


class Zero960EpisodeRuntime:
    def __init__(self, config: EpisodeConfig | None = None) -> None:
        self.config = config or EpisodeConfig()
        self.workspace: WorkspaceManager | None = None
        self.start_position = 0
        self.history: list[str] = []
        self.steps_taken = 0
        self.invalid_edit_count = 0
        self.last_match_score: float | None = None
        self.has_valid_edit = False
        self.has_run_match = False
        self.wrote_since_match = False
        self.shaping_reward_total = 0.0
        self.last_action_type: str | None = None

    def reset(self, chess960_index: int | None = None) -> RuntimeObservation:
        self.close()
        self.workspace = WorkspaceManager()
        self.start_position = chess960_index if chess960_index is not None else random.randrange(960)
        self.history = []
        self.steps_taken = 0
        self.invalid_edit_count = 0
        self.last_match_score = None
        self.has_valid_edit = False
        self.has_run_match = False
        self.wrote_since_match = False
        self.shaping_reward_total = 0.0
        self.last_action_type = None
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
        step_reward = 0.0
        status_message = ""
        info: dict[str, object] = {}

        try:
            if action.action_type == "read_file":
                if action.path is None:
                    raise ValueError("read_file requires path")
                content = self.workspace.read_file(action.path)
                status_message = f"read {action.path} ({len(content)} bytes)"
                if action.path == "eval.py":
                    step_reward -= self.config.redundant_read_penalty
                    status_message += "; eval.py was already visible"
            elif action.action_type == "write_file":
                if action.path is None or action.content is None:
                    raise ValueError("write_file requires path and content")
                previous_content = self.workspace.read_file(action.path)
                self.workspace.write_file(action.path, action.content)
                try:
                    self.workspace.load_eval_function()
                except Exception:
                    self.workspace.write_file(action.path, previous_content)
                    raise

                if action.content == previous_content:
                    step_reward -= self.config.unchanged_write_penalty
                    status_message = f"wrote {action.path}; file unchanged"
                else:
                    step_reward += self.config.valid_write_bonus + self.config.changed_write_bonus
                    self.has_valid_edit = True
                    self.wrote_since_match = True
                    status_message = f"wrote {action.path}; validated evaluate(board)"
                    info["code_changed"] = True
            elif action.action_type == "run_static_eval":
                eval_fn = self.workspace.load_eval_function()
                board = chess.Board.from_chess960_pos(self.start_position)
                board.chess960 = True
                score = eval_fn(board)
                status_message = f"static eval score={score}"
                info["static_eval_score"] = score
                if not self.has_valid_edit:
                    step_reward -= self.config.static_eval_before_write_penalty
                if self.last_action_type == "run_static_eval":
                    step_reward -= self.config.repeated_static_eval_penalty
            elif action.action_type == "run_match":
                self.last_match_score = self._run_training_match()
                self.has_run_match = True
                status_message = f"match score={self.last_match_score:.3f}"
                info["match_score"] = self.last_match_score
                if self.has_valid_edit and self.wrote_since_match:
                    step_reward += self.config.explicit_match_bonus
                    self.wrote_since_match = False
                elif not self.has_valid_edit:
                    step_reward -= self.config.match_without_edit_penalty
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

        if not done:
            reward = step_reward
            self.shaping_reward_total += step_reward

        self.history.append(f"{action.action_type}: {status_message}")
        self.steps_taken += 1
        self.last_action_type = action.action_type

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
        reward = self.last_match_score + self.shaping_reward_total
        if self.has_run_match:
            reward += self.config.finish_after_match_bonus
        if not self.has_valid_edit:
            reward -= self.config.finish_without_edit_penalty
        if not self.has_run_match:
            reward -= self.config.finish_without_match_penalty
        if self.wrote_since_match:
            reward -= self.config.finish_without_retest_penalty
        penalty = self.invalid_edit_count * self.config.crash_penalty
        return reward - penalty

    def _observation(
        self,
        status_message: str,
        reward: float | None = None,
        done: bool = False,
    ) -> RuntimeObservation:
        if self.workspace is None:
            raise RuntimeError("workspace unavailable")
        workflow_hint, suggested_actions = self._workflow_state()
        return RuntimeObservation(
            task=(
                "Improve eval.py for the current Chess960 engine. "
                "The full file is already visible below. Best loop: write_file a valid replacement, "
                "run_match to test it, then finish. Repeated run_static_eval and early finish are penalized."
            ),
            status_message=status_message,
            file_contents={"eval.py": self.workspace.read_file("eval.py")},
            start_position=self.start_position,
            history=list(self.history),
            remaining_steps=max(self.config.max_steps - self.steps_taken, 0),
            last_match_score=self.last_match_score,
            invalid_edit_count=self.invalid_edit_count,
            workflow_hint=workflow_hint,
            suggested_actions=suggested_actions,
            has_valid_edit=self.has_valid_edit,
            has_run_match=self.has_run_match,
            reward=reward,
            done=done,
        )

    def _workflow_state(self) -> tuple[str, list[str]]:
        if not self.has_valid_edit:
            return (
                "eval.py is already shown below. Do not waste a turn on read_file. "
                "Write a full valid replacement for eval.py next.",
                ["write_file", "run_match", "finish"],
            )
        if self.wrote_since_match:
            return (
                "You have a valid untested edit. Run run_match next to measure it.",
                ["run_match", "write_file", "finish"],
            )
        if self.has_run_match:
            return (
                "You have a tested edit. Finish if the score is acceptable, otherwise write_file again.",
                ["finish", "write_file", "run_match"],
            )
        return (
            "A valid edit exists but no explicit match has been run yet. Run run_match next.",
            ["run_match", "finish", "write_file"],
        )
