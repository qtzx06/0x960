from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import chess

from zero960.engine.search import EvalFn, select_move


@dataclass(slots=True)
class MatchResult:
    candidate_points: float
    total_games: int
    finished_games: int


def _new_board(chess960_index: int) -> chess.Board:
    board = chess.Board.from_chess960_pos(chess960_index)
    board.chess960 = True
    return board


def play_game(
    chess960_index: int,
    white_eval: EvalFn,
    black_eval: EvalFn,
    depth: int = 2,
    max_plies: int = 120,
) -> float:
    board = _new_board(chess960_index)

    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        eval_fn = white_eval if board.turn == chess.WHITE else black_eval
        move = select_move(board, depth=depth, eval_fn=eval_fn)
        board.push(move)

    result = board.result(claim_draw=True)
    if result == "1-0":
        return 1.0
    if result == "0-1":
        return 0.0
    return 0.5


def play_match_series(
    candidate_eval: EvalFn,
    baseline_eval: EvalFn,
    start_positions: Iterable[int],
    depth: int = 2,
) -> MatchResult:
    points = 0.0
    total_games = 0

    for chess960_index in start_positions:
        total_games += 2
        points += play_game(chess960_index, candidate_eval, baseline_eval, depth=depth)
        points += 1.0 - play_game(chess960_index, baseline_eval, candidate_eval, depth=depth)

    return MatchResult(
        candidate_points=points,
        total_games=total_games,
        finished_games=total_games,
    )

