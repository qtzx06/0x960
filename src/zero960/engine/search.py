from __future__ import annotations

from collections.abc import Callable

import chess

EvalFn = Callable[[chess.Board], int]
MATE_SCORE = 100_000


def _terminal_score(board: chess.Board) -> int:
    if board.is_checkmate():
        return -MATE_SCORE
    return 0


def _score_for_turn(board: chess.Board, eval_fn: EvalFn) -> int:
    score = eval_fn(board)
    return score if board.turn == chess.WHITE else -score


def negamax(board: chess.Board, depth: int, alpha: int, beta: int, eval_fn: EvalFn) -> int:
    if depth == 0 or board.is_game_over(claim_draw=True):
        if board.is_game_over(claim_draw=True):
            return _terminal_score(board)
        return _score_for_turn(board, eval_fn)

    best_score = -MATE_SCORE
    for move in board.legal_moves:
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha, eval_fn)
        board.pop()

        if score > best_score:
            best_score = score
        if best_score > alpha:
            alpha = best_score
        if alpha >= beta:
            break

    return best_score


def select_move(board: chess.Board, depth: int, eval_fn: EvalFn) -> chess.Move:
    best_move: chess.Move | None = None
    best_score = -MATE_SCORE
    alpha = -MATE_SCORE
    beta = MATE_SCORE

    for move in board.legal_moves:
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha, eval_fn)
        board.pop()

        if best_move is None or score > best_score:
            best_move = move
            best_score = score
        if best_score > alpha:
            alpha = best_score

    if best_move is None:
        raise RuntimeError("no legal move available")
    return best_move

