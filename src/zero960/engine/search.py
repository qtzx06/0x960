from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import chess

EvalFn = Callable[[chess.Board], int]
MATE_SCORE = 100_000
TT_EXACT = "exact"
TT_LOWER = "lower"
TT_UPPER = "upper"
MAX_TT_ENTRIES = 50_000
CAPTURE_ORDER = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}
ENDGAME_PHASE_THRESHOLD = 6
LOW_BRANCHING_THRESHOLD = 12
ENDGAME_BRANCHING_THRESHOLD = 18
OPENING_FULLMOVE_LIMIT = 2
OPENING_BRANCHING_THRESHOLD = 24
NULL_MOVE_DEPTH_REDUCTION = 2
NULL_MOVE_MIN_DEPTH = 3
LMR_MIN_DEPTH = 3
LMR_MIN_MOVE_INDEX = 3
ASPIRATION_WINDOW = 60


class TTEntry(NamedTuple):
    depth: int
    score: int
    bound: str
    best_move: chess.Move | None


_GLOBAL_TT: dict[tuple[object, ...], TTEntry] = {}
_GLOBAL_HISTORY: dict[tuple[int, int], int] = {}


def _terminal_score(board: chess.Board) -> int:
    if board.is_checkmate():
        return -MATE_SCORE
    return 0


def _phase(board: chess.Board) -> int:
    phase = 0
    phase += 4 * (len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK)))
    phase += 2 * (len(board.pieces(chess.ROOK, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.BLACK)))
    phase += len(board.pieces(chess.BISHOP, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.BLACK))
    phase += len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.KNIGHT, chess.BLACK))
    return min(phase, 24)


def _selective_root_depth(board: chess.Board, depth: int, move_count: int) -> int:
    if depth < 2:
        return depth
    if board.fullmove_number <= OPENING_FULLMOVE_LIMIT and move_count <= OPENING_BRANCHING_THRESHOLD:
        return depth + 1
    if board.is_check() or move_count <= LOW_BRANCHING_THRESHOLD:
        return depth + 1
    if _phase(board) <= ENDGAME_PHASE_THRESHOLD and move_count <= ENDGAME_BRANCHING_THRESHOLD:
        return depth + 1
    return depth


def _score_for_turn(board: chess.Board, eval_fn: EvalFn) -> int:
    score = eval_fn(board)
    return score if board.turn == chess.WHITE else -score


def _move_order_score(
    board: chess.Board,
    move: chess.Move,
    *,
    tt_move: chess.Move | None = None,
    killer_moves: tuple[chess.Move, ...] = (),
    history: dict[tuple[int, int], int] | None = None,
) -> int:
    if tt_move is not None and move == tt_move:
        return 1_000_000

    score = 0
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        if victim is not None:
            score += 100 * CAPTURE_ORDER[victim.piece_type]
        if attacker is not None:
            score -= 10 * CAPTURE_ORDER[attacker.piece_type]
    if move.promotion is not None:
        score += 800 + CAPTURE_ORDER.get(move.promotion, 0)
    if board.gives_check(move):
        score += 50
    if board.is_castling(move):
        score += 25
    if not board.is_capture(move) and move.promotion is None:
        for index, killer in enumerate(killer_moves):
            if move == killer:
                score += 90_000 - index * 10_000
                break
        if history is not None:
            piece_type = board.piece_type_at(move.from_square)
            if piece_type is not None:
                score += history.get((piece_type, move.to_square), 0)
    return score


def _ordered_moves(
    board: chess.Board,
    *,
    tt_move: chess.Move | None = None,
    killer_moves: tuple[chess.Move, ...] = (),
    history: dict[tuple[int, int], int] | None = None,
) -> list[chess.Move]:
    return sorted(
        board.legal_moves,
        key=lambda move: _move_order_score(
            board,
            move,
            tt_move=tt_move,
            killer_moves=killer_moves,
            history=history,
        ),
        reverse=True,
    )


def _tactical_moves(board: chess.Board) -> list[chess.Move]:
    return [
        move
        for move in _ordered_moves(board)
        if board.is_capture(move) or move.promotion is not None
    ]


def _record_killer(killers: dict[int, tuple[chess.Move, ...]], ply: int, move: chess.Move) -> None:
    existing = tuple(candidate for candidate in killers.get(ply, ()) if candidate != move)
    killers[ply] = (move, *existing[:1])


def _record_history(
    history: dict[tuple[int, int], int],
    board: chess.Board,
    move: chess.Move,
    depth: int,
) -> None:
    piece_type = board.piece_type_at(move.from_square)
    if piece_type is None:
        return
    key = (piece_type, move.to_square)
    history[key] = history.get(key, 0) + depth * depth


def _quiescence(board: chess.Board, alpha: int, beta: int, eval_fn: EvalFn) -> int:
    if board.is_game_over(claim_draw=True):
        return _terminal_score(board)

    in_check = board.is_check()
    if not in_check:
        stand_pat = _score_for_turn(board, eval_fn)
        if stand_pat >= beta:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat

    moves = _ordered_moves(board) if in_check else _tactical_moves(board)
    for move in moves:
        board.push(move)
        score = -_quiescence(board, -beta, -alpha, eval_fn)
        board.pop()
        if score >= beta:
            return score
        if score > alpha:
            alpha = score
    return alpha


def negamax(
    board: chess.Board,
    depth: int,
    alpha: int,
    beta: int,
    eval_fn: EvalFn,
    tt: dict[tuple[object, ...], TTEntry],
    killers: dict[int, tuple[chess.Move, ...]],
    history: dict[tuple[int, int], int],
    ply: int = 0,
) -> int:
    if board.is_game_over(claim_draw=True):
        return _terminal_score(board)
    if depth == 0:
        return _quiescence(board, alpha, beta, eval_fn)

    alpha_orig = alpha
    key = board._transposition_key()
    entry = tt.get(key)
    tt_move = entry.best_move if entry is not None else None
    if entry is not None and entry.depth >= depth:
        if entry.bound == TT_EXACT:
            return entry.score
        if entry.bound == TT_LOWER:
            alpha = max(alpha, entry.score)
        elif entry.bound == TT_UPPER:
            beta = min(beta, entry.score)
        if alpha >= beta:
            return entry.score

    if (
        depth >= NULL_MOVE_MIN_DEPTH
        and not board.is_check()
        and _phase(board) > ENDGAME_PHASE_THRESHOLD
        and beta < MATE_SCORE
    ):
        board.push(chess.Move.null())
        null_score = -negamax(
            board,
            depth - 1 - NULL_MOVE_DEPTH_REDUCTION,
            -beta,
            -beta + 1,
            eval_fn,
            tt,
            killers,
            history,
            ply + 1,
        )
        board.pop()
        if null_score >= beta:
            return beta

    best_score = -MATE_SCORE
    best_move: chess.Move | None = None
    killer_moves = killers.get(ply, ())
    for move_index, move in enumerate(_ordered_moves(board, tt_move=tt_move, killer_moves=killer_moves, history=history)):
        board.push(move)
        if move_index == 0:
            score = -negamax(board, depth - 1, -beta, -alpha, eval_fn, tt, killers, history, ply + 1)
        else:
            reduced_depth = depth - 1
            if (
                depth >= LMR_MIN_DEPTH
                and move_index >= LMR_MIN_MOVE_INDEX
                and not board.is_check()
                and not board.is_capture(move)
                and move.promotion is None
            ):
                reduced_depth -= 1
            score = -negamax(board, reduced_depth, -alpha - 1, -alpha, eval_fn, tt, killers, history, ply + 1)
            if alpha < score < beta:
                score = -negamax(board, depth - 1, -beta, -alpha, eval_fn, tt, killers, history, ply + 1)
        board.pop()

        if score > best_score:
            best_score = score
            best_move = move
        if best_score > alpha:
            alpha = best_score
        if alpha >= beta:
            if not board.is_capture(move) and move.promotion is None:
                _record_killer(killers, ply, move)
                _record_history(history, board, move, depth)
            break

    bound = TT_EXACT
    if best_score <= alpha_orig:
        bound = TT_UPPER
    elif best_score >= beta:
        bound = TT_LOWER
    if len(tt) >= MAX_TT_ENTRIES:
        tt.clear()
    tt[key] = TTEntry(depth=depth, score=best_score, bound=bound, best_move=best_move)
    return best_score


def select_move(board: chess.Board, depth: int, eval_fn: EvalFn) -> chess.Move:
    best_move: chess.Move | None = None
    best_score = -MATE_SCORE
    alpha = -MATE_SCORE
    beta = MATE_SCORE
    killers: dict[int, tuple[chess.Move, ...]] = {}
    root_entry = _GLOBAL_TT.get(board._transposition_key())
    if root_entry is not None and abs(root_entry.score) < MATE_SCORE // 2:
        alpha = max(-MATE_SCORE, root_entry.score - ASPIRATION_WINDOW)
        beta = min(MATE_SCORE, root_entry.score + ASPIRATION_WINDOW)
    root_moves = _ordered_moves(
        board,
        tt_move=root_entry.best_move if root_entry is not None else None,
        history=_GLOBAL_HISTORY,
    )
    search_depth = _selective_root_depth(board, depth, len(root_moves))
    use_full_window = False

    for move_index, move in enumerate(root_moves):
        board.push(move)
        if move_index == 0:
            score = -negamax(board, search_depth - 1, -beta, -alpha, eval_fn, _GLOBAL_TT, killers, _GLOBAL_HISTORY, 1)
        else:
            reduced_depth = search_depth - 1
            if (
                search_depth >= LMR_MIN_DEPTH
                and move_index >= LMR_MIN_MOVE_INDEX
                and not board.is_check()
                and not board.is_capture(move)
                and move.promotion is None
            ):
                reduced_depth -= 1
            score = -negamax(
                board,
                reduced_depth,
                -alpha - 1,
                -alpha,
                eval_fn,
                _GLOBAL_TT,
                killers,
                _GLOBAL_HISTORY,
                1,
            )
            if alpha < score < beta:
                score = -negamax(board, search_depth - 1, -beta, -alpha, eval_fn, _GLOBAL_TT, killers, _GLOBAL_HISTORY, 1)
        if not use_full_window and (score <= alpha or score >= beta):
            score = -negamax(
                board,
                search_depth - 1,
                -MATE_SCORE,
                MATE_SCORE,
                eval_fn,
                _GLOBAL_TT,
                killers,
                _GLOBAL_HISTORY,
                1,
            )
            alpha = -MATE_SCORE
            beta = MATE_SCORE
            use_full_window = True
        board.pop()

        if best_move is None or score > best_score:
            best_move = move
            best_score = score
        if best_score > alpha:
            alpha = best_score

    if best_move is None:
        raise RuntimeError("no legal move available")
    return best_move
