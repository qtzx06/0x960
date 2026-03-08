from __future__ import annotations

import chess

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


def evaluate(board: chess.Board) -> int:
    score = 0
    for piece_type, piece_value in PIECE_VALUES.items():
        score += piece_value * len(board.pieces(piece_type, chess.WHITE))
        score -= piece_value * len(board.pieces(piece_type, chess.BLACK))

    white_center = sum(1 for square in (chess.D4, chess.E4, chess.D5, chess.E5) if board.color_at(square) == chess.WHITE)
    black_center = sum(1 for square in (chess.D4, chess.E4, chess.D5, chess.E5) if board.color_at(square) == chess.BLACK)
    score += 15 * (white_center - black_center)
    return score
