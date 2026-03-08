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

CENTER_SQUARES = {chess.D4, chess.E4, chess.D5, chess.E5}


def evaluate(board: chess.Board) -> int:
    """Return a simple white-centric score in centipawns."""
    if board.is_checkmate():
        return -100_000 if board.turn == chess.WHITE else 100_000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    score = 0
    for piece_type, piece_value in PIECE_VALUES.items():
        score += piece_value * len(board.pieces(piece_type, chess.WHITE))
        score -= piece_value * len(board.pieces(piece_type, chess.BLACK))

    for square in CENTER_SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        score += 15 if piece.color == chess.WHITE else -15

    score += 2 * board.legal_moves.count() if board.turn == chess.WHITE else -2 * board.legal_moves.count()
    return score

