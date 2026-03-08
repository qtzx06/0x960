from __future__ import annotations

import chess

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 335,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

CENTER_SQUARES = (chess.D4, chess.E4, chess.D5, chess.E5)
EXTENDED_CENTER = (
    chess.C3, chess.D3, chess.E3, chess.F3,
    chess.C4, chess.D4, chess.E4, chess.F4,
    chess.C5, chess.D5, chess.E5, chess.F5,
    chess.C6, chess.D6, chess.E6, chess.F6,
)
PIECE_MOBILITY_WEIGHTS = {
    chess.KNIGHT: 4,
    chess.BISHOP: 5,
    chess.ROOK: 3,
    chess.QUEEN: 2,
}
CENTER_AXIS_BONUS = (0, 1, 2, 3, 3, 2, 1, 0)
BISHOP_PAIR_BONUS = 35
ROOK_OPEN_FILE_BONUS = 20
ROOK_SEMIOPEN_FILE_BONUS = 10
DOUBLED_PAWN_PENALTY = 18
ISOLATED_PAWN_PENALTY = 14
BACK_RANK_MINOR_PENALTY = 10
CENTER_OCCUPANCY_BONUS = 14
CENTER_ATTACK_BONUS = 3
CASTLING_RIGHTS_BONUS = 12
CASTLED_BONUS = 18
KNIGHT_CENTER_BONUS = 6
BISHOP_CENTER_BONUS = 2
KING_ENDGAME_CENTER_BONUS = 5
UNDEFENDED_TARGET_DIVISOR = 16
OVERLOADED_TARGET_DIVISOR = 24
TEMPO_BONUS = 8
PASSED_PAWN_BONUS_BY_RANK = [0, 5, 10, 18, 28, 42, 60, 0]
LOOSE_PIECE_DIVISOR = 24
OUTNUMBERED_PIECE_DIVISOR = 40
PAWN_HARASSMENT_PENALTY = 8
CONNECTED_PAWN_BONUS = 4
PAWN_CHAIN_BONUS = 5
CENTRAL_PAWN_DUO_BONUS = 10
ADVANCED_CENTRAL_PAWN_BONUS = 3


def _phase(board: chess.Board) -> int:
    phase = 0
    phase += 4 * (len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK)))
    phase += 2 * (len(board.pieces(chess.ROOK, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.BLACK)))
    phase += len(board.pieces(chess.BISHOP, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.BLACK))
    phase += len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.KNIGHT, chess.BLACK))
    return min(phase, 24)


def _friendly(square: int, color: chess.Color, board: chess.Board) -> bool:
    return board.color_at(square) == color


def _center_axis_score(square: int) -> int:
    return CENTER_AXIS_BONUS[chess.square_file(square)] + CENTER_AXIS_BONUS[chess.square_rank(square)]


def _file_pawn_counts(board: chess.Board, color: chess.Color) -> list[int]:
    counts = [0] * 8
    for square in board.pieces(chess.PAWN, color):
        counts[chess.square_file(square)] += 1
    return counts


def _pawn_structure_score(board: chess.Board, color: chess.Color) -> int:
    score = 0
    pawns = sorted(board.pieces(chess.PAWN, color))
    enemy_pawns = list(board.pieces(chess.PAWN, not color))
    file_counts = _file_pawn_counts(board, color)

    for count in file_counts:
        if count > 1:
            score -= DOUBLED_PAWN_PENALTY * (count - 1)

    for square in pawns:
        file_index = chess.square_file(square)
        left_count = file_counts[file_index - 1] if file_index > 0 else 0
        right_count = file_counts[file_index + 1] if file_index < 7 else 0
        if left_count == 0 and right_count == 0:
            score -= ISOLATED_PAWN_PENALTY

        rank_index = chess.square_rank(square)
        blocked = False
        for enemy_square in enemy_pawns:
            enemy_file = chess.square_file(enemy_square)
            if abs(enemy_file - file_index) > 1:
                continue
            enemy_rank = chess.square_rank(enemy_square)
            if color == chess.WHITE and enemy_rank > rank_index:
                blocked = True
                break
            if color == chess.BLACK and enemy_rank < rank_index:
                blocked = True
                break
        if not blocked:
            advance = rank_index if color == chess.WHITE else 7 - rank_index
            score += PASSED_PAWN_BONUS_BY_RANK[advance]

    return score


def _mobility_score(board: chess.Board, color: chess.Color) -> int:
    score = 0
    friendly_mask = board.occupied_co[color]
    for piece_type, weight in PIECE_MOBILITY_WEIGHTS.items():
        for square in board.pieces(piece_type, color):
            attacks = board.attacks_mask(square) & ~friendly_mask
            score += weight * chess.popcount(attacks)
    return score


def _center_score(board: chess.Board, color: chess.Color) -> int:
    score = 0
    for square in CENTER_SQUARES:
        if _friendly(square, color, board):
            score += CENTER_OCCUPANCY_BONUS

    for square in EXTENDED_CENTER:
        score += CENTER_ATTACK_BONUS * chess.popcount(board.attackers_mask(color, square))
    return score


def _rook_file_score(board: chess.Board, color: chess.Color) -> int:
    score = 0
    friendly_pawns = board.pieces(chess.PAWN, color)
    enemy_pawns = board.pieces(chess.PAWN, not color)
    for square in board.pieces(chess.ROOK, color):
        file_index = chess.square_file(square)
        friendly_on_file = any(chess.square_file(pawn_square) == file_index for pawn_square in friendly_pawns)
        enemy_on_file = any(chess.square_file(pawn_square) == file_index for pawn_square in enemy_pawns)
        if not friendly_on_file:
            score += ROOK_SEMIOPEN_FILE_BONUS
            if not enemy_on_file:
                score += ROOK_OPEN_FILE_BONUS
    return score


def _king_safety_score(board: chess.Board, color: chess.Color, phase: int) -> int:
    king_square = board.king(color)
    if king_square is None:
        return 0

    score = 0
    king_file = chess.square_file(king_square)
    king_rank = chess.square_rank(king_square)

    for file_index in range(max(0, king_file - 1), min(7, king_file + 1) + 1):
        shelter_ranks = [king_rank + 1, king_rank + 2] if color == chess.WHITE else [king_rank - 1, king_rank - 2]
        for rank_index in shelter_ranks:
            if 0 <= rank_index < 8 and _friendly(chess.square(file_index, rank_index), color, board):
                score += 4

    enemy_pressure = 0
    for square in chess.SquareSet(chess.BB_KING_ATTACKS[king_square]):
        enemy_pressure += chess.popcount(board.attackers_mask(not color, square))
    score -= enemy_pressure * (2 + phase // 8)

    if board.has_castling_rights(color):
        score += CASTLING_RIGHTS_BONUS * phase // 24
    return score


def _development_score(board: chess.Board, color: chess.Color, phase: int) -> int:
    if phase <= 8:
        return 0

    home_rank = 0 if color == chess.WHITE else 7
    penalty = 0
    for piece_type in (chess.KNIGHT, chess.BISHOP):
        for square in board.pieces(piece_type, color):
            if chess.square_rank(square) == home_rank:
                penalty += BACK_RANK_MINOR_PENALTY
    return -penalty


def _base_piece_safety_score(board: chess.Board, color: chess.Color) -> int:
    score = 0
    for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        for square in board.pieces(piece_type, color):
            attackers_mask = board.attackers_mask(not color, square)
            if not attackers_mask:
                continue

            attackers = chess.popcount(attackers_mask)
            defenders = chess.popcount(board.attackers_mask(color, square))
            if defenders == 0:
                score -= PIECE_VALUES[piece_type] // LOOSE_PIECE_DIVISOR
            elif attackers > defenders:
                score -= PIECE_VALUES[piece_type] // OUTNUMBERED_PIECE_DIVISOR

            if defenders <= attackers:
                pawn_pressure = 0
                for attacker_square in chess.SquareSet(attackers_mask):
                    if board.piece_type_at(attacker_square) == chess.PAWN:
                        pawn_pressure += 1
                score -= pawn_pressure * PAWN_HARASSMENT_PENALTY
    return score


def _base_piece_placement_score(board: chess.Board, color: chess.Color, phase: int) -> int:
    score = 0
    for square in board.pieces(chess.KNIGHT, color):
        score += KNIGHT_CENTER_BONUS * _center_axis_score(square)
    for square in board.pieces(chess.BISHOP, color):
        score += BISHOP_CENTER_BONUS * _center_axis_score(square)

    king_square = board.king(color)
    if king_square is not None:
        back_rank = 0 if color == chess.WHITE else 7
        if chess.square_rank(king_square) == back_rank:
            if king_square in (chess.C1, chess.G1, chess.C8, chess.G8):
                rook_square = chess.square(3 if chess.square_file(king_square) == 2 else 5, back_rank)
                if _friendly(rook_square, color, board):
                    score += CASTLED_BONUS * phase // 24
        score += KING_ENDGAME_CENTER_BONUS * _center_axis_score(king_square) * (24 - phase) // 24
    return score


def _base_threat_score(board: chess.Board, color: chess.Color) -> int:
    score = 0
    for piece_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        for square in board.pieces(piece_type, not color):
            attackers = chess.popcount(board.attackers_mask(color, square))
            if attackers == 0:
                continue
            defenders = chess.popcount(board.attackers_mask(not color, square))
            if defenders == 0:
                score += PIECE_VALUES[piece_type] // UNDEFENDED_TARGET_DIVISOR
            elif attackers > defenders:
                score += PIECE_VALUES[piece_type] // OVERLOADED_TARGET_DIVISOR
    return score


def _structure_hook(board: chess.Board, color: chess.Color, phase: int) -> int:
    """Swarm lane: structure and castling heuristics."""
    # SWARM_HOOK: structure
    score = 0
    pawns = board.pieces(chess.PAWN, color)
    direction = 1 if color == chess.WHITE else -1

    for square in pawns:
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)

        for neighbor_file in (file_index - 1, file_index + 1):
            if 0 <= neighbor_file < 8:
                neighbor_square = chess.square(neighbor_file, rank_index)
                if neighbor_square in pawns:
                    score += CONNECTED_PAWN_BONUS

        support_rank = rank_index - direction
        if 0 <= support_rank < 8:
            for support_file in (file_index - 1, file_index + 1):
                if 0 <= support_file < 8:
                    support_square = chess.square(support_file, support_rank)
                    if support_square in pawns:
                        score += PAWN_CHAIN_BONUS

        if file_index in (3, 4):
            advance = rank_index if color == chess.WHITE else 7 - rank_index
            if advance >= 3:
                score += ADVANCED_CENTRAL_PAWN_BONUS

    if chess.D4 in pawns and chess.E4 in pawns:
        score += CENTRAL_PAWN_DUO_BONUS
    if chess.D5 in pawns and chess.E5 in pawns:
        score += CENTRAL_PAWN_DUO_BONUS

    return score * phase // 24


def _tactical_hook(board: chess.Board, color: chess.Color, phase: int) -> int:
    """Swarm lane: tactical safety and loose-piece pressure."""
    # SWARM_HOOK: tactical
    score = _base_piece_safety_score(board, color)
    tactical_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 335,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 1200,
    }

    def _least_tactical_value(attackers_mask: int) -> int:
        least = 10_000
        for attacker_square in chess.SquareSet(attackers_mask):
            piece_type = board.piece_type_at(attacker_square)
            if piece_type is None:
                continue
            value = tactical_values[piece_type]
            if value < least:
                least = value
        return least

    def _exchange_edge(attacking_color: chess.Color, square: int, piece_type: int) -> int:
        attackers_mask = board.attackers_mask(attacking_color, square)
        if not attackers_mask:
            return 0

        least_attacker = _least_tactical_value(attackers_mask)
        target_value = PIECE_VALUES[piece_type]
        if least_attacker >= target_value:
            return 0

        defenders_mask = board.attackers_mask(not attacking_color, square)
        edge = 0
        if not defenders_mask:
            edge += (target_value - least_attacker) // 16 + 4
        else:
            least_defender = _least_tactical_value(defenders_mask)
            if least_defender > least_attacker:
                edge += (least_defender - least_attacker) // 24 + 1
            if chess.popcount(attackers_mask) > chess.popcount(defenders_mask):
                edge += target_value // 96
        return edge

    for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        for square in board.pieces(piece_type, not color):
            score += _exchange_edge(color, square, piece_type)
        for square in board.pieces(piece_type, color):
            score -= _exchange_edge(not color, square, piece_type)

    return score


def _activity_hook(board: chess.Board, color: chess.Color, phase: int) -> int:
    """Swarm lane: activity, centralization, and placement."""
    # SWARM_HOOK: activity
    return _base_piece_placement_score(board, color, phase)


def _pawn_endgame_hook(board: chess.Board, color: chess.Color, phase: int) -> int:
    """Swarm lane: pawn structure and endgame conversion."""
    # SWARM_HOOK: pawn_endgame
    return 0


def _initiative_hook(board: chess.Board, color: chess.Color, phase: int) -> int:
    """Swarm lane: threats, tempo conversion, and initiative."""
    # SWARM_HOOK: initiative
    return _base_threat_score(board, color)


def _side_score(board: chess.Board, color: chess.Color, phase: int) -> int:
    score = 0
    for piece_type, piece_value in PIECE_VALUES.items():
        score += piece_value * len(board.pieces(piece_type, color))

    if len(board.pieces(chess.BISHOP, color)) >= 2:
        score += BISHOP_PAIR_BONUS

    score += _pawn_structure_score(board, color)
    score += _mobility_score(board, color)
    score += _center_score(board, color)
    score += _rook_file_score(board, color)
    score += _king_safety_score(board, color, phase)
    score += _development_score(board, color, phase)
    score += _structure_hook(board, color, phase)
    score += _tactical_hook(board, color, phase)
    score += _activity_hook(board, color, phase)
    score += _pawn_endgame_hook(board, color, phase)
    score += _initiative_hook(board, color, phase)
    return score


def evaluate(board: chess.Board) -> int:
    if board.is_checkmate():
        return -100_000 if board.turn == chess.WHITE else 100_000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    phase = _phase(board)
    score = _side_score(board, chess.WHITE, phase) - _side_score(board, chess.BLACK, phase)
    score += TEMPO_BONUS if board.turn == chess.WHITE else -TEMPO_BONUS
    return score
