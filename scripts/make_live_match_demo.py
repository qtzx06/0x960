from __future__ import annotations

import importlib.util
from pathlib import Path

import chess
import chess.engine
from PIL import Image, ImageDraw, ImageFont

from zero960.engine.search import select_move

ROOT = Path('/Users/qtzx/Desktop/codebase/0x960')
OUT = ROOT / 'media' / 'submission' / 'live_demo'
FRAMES = OUT / 'frames'
OUT.mkdir(parents=True, exist_ok=True)
FRAMES.mkdir(parents=True, exist_ok=True)


def load_eval(path: Path):
    spec = importlib.util.spec_from_file_location('candidate_eval_mod', path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod.evaluate


def draw_board(board: chess.Board, move_log: list[str], title: str, subtitle: str, frame_id: int) -> None:
    w, h = 1280, 720
    sq = 72
    ox, oy = 80, 80
    img = Image.new('RGB', (w, h), (12, 18, 30))
    d = ImageDraw.Draw(img)

    try:
        f_title = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf', 40)
        f_sub = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 24)
        f_piece = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf', 38)
        f_log = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 22)
    except Exception:
        f_title = f_sub = f_piece = f_log = ImageFont.load_default()

    d.text((70, 20), title, fill=(245, 248, 255), font=f_title)
    d.text((70, 62), subtitle, fill=(170, 185, 210), font=f_sub)

    light = (240, 217, 181)
    dark = (181, 136, 99)
    for rank in range(8):
        for file in range(8):
            x0 = ox + file * sq
            y0 = oy + rank * sq
            color = light if (file + rank) % 2 == 0 else dark
            d.rectangle((x0, y0, x0 + sq, y0 + sq), fill=color)

    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        file = chess.square_file(square)
        rank = 7 - chess.square_rank(square)
        x = ox + file * sq + 25
        y = oy + rank * sq + 16
        symbol = piece.symbol()
        fill = (20, 20, 20) if piece.color == chess.BLACK else (250, 250, 250)
        d.text((x, y), symbol.upper() if piece.color == chess.WHITE else symbol.lower(), fill=fill, font=f_piece)

    # frame panel
    d.rounded_rectangle((700, 95, 1230, 670), radius=18, fill=(20, 30, 50))
    d.text((730, 120), 'move log', fill=(245, 248, 255), font=f_sub)
    recent = move_log[-18:]
    y = 160
    for m in recent:
        d.text((730, y), m, fill=(220, 228, 242), font=f_log)
        y += 26

    # border
    d.rectangle((ox-2, oy-2, ox+8*sq+2, oy+8*sq+2), outline=(90, 110, 150), width=3)

    img.save(FRAMES / f'frame_{frame_id:04d}.png')


def main() -> None:
    candidate_eval = load_eval(ROOT / 'src' / 'zero960' / 'workspace_template' / 'eval.py')
    board = chess.Board.from_chess960_pos(518)  # deterministic Chess960 start
    board.chess960 = True

    engine = chess.engine.SimpleEngine.popen_uci('stockfish')
    engine.configure({'UCI_LimitStrength': True, 'UCI_Elo': 1320})

    move_log: list[str] = []
    frame_id = 0
    draw_board(board, move_log, '0x960 live match demo', 'candidate engine vs stockfish-1320 (chess960)', frame_id)
    frame_id += 1

    max_plies = 70
    for ply in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break

        if board.turn == chess.WHITE:
            move = select_move(board, depth=2, eval_fn=candidate_eval)
            side = '0x960'
        else:
            result = engine.play(board, chess.engine.Limit(depth=1))
            move = result.move
            side = 'sf1320'

        if move is None:
            break

        san = board.san(move)
        move_no = board.fullmove_number
        prefix = f"{move_no}." if board.turn == chess.WHITE else f"{move_no}..."
        move_log.append(f"{prefix} {side}: {san}")
        board.push(move)

        for _ in range(6):  # hold each position for visual clarity
            draw_board(board, move_log, '0x960 live match demo', 'candidate engine vs stockfish-1320 (chess960)', frame_id)
            frame_id += 1

    result = board.result(claim_draw=True)
    summary = f'final result: {result}  |  plies: {board.ply()}'
    for _ in range(24):
        draw_board(board, move_log + [summary], '0x960 live match demo', 'candidate engine vs stockfish-1320 (chess960)', frame_id)
        frame_id += 1

    engine.quit()

    (OUT / 'moves.txt').write_text('\n'.join(move_log + [summary]))
    print('frames:', frame_id)


if __name__ == '__main__':
    main()
