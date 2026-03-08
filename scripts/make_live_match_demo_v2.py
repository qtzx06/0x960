from __future__ import annotations

import importlib.util
from pathlib import Path

import chess
import chess.engine
from PIL import Image, ImageDraw, ImageFont

from zero960.engine.search import select_move

ROOT = Path('/Users/qtzx/Desktop/codebase/0x960')
OUT = ROOT / 'media' / 'submission' / 'live_demo_v2'
OUT.mkdir(parents=True, exist_ok=True)

PIECE_GLYPHS = {
    'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
    'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚',
}


def load_eval(path: Path):
    spec = importlib.util.spec_from_file_location('candidate_eval_mod', path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod.evaluate


def make_fonts():
    return {
        'title': ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf', 38),
        'sub': ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 23),
        'piece': ImageFont.truetype('/System/Library/Fonts/Apple Symbols.ttf', 54),
        'log': ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 20),
        'badge': ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf', 20),
    }


def draw_frame(board: chess.Board, move_log: list[str], game_idx: int, frame_id: int, fonts: dict, out_dir: Path):
    W, H = 1280, 720
    sq = 68
    ox, oy = 70, 96
    img = Image.new('RGB', (W, H), (8, 12, 22))
    d = ImageDraw.Draw(img)

    # panels
    d.rounded_rectangle((24, 24, 1256, 696), radius=24, fill=(15, 24, 42), outline=(50, 70, 110), width=2)
    d.rounded_rectangle((54, 82, 54 + 8 * sq + 32, 82 + 8 * sq + 30), radius=16, fill=(20, 30, 48))
    d.rounded_rectangle((700, 96, 1228, 650), radius=16, fill=(20, 30, 50))

    d.text((64, 34), f'0x960 live match {game_idx+1}/2', font=fonts['title'], fill=(242, 247, 255))
    d.text((64, 72), 'candidate engine vs stockfish-1320 anchor', font=fonts['sub'], fill=(172, 189, 220))

    # badges
    d.rounded_rectangle((704, 52, 952, 84), radius=12, fill=(22, 163, 74))
    d.text((718, 58), 'estimated ~1600 local anchor', font=fonts['badge'], fill='white')
    d.rounded_rectangle((960, 52, 1224, 84), radius=12, fill=(37, 99, 235))
    d.text((972, 58), 'approx top ~10% online rapid*', font=fonts['badge'], fill='white')

    # board
    light, dark = (236, 228, 210), (134, 103, 73)
    for rank in range(8):
        for file in range(8):
            x0 = ox + file * sq
            y0 = oy + rank * sq
            d.rectangle((x0, y0, x0 + sq, y0 + sq), fill=light if (file + rank) % 2 == 0 else dark)
    d.rectangle((ox - 3, oy - 3, ox + 8 * sq + 3, oy + 8 * sq + 3), outline=(96, 124, 175), width=3)

    # coordinates
    files = 'abcdefgh'
    for i, f in enumerate(files):
        d.text((ox + i * sq + 26, oy + 8 * sq + 4), f, font=fonts['sub'], fill=(180, 197, 227))
    for i in range(8):
        d.text((ox - 24, oy + i * sq + 20), str(8 - i), font=fonts['sub'], fill=(180, 197, 227))

    # pieces
    for square, piece in board.piece_map().items():
        file = chess.square_file(square)
        rank = 7 - chess.square_rank(square)
        x = ox + file * sq + 12
        y = oy + rank * sq + 5
        glyph = PIECE_GLYPHS[piece.symbol()]
        fill = (250, 250, 250) if piece.color == chess.WHITE else (24, 24, 24)
        d.text((x, y), glyph, font=fonts['piece'], fill=fill)

    # move log panel
    d.text((724, 110), 'move log', font=fonts['title'], fill=(242, 247, 255))
    y = 156
    for m in move_log[-18:]:
        d.text((724, y), m, font=fonts['log'], fill=(225, 233, 246))
        y += 25

    d.text((720, 662), '*percentile is rough heuristic by public rating distributions', font=fonts['log'], fill=(150, 170, 205))

    img.save(out_dir / f'frame_{frame_id:04d}.png')


def render_game(game_idx: int, chess960_index: int):
    candidate_eval = load_eval(ROOT / 'src' / 'zero960' / 'workspace_template' / 'eval.py')
    board = chess.Board.from_chess960_pos(chess960_index)
    board.chess960 = True
    fonts = make_fonts()

    game_dir = OUT / f'game{game_idx+1}'
    frames_dir = game_dir / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)

    engine = chess.engine.SimpleEngine.popen_uci('stockfish')
    engine.configure({'UCI_LimitStrength': True, 'UCI_Elo': 1320})

    move_log: list[str] = []
    frame_id = 0
    draw_frame(board, move_log, game_idx, frame_id, fonts, frames_dir)
    frame_id += 1

    for _ in range(70):
        if board.is_game_over(claim_draw=True):
            break
        if board.turn == chess.WHITE:
            move = select_move(board, depth=2, eval_fn=candidate_eval)
            side = '0x960'
        else:
            res = engine.play(board, chess.engine.Limit(depth=1))
            move = res.move
            side = 'sf1320'
        if move is None:
            break
        san = board.san(move)
        mn = board.fullmove_number
        prefix = f'{mn}.' if board.turn == chess.WHITE else f'{mn}...'
        move_log.append(f'{prefix} {side}: {san}')
        board.push(move)
        for _ in range(6):
            draw_frame(board, move_log, game_idx, frame_id, fonts, frames_dir)
            frame_id += 1

    result = board.result(claim_draw=True)
    move_log.append(f'final result: {result} | plies: {board.ply()}')
    for _ in range(30):
        draw_frame(board, move_log, game_idx, frame_id, fonts, frames_dir)
        frame_id += 1

    (game_dir / 'moves.txt').write_text('\n'.join(move_log))
    engine.quit()


def main():
    render_game(0, 518)
    render_game(1, 123)
    print('rendered 2 games')


if __name__ == '__main__':
    main()
