#!/usr/bin/env python3
"""Play 0x960 engine vs Stockfish in Chess960 and record frames for video.

Outputs:
  - outputs/demo_game/frames/  — one PNG per ply (board + move annotation)
  - outputs/demo_game/game.pgn — full PGN
  - outputs/demo_game/game.gif — animated GIF of the game
  - prints a live board to the terminal as the game plays

Usage:
  uv run python scripts/demo_game.py
  uv run python scripts/demo_game.py --stockfish-elo 1320 --depth 2 --position 518
  uv run python scripts/demo_game.py --stockfish-elo 1600 --depth 2 --max-plies 80
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time

import chess
import chess.engine
import chess.svg

# -- add project root to path so we can import engine code --
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from zero960.engine.search import select_move, _GLOBAL_TT, _GLOBAL_HISTORY


def load_eval_fn(eval_path: str):
    """Dynamically load an evaluate() function from a .py file."""
    spec = importlib.util.spec_from_file_location("champion_eval", eval_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.evaluate


def render_board_terminal(board: chess.Board, last_move: chess.Move | None, ply: int, side: str, move_san: str, think_time: float):
    """Print a nice board + info to terminal."""
    print(f"\n{'='*50}")
    print(f"  Ply {ply}  |  {side}: {move_san}  |  {think_time:.2f}s")
    print(f"{'='*50}")
    print(board.unicode(borders=True, empty_square="·"))
    print()


def render_board_svg(board: chess.Board, last_move: chess.Move | None, ply: int, side: str, move_san: str, think_time: float, out_dir: str):
    """Save board as SVG, then convert to PNG using cairosvg."""
    arrows = []
    fill = {}
    if last_move:
        arrows = [chess.svg.Arrow(last_move.from_square, last_move.to_square, color="#22c55e80")]
        fill = {last_move.from_square: "#22c55e30", last_move.to_square: "#22c55e50"}

    if board.is_check():
        king_sq = board.king(board.turn)
        if king_sq is not None:
            fill[king_sq] = "#ef444460"

    title = f"Ply {ply} — {side}: {move_san} ({think_time:.2f}s)"
    svg_str = chess.svg.board(
        board,
        arrows=arrows,
        fill=fill,
        size=600,
        coordinates=True,
        colors={"square light": "#f0d9b5", "square dark": "#b58863"},
    )
    # inject title into SVG
    svg_str = svg_str.replace("<svg ", f'<svg <!-- {title} --> ', 1)

    svg_path = os.path.join(out_dir, f"frame_{ply:04d}.svg")
    png_path = os.path.join(out_dir, f"frame_{ply:04d}.png")

    with open(svg_path, "w") as f:
        f.write(svg_str)

    try:
        import cairosvg
        cairosvg.svg2png(bytestring=svg_str.encode(), write_to=png_path, output_width=800, output_height=800)
    except ImportError:
        # fallback: just keep SVGs
        pass

    return png_path if os.path.exists(png_path) else svg_path


def make_gif(frame_dir: str, out_path: str, duration_ms: int = 800):
    """Combine PNGs into an animated GIF."""
    try:
        from PIL import Image
    except ImportError:
        print("  [skip gif — pip install Pillow for GIF output]")
        return

    frames = sorted(f for f in os.listdir(frame_dir) if f.endswith(".png"))
    if not frames:
        print("  [skip gif — no PNG frames found]")
        return

    images = [Image.open(os.path.join(frame_dir, f)) for f in frames]
    # hold on last frame longer
    durations = [duration_ms] * len(images)
    durations[-1] = 3000

    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
    )
    print(f"  GIF saved: {out_path} ({len(images)} frames)")


def main():
    parser = argparse.ArgumentParser(description="0x960 vs Stockfish demo game")
    parser.add_argument("--stockfish-elo", type=int, default=1320, help="Stockfish UCI_Elo setting")
    parser.add_argument("--depth", type=int, default=2, help="0x960 search depth")
    parser.add_argument("--position", type=int, default=518, help="Chess960 start position (0-959, 518=standard)")
    parser.add_argument("--max-plies", type=int, default=120, help="Max plies before draw")
    parser.add_argument("--stockfish-time", type=float, default=0.5, help="Stockfish time limit per move (seconds)")
    parser.add_argument("--stockfish-command", type=str, default="stockfish", help="Path to stockfish binary")
    parser.add_argument("--eval-file", type=str, default=None, help="Path to eval.py (default: champion or workspace template)")
    parser.add_argument("--our-color", type=str, default="white", choices=["white", "black"], help="Which color 0x960 plays")
    parser.add_argument("--output-dir", type=str, default="outputs/demo_game", help="Output directory")
    parser.add_argument("--frame-delay", type=int, default=800, help="GIF frame delay in ms")
    args = parser.parse_args()

    # resolve eval file
    if args.eval_file:
        eval_path = args.eval_file
    elif os.path.exists(os.path.join(PROJECT_ROOT, "outputs/codex_swarm/champion_eval.py")):
        eval_path = os.path.join(PROJECT_ROOT, "outputs/codex_swarm/champion_eval.py")
    else:
        eval_path = os.path.join(PROJECT_ROOT, "src/zero960/workspace_template/eval.py")

    print(f"\n{'#'*60}")
    print(f"#  0x960 vs Stockfish {args.stockfish_elo} — Chess960 Demo Game")
    print(f"#  Position: {args.position}  |  Depth: {args.depth}  |  Our color: {args.our_color}")
    print(f"#  Eval: {os.path.basename(eval_path)}")
    print(f"{'#'*60}\n")

    eval_fn = load_eval_fn(eval_path)
    our_white = args.our_color == "white"

    # setup output dirs
    frame_dir = os.path.join(args.output_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)

    # setup board
    board = chess.Board.from_chess960_pos(args.position)
    board.chess960 = True

    # setup stockfish
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_command)
    engine.configure({"UCI_Chess960": True, "UCI_LimitStrength": True, "UCI_Elo": args.stockfish_elo})

    # clear global TT for fresh game
    _GLOBAL_TT.clear()
    _GLOBAL_HISTORY.clear()

    # game loop
    moves_list = []
    ply = 0

    # render initial position
    render_board_svg(board, None, 0, "Start", f"Chess960 #{args.position}", 0.0, frame_dir)

    print(f"  Starting position (Chess960 #{args.position}):")
    print(board.unicode(borders=True, empty_square="·"))
    print()

    while not board.is_game_over(claim_draw=True) and ply < args.max_plies:
        is_our_turn = (board.turn == chess.WHITE) == our_white

        if is_our_turn:
            side = "0x960"
            t0 = time.time()
            move = select_move(board, depth=args.depth, eval_fn=eval_fn)
            think_time = time.time() - t0
        else:
            side = f"Stockfish {args.stockfish_elo}"
            t0 = time.time()
            result = engine.play(board, chess.engine.Limit(time=args.stockfish_time))
            move = result.move
            think_time = time.time() - t0

        move_san = board.san(move)
        board.push(move)
        ply += 1

        moves_list.append({"ply": ply, "side": side, "move": move_san, "time": think_time})

        render_board_terminal(board, move, ply, side, move_san, think_time)
        render_board_svg(board, move, ply, side, move_san, think_time, frame_dir)

    engine.quit()

    # result
    result = board.result(claim_draw=True)
    if result == "1-0":
        winner = "White wins" + (" — 0x960 wins!" if our_white else " — Stockfish wins")
    elif result == "0-1":
        winner = "Black wins" + (" — 0x960 wins!" if not our_white else " — Stockfish wins")
    else:
        winner = "Draw"

    print(f"\n{'='*60}")
    print(f"  GAME OVER — {result} — {winner}")
    print(f"  {ply} plies played")
    print(f"{'='*60}\n")

    # save PGN
    pgn_path = os.path.join(args.output_dir, "game.pgn")
    white_name = "0x960" if our_white else f"Stockfish {args.stockfish_elo}"
    black_name = f"Stockfish {args.stockfish_elo}" if our_white else "0x960"
    with open(pgn_path, "w") as f:
        f.write(f'[Event "0x960 Demo Game"]\n')
        f.write(f'[Site "Local"]\n')
        f.write(f'[White "{white_name}"]\n')
        f.write(f'[Black "{black_name}"]\n')
        f.write(f'[Result "{result}"]\n')
        f.write(f'[Variant "Chess960"]\n')
        f.write(f'[FEN "{chess.Board.from_chess960_pos(args.position).fen()}"]\n')
        f.write(f'[SetUp "1"]\n\n')
        # write moves
        move_strs = []
        for i, m in enumerate(moves_list):
            if i % 2 == 0:
                move_strs.append(f"{i // 2 + 1}. {m['move']}")
            else:
                move_strs[-1] += f" {m['move']}"
        f.write(" ".join(move_strs))
        f.write(f" {result}\n")
    print(f"  PGN saved: {pgn_path}")

    # save move log
    log_path = os.path.join(args.output_dir, "game_log.txt")
    with open(log_path, "w") as f:
        f.write(f"0x960 vs Stockfish {args.stockfish_elo} — Chess960 #{args.position}\n")
        f.write(f"0x960 plays: {args.our_color} | Depth: {args.depth}\n")
        f.write(f"Result: {result} — {winner}\n\n")
        for m in moves_list:
            f.write(f"Ply {m['ply']:3d}  {m['side']:<25s}  {m['move']:<10s}  {m['time']:.2f}s\n")
    print(f"  Log saved: {log_path}")

    # make GIF
    gif_path = os.path.join(args.output_dir, "game.gif")
    make_gif(frame_dir, gif_path, duration_ms=args.frame_delay)

    print(f"\n  All outputs in: {args.output_dir}/")
    print(f"  To make a video from frames:")
    print(f"    ffmpeg -framerate 2 -i {frame_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {args.output_dir}/game.mp4\n")


if __name__ == "__main__":
    main()
