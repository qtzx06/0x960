"""Benchmark two full engine roots so each side uses its own search and eval code."""

from __future__ import annotations

import argparse
import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import chess

from train.benchmark_eval import BenchmarkResult, _elo_from_score, _sample_positions

EvalFn = Callable[[chess.Board], int]
SelectMoveFn = Callable[[chess.Board, int, EvalFn], chess.Move]


@dataclass(slots=True)
class EngineHandle:
    root: Path
    eval_path: Path
    search_path: Path
    evaluate: EvalFn
    select_move: SelectMoveFn


def _load_module(path: Path, module_name: str) -> object:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_engine(root: Path, eval_rel: str, search_rel: str, label: str) -> EngineHandle:
    eval_path = (root / eval_rel).resolve()
    search_path = (root / search_rel).resolve()
    eval_module = _load_module(eval_path, f"zero960_eval_{label}")
    search_module = _load_module(search_path, f"zero960_search_{label}")

    evaluate = getattr(eval_module, "evaluate", None)
    select_move = getattr(search_module, "select_move", None)
    if evaluate is None or not callable(evaluate):
        raise RuntimeError(f"{eval_path} does not define evaluate(board)")
    if select_move is None or not callable(select_move):
        raise RuntimeError(f"{search_path} does not define select_move(board, depth, eval_fn)")

    return EngineHandle(
        root=root.resolve(),
        eval_path=eval_path,
        search_path=search_path,
        evaluate=evaluate,
        select_move=select_move,
    )


def _new_board(chess960_index: int) -> chess.Board:
    board = chess.Board.from_chess960_pos(chess960_index)
    board.chess960 = True
    return board


def _play_game(
    chess960_index: int,
    white_engine: EngineHandle,
    black_engine: EngineHandle,
    *,
    depth: int,
    max_plies: int,
) -> float:
    board = _new_board(chess960_index)

    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        engine = white_engine if board.turn == chess.WHITE else black_engine
        move = engine.select_move(board, depth=depth, eval_fn=engine.evaluate)
        board.push(move)

    result = board.result(claim_draw=True)
    if result == "1-0":
        return 1.0
    if result == "0-1":
        return 0.0
    return 0.5


def benchmark_engine_roots(
    candidate_root: Path,
    baseline_root: Path,
    *,
    candidate_eval_rel: str = "src/zero960/workspace_template/eval.py",
    baseline_eval_rel: str = "src/zero960/workspace_template/eval.py",
    candidate_search_rel: str = "src/zero960/engine/search.py",
    baseline_search_rel: str = "src/zero960/engine/search.py",
    positions: int = 64,
    depth: int = 2,
    max_plies: int = 120,
    seed: int = 42,
) -> BenchmarkResult:
    candidate = _load_engine(candidate_root, candidate_eval_rel, candidate_search_rel, "candidate")
    baseline = _load_engine(baseline_root, baseline_eval_rel, baseline_search_rel, "baseline")
    start_positions = _sample_positions(positions, seed)

    wins = 0
    draws = 0
    losses = 0
    points = 0.0

    for chess960_index in start_positions:
        white_result = _play_game(
            chess960_index,
            candidate,
            baseline,
            depth=depth,
            max_plies=max_plies,
        )
        points += white_result
        if white_result == 1.0:
            wins += 1
        elif white_result == 0.5:
            draws += 1
        else:
            losses += 1

        black_result = 1.0 - _play_game(
            chess960_index,
            baseline,
            candidate,
            depth=depth,
            max_plies=max_plies,
        )
        points += black_result
        if black_result == 1.0:
            wins += 1
        elif black_result == 0.5:
            draws += 1
        else:
            losses += 1

    total_games = len(start_positions) * 2
    score = points / total_games if total_games else 0.0
    return BenchmarkResult(
        candidate_path=candidate.root,
        baseline_path=baseline.root,
        positions=len(start_positions),
        depth=depth,
        max_plies=max_plies,
        seed=seed,
        wins=wins,
        draws=draws,
        losses=losses,
        points=points,
        total_games=total_games,
        score=score,
        elo_delta_estimate=_elo_from_score(score),
    )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", default=str(root))
    parser.add_argument("--baseline-root", default=str(root))
    parser.add_argument("--candidate-eval-rel", default="src/zero960/workspace_template/eval.py")
    parser.add_argument("--baseline-eval-rel", default="src/zero960/workspace_template/eval.py")
    parser.add_argument("--candidate-search-rel", default="src/zero960/engine/search.py")
    parser.add_argument("--baseline-search-rel", default="src/zero960/engine/search.py")
    parser.add_argument("--positions", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--max-plies", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = benchmark_engine_roots(
        Path(args.candidate_root).resolve(),
        Path(args.baseline_root).resolve(),
        candidate_eval_rel=args.candidate_eval_rel,
        baseline_eval_rel=args.baseline_eval_rel,
        candidate_search_rel=args.candidate_search_rel,
        baseline_search_rel=args.baseline_search_rel,
        positions=args.positions,
        depth=args.depth,
        max_plies=args.max_plies,
        seed=args.seed,
    )

    print(f"candidate_root: {result.candidate_path}")
    print(f"baseline_root:  {result.baseline_path}")
    print(
        f"positions={result.positions} depth={result.depth} max_plies={result.max_plies} "
        f"games={result.total_games} seed={result.seed}"
    )
    print(
        f"record={result.wins}-{result.draws}-{result.losses} "
        f"points={result.points:.1f}/{result.total_games}"
    )
    print(f"score={result.score:.3f} elo_delta_estimate={result.elo_delta_estimate:.1f}")


if __name__ == "__main__":
    main()
