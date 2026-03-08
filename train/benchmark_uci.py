"""Benchmark a local Chess960 eval file against a UCI engine such as Stockfish."""

from __future__ import annotations

import argparse
import importlib.util
import math
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import chess
import chess.engine

from zero960.engine.search import select_move

EvalFn = Callable[[chess.Board], int]


@dataclass(slots=True)
class UciBenchmarkResult:
    candidate_path: Path
    engine_command: str
    engine_options: dict[str, bool | int | float | str]
    positions: int
    max_plies: int
    seed: int
    candidate_depth: int | None
    candidate_nodes: int | None
    engine_depth: int | None
    engine_nodes: int | None
    wins: int
    draws: int
    losses: int
    points: float
    total_games: int
    score: float
    elo_delta_estimate: float

    def to_json(self) -> dict[str, object]:
        payload = asdict(self)
        payload["candidate_path"] = str(self.candidate_path)
        return payload


def _load_eval(path: Path) -> EvalFn:
    spec = importlib.util.spec_from_file_location(f"zero960_uci_benchmark_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    evaluate = getattr(module, "evaluate", None)
    if evaluate is None or not callable(evaluate):
        raise RuntimeError(f"{path} does not define evaluate(board)")
    return evaluate


def _sample_positions(count: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    population = list(range(960))
    if count <= len(population):
        return rng.sample(population, count)
    return [rng.choice(population) for _ in range(count)]


def _elo_from_score(score: float) -> float:
    clipped = min(max(score, 0.01), 0.99)
    return -400.0 * math.log10((1.0 / clipped) - 1.0)


def _new_board(chess960_index: int) -> chess.Board:
    board = chess.Board.from_chess960_pos(chess960_index)
    board.chess960 = True
    return board


def _engine_limit(depth: int | None, nodes: int | None) -> chess.engine.Limit:
    if depth is not None:
        return chess.engine.Limit(depth=depth)
    if nodes is not None:
        return chess.engine.Limit(nodes=nodes)
    raise ValueError("expected depth or nodes limit")


def _parse_option_value(raw_value: str) -> bool | int | float | str:
    lowered = raw_value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(raw_value)
    except ValueError:
        pass
    try:
        return float(raw_value)
    except ValueError:
        pass
    return raw_value


def _parse_engine_options(pairs: list[str]) -> dict[str, bool | int | float | str]:
    options: dict[str, bool | int | float | str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"invalid --engine-option {pair!r}; expected NAME=VALUE")
        name, raw_value = pair.split("=", 1)
        option_name = name.strip()
        if not option_name:
            raise ValueError(f"invalid --engine-option {pair!r}; missing option name")
        options[option_name] = _parse_option_value(raw_value.strip())
    return options


def _play_game_vs_engine(
    chess960_index: int,
    candidate_eval: EvalFn,
    engine: chess.engine.SimpleEngine,
    *,
    candidate_is_white: bool,
    candidate_depth: int | None,
    candidate_nodes: int | None,
    engine_depth: int | None,
    engine_nodes: int | None,
    max_plies: int,
) -> float:
    board = _new_board(chess960_index)
    candidate_limit = _engine_limit(candidate_depth, candidate_nodes)
    opponent_limit = _engine_limit(engine_depth, engine_nodes)

    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break

        candidate_turn = board.turn == chess.WHITE if candidate_is_white else board.turn == chess.BLACK
        if candidate_turn:
            if candidate_limit.depth is not None:
                move = select_move(board, depth=candidate_limit.depth, eval_fn=candidate_eval)
            else:
                raise ValueError("candidate_nodes is not supported by the local engine path")
        else:
            result = engine.play(board, opponent_limit)
            move = result.move
            if move is None:
                raise RuntimeError("UCI engine returned no move")

        board.push(move)

    result = board.result(claim_draw=True)
    if result == "1-0":
        return 1.0 if candidate_is_white else 0.0
    if result == "0-1":
        return 0.0 if candidate_is_white else 1.0
    return 0.5


def benchmark_eval_vs_uci(
    candidate_path: Path,
    engine_command: str,
    *,
    engine_options: dict[str, bool | int | float | str] | None = None,
    positions: int = 32,
    candidate_depth: int = 2,
    candidate_nodes: int | None = None,
    engine_depth: int = 1,
    engine_nodes: int | None = None,
    max_plies: int = 120,
    seed: int = 42,
) -> UciBenchmarkResult:
    candidate_eval = _load_eval(candidate_path)
    start_positions = _sample_positions(positions, seed)
    configured_engine_options = dict(engine_options or {})

    wins = 0
    draws = 0
    losses = 0
    points = 0.0

    with chess.engine.SimpleEngine.popen_uci(engine_command) as engine:
        if configured_engine_options:
            engine.configure(configured_engine_options)
        for chess960_index in start_positions:
            white_result = _play_game_vs_engine(
                chess960_index,
                candidate_eval,
                engine,
                candidate_is_white=True,
                candidate_depth=candidate_depth,
                candidate_nodes=candidate_nodes,
                engine_depth=engine_depth,
                engine_nodes=engine_nodes,
                max_plies=max_plies,
            )
            points += white_result
            if white_result == 1.0:
                wins += 1
            elif white_result == 0.5:
                draws += 1
            else:
                losses += 1

            black_result = _play_game_vs_engine(
                chess960_index,
                candidate_eval,
                engine,
                candidate_is_white=False,
                candidate_depth=candidate_depth,
                candidate_nodes=candidate_nodes,
                engine_depth=engine_depth,
                engine_nodes=engine_nodes,
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
    return UciBenchmarkResult(
        candidate_path=candidate_path,
        engine_command=engine_command,
        engine_options=configured_engine_options,
        positions=len(start_positions),
        max_plies=max_plies,
        seed=seed,
        candidate_depth=candidate_depth,
        candidate_nodes=candidate_nodes,
        engine_depth=engine_depth,
        engine_nodes=engine_nodes,
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
    parser = argparse.ArgumentParser(description="Benchmark a local eval file against a UCI engine.")
    parser.add_argument(
        "--candidate-file",
        default=str(root / "src/zero960/workspace_template/eval.py"),
        help="Path to the candidate eval.py file.",
    )
    parser.add_argument(
        "--engine-command",
        default="stockfish",
        help="UCI engine command, for example 'stockfish'.",
    )
    parser.add_argument(
        "--engine-option",
        action="append",
        default=[],
        help="Repeated engine option in NAME=VALUE form, for example UCI_LimitStrength=true.",
    )
    parser.add_argument("--positions", type=int, default=32)
    parser.add_argument("--candidate-depth", type=int, default=2)
    parser.add_argument("--candidate-nodes", type=int, default=None)
    parser.add_argument("--engine-depth", type=int, default=1)
    parser.add_argument("--engine-nodes", type=int, default=None)
    parser.add_argument("--max-plies", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_path = Path(args.candidate_file).resolve()
    engine_options = _parse_engine_options(args.engine_option)
    result = benchmark_eval_vs_uci(
        candidate_path,
        args.engine_command,
        engine_options=engine_options,
        positions=args.positions,
        candidate_depth=args.candidate_depth,
        candidate_nodes=args.candidate_nodes,
        engine_depth=args.engine_depth,
        engine_nodes=args.engine_nodes,
        max_plies=args.max_plies,
        seed=args.seed,
    )

    print(f"candidate: {result.candidate_path}")
    print(f"engine:    {result.engine_command}")
    if result.engine_options:
        print(f"engine_options={result.engine_options}")
    print(
        f"positions={result.positions} max_plies={result.max_plies} games={result.total_games} seed={result.seed} "
        f"candidate_depth={result.candidate_depth} engine_depth={result.engine_depth} "
        f"candidate_nodes={result.candidate_nodes} engine_nodes={result.engine_nodes}"
    )
    print(
        f"record={result.wins}-{result.draws}-{result.losses} "
        f"points={result.points:.1f}/{result.total_games}"
    )
    print(f"score={result.score:.3f} elo_delta_estimate={result.elo_delta_estimate:.1f}")


if __name__ == "__main__":
    main()
