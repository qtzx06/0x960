"""Benchmark two Chess960 eval functions against each other."""

from __future__ import annotations

import argparse
import importlib.util
import math
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import chess

from zero960.engine.match import play_game

EvalFn = Callable[[chess.Board], int]


@dataclass(slots=True)
class BenchmarkResult:
    candidate_path: Path
    baseline_path: Path
    positions: int
    depth: int
    max_plies: int
    seed: int
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
        payload["baseline_path"] = str(self.baseline_path)
        return payload


def _load_eval(path: Path) -> EvalFn:
    spec = importlib.util.spec_from_file_location(f"zero960_benchmark_{path.stem}", path)
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


def benchmark_eval_files(
    candidate_path: Path,
    baseline_path: Path,
    *,
    positions: int = 64,
    depth: int = 2,
    max_plies: int = 120,
    seed: int = 42,
) -> BenchmarkResult:
    candidate_eval = _load_eval(candidate_path)
    baseline_eval = _load_eval(baseline_path)
    start_positions = _sample_positions(positions, seed)

    wins = 0
    draws = 0
    losses = 0
    points = 0.0

    for chess960_index in start_positions:
        white_result = play_game(
            chess960_index,
            candidate_eval,
            baseline_eval,
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

        black_result = 1.0 - play_game(
            chess960_index,
            baseline_eval,
            candidate_eval,
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
        candidate_path=candidate_path,
        baseline_path=baseline_path,
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
    parser = argparse.ArgumentParser(description="Benchmark two Chess960 eval functions.")
    parser.add_argument(
        "--candidate-file",
        default=str(root / "src/zero960/workspace_template/eval.py"),
        help="Path to the candidate eval.py file.",
    )
    parser.add_argument(
        "--baseline-file",
        default=str(root / "src/zero960/engine/default_eval.py"),
        help="Path to the baseline eval.py file.",
    )
    parser.add_argument("--positions", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--max-plies", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_path = Path(args.candidate_file).resolve()
    baseline_path = Path(args.baseline_file).resolve()
    result = benchmark_eval_files(
        candidate_path,
        baseline_path,
        positions=args.positions,
        depth=args.depth,
        max_plies=args.max_plies,
        seed=args.seed,
    )

    print(f"candidate: {candidate_path}")
    print(f"baseline:  {baseline_path}")
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
