"""Benchmark a candidate eval against a league of accepted swarm champions."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from train.benchmark_eval import BenchmarkResult, benchmark_eval_files


@dataclass(slots=True)
class LeagueOpponentResult:
    opponent_path: Path
    label: str
    result: BenchmarkResult

    def to_json(self) -> dict[str, object]:
        payload = asdict(self)
        payload["opponent_path"] = str(self.opponent_path)
        payload["result"] = self.result.to_json()
        return payload


@dataclass(slots=True)
class LeagueResult:
    candidate_path: Path
    opponents: list[LeagueOpponentResult]
    total_points: float
    total_games: int
    overall_score: float
    overall_elo_delta_estimate: float

    def to_json(self) -> dict[str, object]:
        return {
            "candidate_path": str(self.candidate_path),
            "opponents": [opponent.to_json() for opponent in self.opponents],
            "total_points": self.total_points,
            "total_games": self.total_games,
            "overall_score": self.overall_score,
            "overall_elo_delta_estimate": self.overall_elo_delta_estimate,
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_candidate(root: Path) -> Path:
    return root / "outputs" / "codex_swarm" / "champion_eval.py"


def _default_baseline(root: Path) -> Path:
    return root / "src" / "zero960" / "engine" / "default_eval.py"


def _accepted_snapshots(root: Path) -> list[Path]:
    accepted_dir = root / "outputs" / "codex_swarm" / "accepted"
    if not accepted_dir.exists():
        return []
    return sorted(accepted_dir.glob("*_eval.py"))


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        ordered.append(resolved)
    return ordered


def _same_contents(left: Path, right: Path) -> bool:
    return left.read_text(encoding="utf-8") == right.read_text(encoding="utf-8")


def _label_for_path(root: Path, path: Path) -> str:
    resolved = path.resolve()
    champion = (root / "outputs" / "codex_swarm" / "champion_eval.py").resolve()
    baseline = (root / "src" / "zero960" / "engine" / "default_eval.py").resolve()
    if resolved == champion:
        return "current_champion"
    if resolved == baseline:
        return "original_baseline"
    if "outputs/codex_swarm/accepted" in str(resolved):
        return resolved.stem
    return resolved.stem


def default_league_opponents(
    *,
    candidate_path: Path,
    include_baseline: bool,
    include_champion: bool,
    accepted_limit: int | None,
) -> list[Path]:
    root = _repo_root()
    opponents: list[Path] = []
    if include_baseline:
        opponents.append(_default_baseline(root))
    if include_champion:
        opponents.append(_default_candidate(root))

    accepted = _accepted_snapshots(root)
    if accepted_limit is not None:
        accepted = accepted[-accepted_limit:]
    opponents.extend(accepted)

    filtered = []
    for path in _dedupe_paths(opponents):
        if path.resolve() == candidate_path.resolve():
            continue
        if _same_contents(path, candidate_path):
            continue
        filtered.append(path)
    return filtered


def benchmark_league(
    candidate_path: Path,
    opponent_paths: list[Path],
    *,
    positions: int,
    depth: int,
    max_plies: int,
    seed: int,
) -> LeagueResult:
    root = _repo_root()
    opponent_results: list[LeagueOpponentResult] = []
    total_points = 0.0
    total_games = 0

    for offset, opponent_path in enumerate(opponent_paths):
        result = benchmark_eval_files(
            candidate_path,
            opponent_path,
            positions=positions,
            depth=depth,
            max_plies=max_plies,
            seed=seed + offset,
        )
        opponent_results.append(
            LeagueOpponentResult(
                opponent_path=opponent_path,
                label=_label_for_path(root, opponent_path),
                result=result,
            )
        )
        total_points += result.points
        total_games += result.total_games

    overall_score = total_points / total_games if total_games else 0.0
    overall_elo = 0.0
    if total_games:
        from train.benchmark_eval import _elo_from_score  # local reuse

        overall_elo = _elo_from_score(overall_score)

    return LeagueResult(
        candidate_path=candidate_path,
        opponents=opponent_results,
        total_points=total_points,
        total_games=total_games,
        overall_score=overall_score,
        overall_elo_delta_estimate=overall_elo,
    )


def parse_args() -> argparse.Namespace:
    root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-file",
        default=str(_default_candidate(root)),
        help="Path to the candidate eval.py file.",
    )
    parser.add_argument(
        "--opponent-file",
        action="append",
        default=[],
        help="Optional explicit opponent file. Repeat to add more than one.",
    )
    parser.add_argument("--positions", type=int, default=16)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--max-plies", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--accepted-limit",
        type=int,
        default=4,
        help="How many accepted swarm snapshots to include by default.",
    )
    parser.add_argument("--no-baseline", action="store_true", help="Exclude the original baseline from the league.")
    parser.add_argument("--no-champion", action="store_true", help="Exclude the current champion from the league.")
    parser.add_argument("--json", action="store_true", help="Print the full result as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_path = Path(args.candidate_file).resolve()
    explicit_opponents = [Path(path).resolve() for path in args.opponent_file]

    opponents = _dedupe_paths(explicit_opponents)
    if not opponents:
        opponents = default_league_opponents(
            candidate_path=candidate_path,
            include_baseline=not args.no_baseline,
            include_champion=not args.no_champion,
            accepted_limit=args.accepted_limit,
        )

    if not opponents:
        raise SystemExit("No league opponents found.")

    result = benchmark_league(
        candidate_path,
        opponents,
        positions=args.positions,
        depth=args.depth,
        max_plies=args.max_plies,
        seed=args.seed,
    )

    if args.json:
        print(json.dumps(result.to_json(), indent=2, sort_keys=True))
        return

    print(f"candidate: {candidate_path}")
    print(f"league opponents: {len(result.opponents)}")
    for opponent in result.opponents:
        benchmark = opponent.result
        print(
            f"- {opponent.label}: record={benchmark.wins}-{benchmark.draws}-{benchmark.losses} "
            f"points={benchmark.points:.1f}/{benchmark.total_games} score={benchmark.score:.3f} "
            f"elo_delta_estimate={benchmark.elo_delta_estimate:.1f}"
        )
    print(
        f"overall: points={result.total_points:.1f}/{result.total_games} "
        f"score={result.overall_score:.3f} elo_delta_estimate={result.overall_elo_delta_estimate:.1f}"
    )


if __name__ == "__main__":
    main()
