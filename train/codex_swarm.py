"""Local Codex swarm coordinator for champion/challenger engine iteration."""

from __future__ import annotations

import argparse
import difflib
import json
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from train.benchmark_engine import benchmark_engine_roots
from train.benchmark_eval import BenchmarkResult, benchmark_eval_files

DEFAULT_MODEL = "gpt-5.3-codex"
DEFAULT_WORKER_COUNT = 5
DEFAULT_SCREEN_POSITIONS = 8
DEFAULT_POSITIONS = 32
DEFAULT_DEPTH = 2
DEFAULT_MAX_PLIES = 120
DEFAULT_SEARCH_SCREEN_POSITIONS = 1
DEFAULT_SEARCH_SCREEN_DEPTH = 1
DEFAULT_SEARCH_SCREEN_MAX_PLIES = 20
DEFAULT_FINAL_BENCHMARK_TIMEOUT_SEC = 180
DEFAULT_SCREEN_MIN_SCORE = 0.52
DEFAULT_MIN_SCORE = 0.53
DEFAULT_MAX_DIFF_LINES = 80
DEFAULT_EDITABLE_FILES = ("src/zero960/workspace_template/eval.py",)
DEFAULT_WORKER_TIMEOUT_SEC = 600
DEFAULT_BENCHMARK_TIMEOUT_SEC = 180
DEFAULT_REFERENCE_PATHS = (
    "AGENTS.md",
    "README.md",
    "docs/codex-swarm-plan.md",
    "docs/process.md",
    "train/benchmark_eval.py",
    "train/benchmark_engine.py",
    "train/benchmark_league.py",
    "train/benchmark_uci.py",
    "src/zero960/engine/default_eval.py",
    "src/zero960/engine/search.py",
)
DEFAULT_SYNC_PATHS = (
    *DEFAULT_REFERENCE_PATHS,
    "src/zero960/workspace_template/eval.py",
)
DEFAULT_SURFACE = "eval"
DEFAULT_EVAL_EDITABLE_FILES = ("src/zero960/workspace_template/eval.py",)
DEFAULT_SEARCH_EDITABLE_FILES = ("src/zero960/engine/search.py",)
DEFAULT_WORKER_SPECIALIZATIONS = (
    (
        "Structure Researcher",
        "Study Chess960-specific king safety, castling structure, and pawn cover weaknesses around both kings.",
        "_structure_hook",
    ),
    (
        "Pawn-Endgame Researcher",
        "Study pawn structure, passed-pawn pressure, rook-file coordination, and simple endgame conversion bonuses.",
        "_pawn_endgame_hook",
    ),
    (
        "Initiative Tuner",
        "Study tempo, mobility pressure, queen safety, and initiative terms that might convert shallow-search advantages faster.",
        "_initiative_hook",
    ),
    (
        "Activity Researcher",
        "Study piece activity, development, space, and centralization terms that help when search depth is limited.",
        "_activity_hook",
    ),
    (
        "Tactical Safety Researcher",
        "Study loose-piece pressure, attacked-undefended pieces, and tactical safety terms that matter at shallow search depth.",
        "_tactical_hook",
    ),
)
DEFAULT_SEARCH_SPECIALIZATIONS = (
    (
        "Move Ordering Researcher",
        "Study move ordering, capture ordering, and cheap heuristics that push strong moves to the front early.",
        "_move_order_score",
    ),
    (
        "Quiescence Researcher",
        "Study tactical horizon control and leaf evaluation extension without exploding the tree.",
        "_quiescence",
    ),
    (
        "Tree Search Researcher",
        "Study alpha-beta search control, pruning safety, and transposition-table usage in the main negamax loop.",
        "negamax",
    ),
    (
        "Root Policy Researcher",
        "Study root move selection, aspiration behavior, and tie-breaking that helps shallow search convert edges.",
        "select_move",
    ),
    (
        "Tactical Move Filter Researcher",
        "Study which tactical moves should survive into the quiescence frontier without causing pointless explosion.",
        "_tactical_moves",
    ),
)


@dataclass(slots=True)
class SwarmPaths:
    repo_root: Path
    state_root: Path
    worktree_root: Path
    champion_eval: Path
    champion_search: Path
    ledger_path: Path


@dataclass(slots=True)
class WorkerResult:
    worker_name: str
    worktree_dir: Path
    round_dir: Path
    prompt_path: Path
    final_message_path: Path
    stdout_path: Path
    stderr_path: Path
    candidate_file: Path
    changed_files: list[str]
    diff_lines_added: int
    diff_lines_deleted: int
    screen_benchmark: BenchmarkResult | None
    benchmark: BenchmarkResult | None
    exit_code: int | None
    accepted: bool
    summary: str
    sandbox_mode: str

    def to_json(self) -> dict[str, object]:
        return {
            "worker_name": self.worker_name,
            "worktree_dir": str(self.worktree_dir),
            "round_dir": str(self.round_dir),
            "prompt_path": str(self.prompt_path),
            "final_message_path": str(self.final_message_path),
            "stdout_path": str(self.stdout_path),
            "stderr_path": str(self.stderr_path),
            "candidate_file": str(self.candidate_file),
            "changed_files": self.changed_files,
            "diff_lines_added": self.diff_lines_added,
            "diff_lines_deleted": self.diff_lines_deleted,
            "screen_benchmark": None if self.screen_benchmark is None else self.screen_benchmark.to_json(),
            "benchmark": None if self.benchmark is None else self.benchmark.to_json(),
            "exit_code": self.exit_code,
            "accepted": self.accepted,
            "summary": self.summary,
            "sandbox_mode": self.sandbox_mode,
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_paths() -> SwarmPaths:
    root = _repo_root()
    state_root = root / "outputs" / "codex_swarm"
    return SwarmPaths(
        repo_root=root,
        state_root=state_root,
        worktree_root=Path("/tmp") / "0x960-codex-swarm",
        champion_eval=state_root / "champion_eval.py",
        champion_search=state_root / "champion_search.py",
        ledger_path=state_root / "ledger.jsonl",
    )


def _run(
    args: list[str],
    *,
    cwd: Path,
    capture_output: bool = True,
    check: bool = True,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        input=input_text,
        text=True,
        capture_output=capture_output,
        check=check,
    )


def _git_output(repo_root: Path, args: list[str]) -> str:
    result = _run(["git", *args], cwd=repo_root)
    return result.stdout.strip()


def _ensure_state_dirs(paths: SwarmPaths) -> None:
    paths.state_root.mkdir(parents=True, exist_ok=True)
    (paths.state_root / "runs").mkdir(parents=True, exist_ok=True)
    (paths.state_root / "accepted").mkdir(parents=True, exist_ok=True)


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)


def _prepare_worker_dir(worker_dir: Path) -> Path:
    if worker_dir.exists() and not any(worker_dir.iterdir()):
        worker_dir.rmdir()
    return worker_dir


def _infer_repo_mode(worker_dir: Path) -> str:
    git_path = worker_dir / ".git"
    if git_path.is_file():
        return "worktree"
    if git_path.is_dir():
        return "clone"
    return "unknown"


def _sync_worker_snapshot(paths: SwarmPaths, worker_dir: Path, sync_paths: tuple[str, ...]) -> None:
    for rel_path in sync_paths:
        src = paths.repo_root / rel_path
        dst = worker_dir / rel_path
        if src.is_file():
            _copy_file(src, dst)
    _copy_file(paths.champion_eval, worker_dir / "src" / "zero960" / "workspace_template" / "eval.py")
    _copy_file(paths.champion_search, worker_dir / "src" / "zero960" / "engine" / "search.py")
    accepted_src = paths.state_root / "accepted"
    accepted_dst = worker_dir / "outputs" / "codex_swarm" / "accepted"
    if accepted_src.exists():
        _copy_tree(accepted_src, accepted_dst)
    else:
        accepted_dst.mkdir(parents=True, exist_ok=True)
    _copy_file(paths.champion_eval, worker_dir / "outputs" / "codex_swarm" / "champion_eval.py")
    _copy_file(paths.champion_search, worker_dir / "outputs" / "codex_swarm" / "champion_search.py")
    ledger_copy = worker_dir / "outputs" / "codex_swarm" / "ledger.jsonl"
    ledger_copy.parent.mkdir(parents=True, exist_ok=True)
    if paths.ledger_path.exists():
        shutil.copy2(paths.ledger_path, ledger_copy)
    else:
        ledger_copy.write_text("", encoding="utf-8")


def _setup_workers(paths: SwarmPaths, worker_count: int, sync_paths: tuple[str, ...]) -> list[tuple[Path, str]]:
    worker_dirs: list[tuple[Path, str]] = []
    paths.worktree_root.mkdir(parents=True, exist_ok=True)
    for worker_index in range(1, worker_count + 1):
        worker_dir = paths.worktree_root / f"worker-{worker_index}"
        sandbox_mode = "existing"
        if not (worker_dir / ".git").exists():
            worker_dir = _prepare_worker_dir(worker_dir)
            try:
                _run(
                    ["git", "worktree", "add", "--detach", str(worker_dir), "HEAD"],
                    cwd=paths.repo_root,
                )
                sandbox_mode = "worktree"
            except subprocess.CalledProcessError:
                worker_dir = _prepare_worker_dir(worker_dir)
                _run(
                    ["git", "clone", "--shared", str(paths.repo_root), str(worker_dir)],
                    cwd=paths.repo_root,
                )
                sandbox_mode = "clone"
        else:
            sandbox_mode = _infer_repo_mode(worker_dir)
        _sync_worker_snapshot(paths, worker_dir, sync_paths)
        worker_dirs.append((worker_dir, sandbox_mode))
    return worker_dirs


def _last_ledger_entries(paths: SwarmPaths, limit: int = 5) -> list[dict[str, object]]:
    if not paths.ledger_path.exists():
        return []
    lines = [line for line in paths.ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    entries = [json.loads(line) for line in lines[-limit:]]
    return entries


def _extract_hook_body(champion_text: str, hook_name: str) -> str:
    marker = f"def {hook_name}("
    start = champion_text.find(marker)
    if start == -1:
        return ""
    next_def = champion_text.find("\ndef ", start + len(marker))
    if next_def == -1:
        next_def = len(champion_text)
    return champion_text[start:next_def]


def _hook_state_rank(champion_text: str, hook_name: str) -> int:
    body = _extract_hook_body(champion_text, hook_name)
    if not body:
        return 99
    terminal_lines = [line.strip() for line in body.splitlines() if line.strip()]
    terminal_return = terminal_lines[-1] if terminal_lines else ""
    if terminal_return == "return 0":
        return 0
    if terminal_return.startswith("return _base_"):
        return 1
    return 2


def _ordered_specializations(paths: SwarmPaths, surface: str) -> list[tuple[str, str, str]]:
    if surface == "search":
        return list(DEFAULT_SEARCH_SPECIALIZATIONS)
    champion_text = paths.champion_eval.read_text(encoding="utf-8") if paths.champion_eval.exists() else ""
    return sorted(
        DEFAULT_WORKER_SPECIALIZATIONS,
        key=lambda spec: (_hook_state_rank(champion_text, spec[2]), DEFAULT_WORKER_SPECIALIZATIONS.index(spec)),
    )


def _build_worker_prompt(
    *,
    worker_name: str,
    worker_role: str,
    worker_lane: str,
    target_hook: str,
    target_file: str,
    recent_entries: list[dict[str, object]],
) -> str:
    history_lines: list[str] = []
    for entry in recent_entries:
        if not entry.get("accepted"):
            continue
        benchmark = entry.get("benchmark") or {}
        history_lines.append(
            f"- {entry['worker_name']}: score={benchmark.get('score')} "
            f"elo={benchmark.get('elo_delta_estimate')} summary={entry.get('summary')}"
        )
    recent_history = "\n".join(history_lines) if history_lines else "- no accepted candidates yet"
    return f"""Improve the current Chess960 champion in `{target_file}`.

Lane:
- {worker_lane}

Target hook:
- edit only `{target_hook}` and keep the rest of the file unchanged
- if you need helper values, define them directly inside `{target_hook}` or make the smallest possible local constant change

Before editing, inspect:
- `{target_file}`
- `outputs/codex_swarm/champion_eval.py`
- `outputs/codex_swarm/champion_search.py`
- `outputs/codex_swarm/ledger.jsonl`
- `outputs/codex_swarm/accepted/`

Requirements:
- edit only `{target_file}`
- make one small surgical patch inside `{target_hook}`
- avoid duplicating prior accepted winners
- do not run held-out benchmarks; the coordinator does that
- finish quickly with a short summary of the patch and why it should help

Recent accepted candidates:
{recent_history}
"""


def _build_worker_agents_override(
    *,
    worker_name: str,
    worker_role: str,
    worker_lane: str,
    target_hook: str,
    target_file: str,
    max_diff_lines: int,
) -> str:
    return f"""# Codex swarm worker override

You are {worker_name}, the {worker_role}, in the 0x960 Codex swarm.

Primary lane:
- {worker_lane}

Hard requirements:
- Edit only `{target_file}`.
- Touch only the `{target_hook}` function body unless a tiny adjacent constant change is absolutely necessary.
- Use `apply_patch` or similarly surgical edits. Do not rewrite the whole file.
- Keep the final diff within about {max_diff_lines} changed lines total.
- If your current diff exceeds that budget, revert the excess and reduce the patch before finishing.
- Run at most one small local probe. Do not run held-out benchmarks yourself; the coordinator handles them.
- Do not browse the web or use internet-dependent tools for this task.
- Stop immediately after the patch and one tiny sanity check. Do not spend time on extra diffs, `rg`, or `sed` inspections once the patch is in place.
- Write a concise summary of the change and why it should help, then exit.
"""


def _surface_config(surface: str) -> tuple[tuple[str, ...], str]:
    if surface == "search":
        return DEFAULT_SEARCH_EDITABLE_FILES, DEFAULT_SEARCH_EDITABLE_FILES[0]
    return DEFAULT_EVAL_EDITABLE_FILES, DEFAULT_EVAL_EDITABLE_FILES[0]


def _screen_settings(args: argparse.Namespace) -> tuple[int, int, int]:
    if args.surface == "search":
        return args.search_screen_positions, args.search_screen_depth, args.search_screen_max_plies
    return args.screen_positions, args.depth, args.max_plies


def _baseline_snapshot_root(paths: SwarmPaths, round_dir: Path) -> Path:
    baseline_root = round_dir / "baseline_root"
    _copy_file(
        paths.champion_eval,
        baseline_root / "src" / "zero960" / "workspace_template" / "eval.py",
    )
    _copy_file(
        paths.champion_search,
        baseline_root / "src" / "zero960" / "engine" / "search.py",
    )
    return baseline_root


def _snapshot_files(worker_dir: Path, rel_paths: tuple[str, ...]) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for rel_path in rel_paths:
        file_path = worker_dir / rel_path
        if file_path.exists():
            snapshot[rel_path] = file_path.read_text(encoding="utf-8")
    return snapshot


def _changed_snapshot_paths(before: dict[str, str], worker_dir: Path, rel_paths: tuple[str, ...]) -> list[str]:
    changed: list[str] = []
    for rel_path in rel_paths:
        file_path = worker_dir / rel_path
        after = file_path.read_text(encoding="utf-8") if file_path.exists() else ""
        if before.get(rel_path, "") != after:
            changed.append(rel_path)
    return changed


def _snapshot_diff_line_counts(
    before: dict[str, str],
    worker_dir: Path,
    rel_paths: tuple[str, ...],
) -> tuple[int, int]:
    added = 0
    deleted = 0
    for rel_path in rel_paths:
        before_text = before.get(rel_path, "")
        file_path = worker_dir / rel_path
        after_text = file_path.read_text(encoding="utf-8") if file_path.exists() else ""
        for line in difflib.unified_diff(
            before_text.splitlines(),
            after_text.splitlines(),
            fromfile=rel_path,
            tofile=rel_path,
            lineterm="",
        ):
            if line.startswith(("---", "+++", "@@")):
                continue
            if line.startswith("+"):
                added += 1
            elif line.startswith("-"):
                deleted += 1
    return added, deleted


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _decode_timeout_output(payload: str | bytes | None) -> str:
    if payload is None:
        return ""
    if isinstance(payload, bytes):
        return payload.decode("utf-8", errors="replace")
    return payload


def _run_worker(
    *,
    paths: SwarmPaths,
    worker_dir: Path,
    round_dir: Path,
    worker_name: str,
    worker_role: str,
    worker_lane: str,
    target_hook: str,
    target_file: str,
    model: str,
    editable_files: tuple[str, ...],
    candidate_file_rel: str,
    positions: int,
    depth: int,
    max_plies: int,
    seed: int,
    min_score: float,
    max_diff_lines: int,
    worker_timeout_sec: int,
    dry_run: bool,
    sandbox_mode: str,
) -> WorkerResult:
    worker_dir = worker_dir.resolve()
    candidate_file = worker_dir / candidate_file_rel
    prompt_path = round_dir / f"{worker_name}_prompt.txt"
    final_message_path = round_dir / f"{worker_name}_final.txt"
    stdout_path = round_dir / f"{worker_name}_stdout.log"
    stderr_path = round_dir / f"{worker_name}_stderr.log"
    recent_entries = _last_ledger_entries(paths)
    prompt = _build_worker_prompt(
        worker_name=worker_name,
        worker_role=worker_role,
        worker_lane=worker_lane,
        target_hook=target_hook,
        target_file=target_file,
        recent_entries=recent_entries,
    )
    prompt_path.write_text(prompt, encoding="utf-8")
    (worker_dir / "AGENTS.override.md").write_text(
        _build_worker_agents_override(
            worker_name=worker_name,
            worker_role=worker_role,
            worker_lane=worker_lane,
            target_hook=target_hook,
            target_file=target_file,
            max_diff_lines=max_diff_lines,
        ),
        encoding="utf-8",
    )
    before_snapshot = _snapshot_files(worker_dir, editable_files)

    if dry_run:
        stdout_path.write_text("dry-run\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        final_message_path.write_text("dry-run\n", encoding="utf-8")
        changed_files = _changed_snapshot_paths(before_snapshot, worker_dir, editable_files)
        return WorkerResult(
            worker_name=worker_name,
            worktree_dir=worker_dir,
            round_dir=round_dir,
            prompt_path=prompt_path,
            final_message_path=final_message_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            candidate_file=candidate_file,
            changed_files=changed_files,
            diff_lines_added=0,
            diff_lines_deleted=0,
            screen_benchmark=None,
            benchmark=None,
            exit_code=0,
            accepted=False,
            summary="dry-run",
            sandbox_mode=sandbox_mode,
        )

    try:
        completed = subprocess.run(
            [
                "codex",
                "exec",
                "-m",
                model,
                "--full-auto",
                "--ephemeral",
                "--json",
                "-c",
                'web_search="disabled"',
                "--color",
                "never",
                "--output-last-message",
                str(final_message_path),
                "-",
            ],
            cwd=worker_dir,
            input=prompt,
            text=True,
            capture_output=True,
            check=False,
            timeout=worker_timeout_sec,
        )
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
    except subprocess.TimeoutExpired as exc:
        stdout_text = _decode_timeout_output(exc.stdout)
        stderr_text = _decode_timeout_output(exc.stderr)
        stdout_path.write_text(stdout_text, encoding="utf-8")
        stderr_path.write_text(stderr_text + f"\nTimed out after {worker_timeout_sec} seconds.\n", encoding="utf-8")
        if not final_message_path.exists():
            final_message_path.write_text("", encoding="utf-8")
        changed_files = _changed_snapshot_paths(before_snapshot, worker_dir, editable_files)
        diff_lines_added, diff_lines_deleted = _snapshot_diff_line_counts(before_snapshot, worker_dir, editable_files)
        return WorkerResult(
            worker_name=worker_name,
            worktree_dir=worker_dir,
            round_dir=round_dir,
            prompt_path=prompt_path,
            final_message_path=final_message_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            candidate_file=candidate_file,
            changed_files=changed_files,
            diff_lines_added=diff_lines_added,
            diff_lines_deleted=diff_lines_deleted,
            screen_benchmark=None,
            benchmark=None,
            exit_code=None,
            accepted=False,
            summary=f"timed out after {worker_timeout_sec}s",
            sandbox_mode=sandbox_mode,
        )
    if not final_message_path.exists():
        final_message_path.write_text("", encoding="utf-8")

    changed_files = _changed_snapshot_paths(before_snapshot, worker_dir, editable_files)
    diff_lines_added, diff_lines_deleted = _snapshot_diff_line_counts(before_snapshot, worker_dir, editable_files)
    summary = final_message_path.read_text(encoding="utf-8").strip()
    if completed.returncode != 0:
        summary = summary or "codex exec failed"

    return WorkerResult(
        worker_name=worker_name,
        worktree_dir=worker_dir,
        round_dir=round_dir,
        prompt_path=prompt_path,
        final_message_path=final_message_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        candidate_file=candidate_file,
        changed_files=changed_files,
        diff_lines_added=diff_lines_added,
        diff_lines_deleted=diff_lines_deleted,
        screen_benchmark=None,
        benchmark=None,
        exit_code=completed.returncode,
        accepted=False,
        summary=summary,
        sandbox_mode=sandbox_mode,
    )


def _eligible_for_screen(result: WorkerResult, max_diff_lines: int) -> bool:
    if result.exit_code not in (0, None):
        return False
    if not result.candidate_file.exists():
        return False
    if not result.changed_files:
        return False
    return (result.diff_lines_added + result.diff_lines_deleted) <= max_diff_lines


def _candidate_compiles(candidate_file: Path) -> bool:
    completed = subprocess.run(
        ["python3", "-m", "py_compile", str(candidate_file)],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0


def _benchmark_eval_task(
    candidate_file: str,
    baseline_file: str,
    positions: int,
    depth: int,
    max_plies: int,
    seed: int,
) -> BenchmarkResult:
    return benchmark_eval_files(
        Path(candidate_file).resolve(),
        Path(baseline_file).resolve(),
        positions=positions,
        depth=depth,
        max_plies=max_plies,
        seed=seed,
    )


def _benchmark_engine_task(
    candidate_root: str,
    baseline_root: str,
    positions: int,
    depth: int,
    max_plies: int,
    seed: int,
) -> BenchmarkResult:
    return benchmark_engine_roots(
        Path(candidate_root).resolve(),
        Path(baseline_root).resolve(),
        positions=positions,
        depth=depth,
        max_plies=max_plies,
        seed=seed,
    )


def _run_benchmark_with_timeout(
    *,
    surface: str,
    candidate_path: Path,
    baseline_path: Path,
    positions: int,
    depth: int,
    max_plies: int,
    seed: int,
    timeout_sec: int,
) -> BenchmarkResult | None:
    task = _benchmark_engine_task if surface == "search" else _benchmark_eval_task
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            task,
            str(candidate_path),
            str(baseline_path),
            positions,
            depth,
            max_plies,
            seed,
        )
        try:
            return future.result(timeout=timeout_sec)
        except TimeoutError:
            future.cancel()
            return None


def _best_screened(results: list[WorkerResult], screen_min_score: float, surface: str) -> WorkerResult | None:
    comparator = (
        (lambda score: score >= screen_min_score)
        if surface == "search"
        else (lambda score: score > screen_min_score)
    )
    screened = [
        result
        for result in results
        if result.screen_benchmark is not None and comparator(result.screen_benchmark.score)
    ]
    if not screened:
        return None
    return max(screened, key=lambda result: result.screen_benchmark.score)


def _promote_winner(paths: SwarmPaths, winner: WorkerResult, promote_source: bool) -> None:
    accepted_dir = paths.state_root / "accepted"
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if "src/zero960/workspace_template/eval.py" in winner.changed_files:
        _copy_file(winner.worktree_dir / "src/zero960/workspace_template/eval.py", paths.champion_eval)
        _copy_file(
            winner.worktree_dir / "src/zero960/workspace_template/eval.py",
            accepted_dir / f"{timestamp}_{winner.worker_name}_eval.py",
        )
    if "src/zero960/engine/search.py" in winner.changed_files:
        _copy_file(winner.worktree_dir / "src/zero960/engine/search.py", paths.champion_search)
        _copy_file(
            winner.worktree_dir / "src/zero960/engine/search.py",
            accepted_dir / f"{timestamp}_{winner.worker_name}_search.py",
        )
    if promote_source and "src/zero960/workspace_template/eval.py" in winner.changed_files:
        _copy_file(winner.candidate_file, paths.repo_root / "src/zero960/workspace_template/eval.py")
        _copy_file(winner.candidate_file, paths.repo_root / "src/zero960/engine/default_eval.py")
    if promote_source and "src/zero960/engine/search.py" in winner.changed_files:
        _copy_file(winner.worktree_dir / "src/zero960/engine/search.py", paths.repo_root / "src/zero960/engine/search.py")


def _state_summary(paths: SwarmPaths) -> str:
    entries = [
        entry
        for entry in _last_ledger_entries(paths, limit=20)
        if (entry.get("benchmark") or {}).get("score") is not None
    ]
    if not paths.champion_eval.exists():
        return "no champion yet"
    if not entries:
        return f"champion={paths.champion_eval}"
    last = entries[-1]
    benchmark = last.get("benchmark") or {}
    return (
        f"champion={paths.champion_eval} "
        f"last_worker={last.get('worker_name')} "
        f"score={benchmark.get('score')} "
        f"elo={benchmark.get('elo_delta_estimate')}"
    )


def parse_args() -> argparse.Namespace:
    paths = _default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup = subparsers.add_parser("setup", help="Create local worker worktrees and initialize the champion.")
    setup.add_argument("--workers", type=int, default=DEFAULT_WORKER_COUNT)
    setup.add_argument("--worktree-root", default=str(paths.worktree_root))
    setup.add_argument("--reset-champion", action="store_true")

    run = subparsers.add_parser("run", help="Run one or more champion/challenger rounds.")
    run.add_argument("--workers", type=int, default=DEFAULT_WORKER_COUNT)
    run.add_argument("--rounds", type=int, default=1)
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--surface", choices=("eval", "search"), default=DEFAULT_SURFACE)
    run.add_argument("--worktree-root", default=str(paths.worktree_root))
    run.add_argument("--screen-positions", type=int, default=DEFAULT_SCREEN_POSITIONS)
    run.add_argument("--positions", type=int, default=DEFAULT_POSITIONS)
    run.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    run.add_argument("--max-plies", type=int, default=DEFAULT_MAX_PLIES)
    run.add_argument("--search-screen-positions", type=int, default=DEFAULT_SEARCH_SCREEN_POSITIONS)
    run.add_argument("--search-screen-depth", type=int, default=DEFAULT_SEARCH_SCREEN_DEPTH)
    run.add_argument("--search-screen-max-plies", type=int, default=DEFAULT_SEARCH_SCREEN_MAX_PLIES)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--screen-min-score", type=float, default=DEFAULT_SCREEN_MIN_SCORE)
    run.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    run.add_argument("--max-diff-lines", type=int, default=DEFAULT_MAX_DIFF_LINES)
    run.add_argument("--worker-timeout-sec", type=int, default=DEFAULT_WORKER_TIMEOUT_SEC)
    run.add_argument("--benchmark-timeout-sec", type=int, default=DEFAULT_BENCHMARK_TIMEOUT_SEC)
    run.add_argument("--final-benchmark-timeout-sec", type=int, default=DEFAULT_FINAL_BENCHMARK_TIMEOUT_SEC)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--serial", action="store_true", help="Run workers sequentially instead of in parallel.")
    run.add_argument("--promote-source", action="store_true")
    run.add_argument("--continuous", action="store_true", help="Keep running rounds until interrupted or stalled.")
    run.add_argument(
        "--max-stall-rounds",
        type=int,
        default=3,
        help="Stop continuous mode after this many consecutive non-promotion rounds. Use 0 to disable.",
    )
    run.add_argument("--sleep-sec", type=float, default=0.0, help="Sleep between continuous rounds.")

    status = subparsers.add_parser("status", help="Print the current champion summary and recent results.")

    promote = subparsers.add_parser("promote", help="Copy the current swarm champion into the source tree.")
    promote.add_argument("--source-only", action="store_true", help="Skip copying to default_eval.py.")

    return parser.parse_args()


def _resolve_paths(args: argparse.Namespace) -> SwarmPaths:
    paths = _default_paths()
    if hasattr(args, "worktree_root"):
        paths.worktree_root = Path(args.worktree_root).resolve()
    return paths


def _setup_command(args: argparse.Namespace) -> int:
    paths = _resolve_paths(args)
    _ensure_state_dirs(paths)
    if args.reset_champion or not paths.champion_eval.exists():
        _copy_file(paths.repo_root / "src/zero960/workspace_template/eval.py", paths.champion_eval)
    if args.reset_champion or not paths.champion_search.exists():
        _copy_file(paths.repo_root / "src/zero960/engine/search.py", paths.champion_search)
    worker_dirs = _setup_workers(paths, args.workers, DEFAULT_SYNC_PATHS)
    print(f"initialized champion: {paths.champion_eval}")
    for worker_dir, sandbox_mode in worker_dirs:
        print(f"worker: {worker_dir} mode={sandbox_mode}")
    return 0


def _run_command(args: argparse.Namespace) -> int:
    paths = _resolve_paths(args)
    _ensure_state_dirs(paths)
    if not paths.champion_eval.exists():
        _copy_file(paths.repo_root / "src/zero960/workspace_template/eval.py", paths.champion_eval)
    if not paths.champion_search.exists():
        _copy_file(paths.repo_root / "src/zero960/engine/search.py", paths.champion_search)
    worker_dirs = _setup_workers(paths, args.workers, DEFAULT_SYNC_PATHS)
    editable_files, candidate_file_rel = _surface_config(args.surface)

    round_index = 0
    stall_rounds = 0
    target_rounds = None if args.continuous else args.rounds

    while target_rounds is None or round_index < target_rounds:
        round_index += 1
        round_seed = args.seed + round_index - 1
        round_timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        round_dir = paths.state_root / "runs" / f"round_{round_timestamp}_{round_index}"
        round_dir.mkdir(parents=True, exist_ok=True)
        baseline_root = _baseline_snapshot_root(paths, round_dir)
        round_specializations = _ordered_specializations(paths, args.surface)
        screen_positions, screen_depth, screen_max_plies = _screen_settings(args)
        print(f"round {round_index}: champion frozen at {paths.champion_eval}")
        print(f"round {round_index}: surface={args.surface}")
        print(
            "round hooks: "
            + ", ".join(spec[2] for spec in round_specializations[: len(worker_dirs)])
        )

        jobs = []
        if args.serial or args.dry_run:
            for worker_index, (worker_dir, sandbox_mode) in enumerate(worker_dirs, start=1):
                _sync_worker_snapshot(paths, worker_dir, DEFAULT_SYNC_PATHS)
                worker_role, worker_lane, target_hook = round_specializations[(worker_index - 1) % len(round_specializations)]
                result = _run_worker(
                    paths=paths,
                    worker_dir=worker_dir,
                    round_dir=round_dir,
                    worker_name=f"worker-{worker_index}",
                    worker_role=worker_role,
                    worker_lane=worker_lane,
                    target_hook=target_hook,
                    target_file=candidate_file_rel,
                    model=args.model,
                    editable_files=editable_files,
                    candidate_file_rel=candidate_file_rel,
                    positions=args.positions,
                    depth=args.depth,
                    max_plies=args.max_plies,
                    seed=round_seed,
                    min_score=args.min_score,
                    max_diff_lines=args.max_diff_lines,
                    worker_timeout_sec=args.worker_timeout_sec,
                    dry_run=args.dry_run,
                    sandbox_mode=sandbox_mode,
                )
                jobs.append(result)
        else:
            with ThreadPoolExecutor(max_workers=len(worker_dirs)) as executor:
                futures = []
                for worker_index, (worker_dir, sandbox_mode) in enumerate(worker_dirs, start=1):
                    _sync_worker_snapshot(paths, worker_dir, DEFAULT_SYNC_PATHS)
                    worker_role, worker_lane, target_hook = round_specializations[(worker_index - 1) % len(round_specializations)]
                    futures.append(
                        executor.submit(
                            _run_worker,
                            paths=paths,
                            worker_dir=worker_dir,
                            round_dir=round_dir,
                            worker_name=f"worker-{worker_index}",
                            worker_role=worker_role,
                            worker_lane=worker_lane,
                            target_hook=target_hook,
                            target_file=candidate_file_rel,
                            model=args.model,
                            editable_files=editable_files,
                            candidate_file_rel=candidate_file_rel,
                            positions=args.positions,
                            depth=args.depth,
                            max_plies=args.max_plies,
                            seed=round_seed,
                            min_score=args.min_score,
                            max_diff_lines=args.max_diff_lines,
                            worker_timeout_sec=args.worker_timeout_sec,
                            dry_run=args.dry_run,
                            sandbox_mode=sandbox_mode,
                        )
                    )
                for future in as_completed(futures):
                    jobs.append(future.result())

        jobs.sort(key=lambda result: result.worker_name)
        for result in jobs:
            diff_total = result.diff_lines_added + result.diff_lines_deleted
            if result.exit_code not in (0, None):
                continue
            if not result.candidate_file.exists():
                continue
            if not result.changed_files:
                rejection = "rejected before benchmark: no file changes"
                result.summary = f"{result.summary}\n{rejection}".strip() if result.summary else rejection
                continue
            if diff_total > args.max_diff_lines:
                overflow = diff_total - args.max_diff_lines
                rejection = f"rejected before benchmark: diff budget exceeded by {overflow} lines"
                result.summary = f"{result.summary}\n{rejection}".strip() if result.summary else rejection
                continue
            if not _candidate_compiles(result.candidate_file):
                rejection = "rejected before benchmark: candidate failed py_compile"
                result.summary = f"{result.summary}\n{rejection}".strip() if result.summary else rejection
                continue
            screen_candidate = result.worktree_dir.resolve() if args.surface == "search" else result.candidate_file.resolve()
            screen_baseline = (
                baseline_root.resolve()
                if args.surface == "search"
                else (baseline_root / "src" / "zero960" / "workspace_template" / "eval.py").resolve()
            )
            result.screen_benchmark = _run_benchmark_with_timeout(
                surface=args.surface,
                candidate_path=screen_candidate,
                baseline_path=screen_baseline,
                positions=screen_positions,
                depth=screen_depth,
                max_plies=screen_max_plies,
                seed=round_seed,
                timeout_sec=args.benchmark_timeout_sec,
            )
            if result.screen_benchmark is None:
                rejection = f"rejected during screen benchmark: timed out after {args.benchmark_timeout_sec}s"
                result.summary = f"{result.summary}\n{rejection}".strip() if result.summary else rejection

        winner = _best_screened(jobs, args.screen_min_score, args.surface)
        if winner is not None:
            final_candidate = winner.worktree_dir.resolve() if args.surface == "search" else winner.candidate_file.resolve()
            final_baseline = (
                baseline_root.resolve()
                if args.surface == "search"
                else (baseline_root / "src" / "zero960" / "workspace_template" / "eval.py").resolve()
            )
            winner.benchmark = _run_benchmark_with_timeout(
                surface=args.surface,
                candidate_path=final_candidate,
                baseline_path=final_baseline,
                positions=args.positions,
                depth=args.depth,
                max_plies=args.max_plies,
                seed=round_seed,
                timeout_sec=args.final_benchmark_timeout_sec,
            )
            winner.accepted = winner.benchmark is not None and winner.benchmark.score > args.min_score
            if winner.benchmark is None:
                rejection = f"screen winner timed out in final benchmark after {args.final_benchmark_timeout_sec}s"
                winner.summary = f"{winner.summary}\n{rejection}".strip() if winner.summary else rejection
                winner = None
            elif not winner.accepted:
                rejection = (
                    f"screen winner failed final benchmark: "
                    f"{winner.benchmark.score:.3f} <= {args.min_score:.3f}"
                )
                winner.summary = f"{winner.summary}\n{rejection}".strip() if winner.summary else rejection
                winner = None

        for result in jobs:
            payload = result.to_json()
            payload["round_index"] = round_index
            payload["winner"] = bool(winner and winner.worker_name == result.worker_name)
            payload["surface"] = args.surface
            _write_json(round_dir / f"{result.worker_name}_result.json", payload)
            if not args.dry_run:
                _append_jsonl(paths.ledger_path, payload)
            screen_text = "n/a" if result.screen_benchmark is None else f"{result.screen_benchmark.score:.3f}"
            final_text = "n/a" if result.benchmark is None else f"{result.benchmark.score:.3f}"
            print(
                f"{result.worker_name}: exit={result.exit_code} "
                f"screen={screen_text} final={final_text} accepted={result.accepted} changed={len(result.changed_files)} "
                f"diff=+{result.diff_lines_added}/-{result.diff_lines_deleted} mode={result.sandbox_mode}"
            )

        if winner is None:
            print(f"round {round_index}: no challenger beat the champion")
            stall_rounds += 1
            if args.continuous and args.max_stall_rounds and stall_rounds >= args.max_stall_rounds:
                print(f"stopping after {stall_rounds} consecutive non-promotion rounds")
                break
            if args.continuous and args.sleep_sec > 0:
                time.sleep(args.sleep_sec)
            continue

        _promote_winner(paths, winner, args.promote_source)
        stall_rounds = 0
        print(
            f"round {round_index}: promoted {winner.worker_name} "
            f"score={winner.benchmark.score:.3f} elo={winner.benchmark.elo_delta_estimate:.1f}"
        )
        if args.continuous and args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    print(_state_summary(paths))
    return 0


def _status_command() -> int:
    paths = _default_paths()
    print(_state_summary(paths))
    for entry in _last_ledger_entries(paths):
        benchmark = entry.get("benchmark") or {}
        if benchmark.get("score") is None:
            continue
        print(
            f"{entry.get('worker_name')}: accepted={entry.get('accepted')} "
            f"score={benchmark.get('score')} elo={benchmark.get('elo_delta_estimate')}"
        )
    return 0


def _promote_command(args: argparse.Namespace) -> int:
    paths = _default_paths()
    if not paths.champion_eval.exists():
        raise SystemExit("no champion available; run setup or run first")
    _copy_file(paths.champion_eval, paths.repo_root / "src/zero960/workspace_template/eval.py")
    if paths.champion_search.exists():
        _copy_file(paths.champion_search, paths.repo_root / "src/zero960/engine/search.py")
    if not args.source_only:
        _copy_file(paths.champion_eval, paths.repo_root / "src/zero960/engine/default_eval.py")
    print(f"promoted champion from {paths.champion_eval}")
    return 0


def main() -> None:
    args = parse_args()
    if args.command == "setup":
        raise SystemExit(_setup_command(args))
    if args.command == "run":
        raise SystemExit(_run_command(args))
    if args.command == "status":
        raise SystemExit(_status_command())
    if args.command == "promote":
        raise SystemExit(_promote_command(args))
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
