"""Build a self-contained HTML dashboard for swarm and benchmark results."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from train.benchmark_engine import benchmark_engine_roots
from train.benchmark_league import benchmark_league, default_league_opponents
from train.benchmark_uci import benchmark_eval_vs_uci


@dataclass(slots=True)
class DashboardData:
    generated_at: str
    current_champion: str
    accepted_count: int
    all_results: list[dict[str, object]]
    accepted_results: list[dict[str, object]]
    engine_progress: dict[str, object] | None
    league: dict[str, object] | None
    stockfish_anchors: list[dict[str, object]]

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _short_summary(summary: str, limit: int = 180) -> str:
    compact = " ".join(summary.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _normalize_result(entry: dict[str, object]) -> dict[str, object]:
    benchmark = entry.get("benchmark") or {}
    round_dir = str(entry.get("round_dir", ""))
    round_name = Path(round_dir).name if round_dir else "unknown"
    return {
        "worker_name": entry.get("worker_name"),
        "accepted": bool(entry.get("accepted")),
        "winner": bool(entry.get("winner")),
        "round_name": round_name,
        "score": benchmark.get("score"),
        "elo_delta_estimate": benchmark.get("elo_delta_estimate"),
        "wins": benchmark.get("wins"),
        "draws": benchmark.get("draws"),
        "losses": benchmark.get("losses"),
        "points": benchmark.get("points"),
        "total_games": benchmark.get("total_games"),
        "candidate_file": entry.get("candidate_file"),
        "summary": _short_summary(str(entry.get("summary", ""))),
        "surface": entry.get("surface", "eval"),
    }


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def _build_engine_progress(
    root: Path,
    champion_eval_path: Path,
    *,
    baseline_root: Path,
    positions: int,
    depth: int,
    max_plies: int,
    seed: int,
) -> dict[str, object] | None:
    if not baseline_root.exists():
        return None
    baseline_eval = baseline_root / "src" / "zero960" / "workspace_template" / "eval.py"
    baseline_search = baseline_root / "src" / "zero960" / "engine" / "search.py"
    if not baseline_eval.exists() or not baseline_search.exists():
        return None

    with tempfile.TemporaryDirectory(prefix="0x960-dashboard-engine-") as temp_dir:
        candidate_root = Path(temp_dir)
        _copy_file(
            champion_eval_path,
            candidate_root / "src" / "zero960" / "workspace_template" / "eval.py",
        )
        _copy_file(
            root / "src" / "zero960" / "engine" / "search.py",
            candidate_root / "src" / "zero960" / "engine" / "search.py",
        )
        result = benchmark_engine_roots(
            candidate_root,
            baseline_root,
            positions=positions,
            depth=depth,
            max_plies=max_plies,
            seed=seed,
        )
    return {
        "label": "Current engine vs search baseline",
        "candidate_eval_path": str(champion_eval_path),
        "candidate_search_path": str((root / "src" / "zero960" / "engine" / "search.py").resolve()),
        "baseline_root": str(baseline_root),
        "result": result.to_json(),
    }


def _build_stockfish_anchors(
    candidate_path: Path,
    *,
    positions: int,
    candidate_depth: int,
    engine_depth: int,
    max_plies: int,
    seed: int,
    engine_command: str,
    anchor_elos: list[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for elo in anchor_elos:
        result = benchmark_eval_vs_uci(
            candidate_path,
            engine_command,
            engine_options={"UCI_LimitStrength": True, "UCI_Elo": elo},
            positions=positions,
            candidate_depth=candidate_depth,
            engine_depth=engine_depth,
            max_plies=max_plies,
            seed=seed,
        )
        rows.append(
            {
                "label": f"Stockfish {elo}",
                "uci_elo": elo,
                "score": result.score,
                "elo_delta_estimate": result.elo_delta_estimate,
                "wins": result.wins,
                "draws": result.draws,
                "losses": result.losses,
                "points": result.points,
                "total_games": result.total_games,
            }
        )
    return rows


def _build_dashboard_data(args: argparse.Namespace) -> DashboardData:
    root = _repo_root()
    ledger_path = root / "outputs" / "codex_swarm" / "ledger.jsonl"
    champion_path = Path(args.candidate_file).resolve()
    ledger_rows = _load_jsonl(ledger_path)
    normalized_rows = [_normalize_result(row) for row in ledger_rows if row.get("benchmark") is not None]
    accepted_rows = [row for row in normalized_rows if row["accepted"]]

    league_payload: dict[str, object] | None = None
    opponents = default_league_opponents(
        candidate_path=champion_path,
        include_baseline=True,
        include_champion=True,
        accepted_limit=args.league_accepted_limit,
    )
    if opponents:
        league_result = benchmark_league(
            champion_path,
            opponents,
            positions=args.league_positions,
            depth=args.depth,
            max_plies=args.max_plies,
            seed=args.seed,
        )
        league_payload = league_result.to_json()

    stockfish_rows: list[dict[str, object]] = []
    engine_progress: dict[str, object] | None = None
    if args.include_engine_progress:
        engine_progress = _build_engine_progress(
            root,
            champion_path,
            baseline_root=Path(args.engine_baseline_root).resolve(),
            positions=args.engine_positions,
            depth=args.depth,
            max_plies=args.max_plies,
            seed=args.seed,
        )
    if args.include_stockfish:
        stockfish_rows = _build_stockfish_anchors(
            champion_path,
            positions=args.stockfish_positions,
            candidate_depth=args.depth,
            engine_depth=args.stockfish_depth,
            max_plies=args.max_plies,
            seed=args.seed,
            engine_command=args.engine_command,
            anchor_elos=args.stockfish_elo,
        )

    return DashboardData(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        current_champion=str(champion_path),
        accepted_count=len(accepted_rows),
        all_results=normalized_rows,
        accepted_results=accepted_rows,
        engine_progress=engine_progress,
        league=league_payload,
        stockfish_anchors=stockfish_rows,
    )


def _dashboard_html(payload: dict[str, object]) -> str:
    data_json = json.dumps(payload)
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>0x960 Dashboard</title>
  <style>
    :root {{
      --bg: #0d1117;
      --panel: #151b23;
      --panel-2: #1d2733;
      --text: #e6edf3;
      --muted: #9fb0c0;
      --green: #3fb950;
      --red: #f85149;
      --amber: #d29922;
      --blue: #58a6ff;
      --border: #2d3a49;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(88,166,255,0.14), transparent 28%),
        radial-gradient(circle at top right, rgba(63,185,80,0.12), transparent 22%),
        linear-gradient(180deg, #0b1016 0%, var(--bg) 100%);
      color: var(--text);
    }}
    .wrap {{
      width: min(1200px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 48px;
    }}
    h1, h2, h3, p {{ margin: 0; }}
    .hero {{
      display: grid;
      gap: 12px;
      margin-bottom: 20px;
    }}
    .hero p {{ color: var(--muted); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 16px;
    }}
    .card {{
      background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 18px;
      backdrop-filter: blur(8px);
      box-shadow: 0 18px 50px rgba(0,0,0,0.22);
    }}
    .span-3 {{ grid-column: span 3; }}
    .span-4 {{ grid-column: span 4; }}
    .span-5 {{ grid-column: span 5; }}
    .span-6 {{ grid-column: span 6; }}
    .span-7 {{ grid-column: span 7; }}
    .span-8 {{ grid-column: span 8; }}
    .span-12 {{ grid-column: span 12; }}
    .kpi-label {{ color: var(--muted); font-size: 13px; margin-bottom: 8px; }}
    .kpi-value {{ font-size: 34px; font-weight: 700; letter-spacing: -0.03em; }}
    .kpi-sub {{ color: var(--muted); margin-top: 6px; font-size: 13px; }}
    .section-title {{ font-size: 18px; margin-bottom: 14px; }}
    .chart {{ width: 100%; height: 280px; }}
    .bars .row, .table-row {{
      display: grid;
      gap: 12px;
      align-items: center;
    }}
    .bars .row {{
      grid-template-columns: 190px 1fr 72px;
      margin-bottom: 10px;
    }}
    .bar-track {{
      height: 12px;
      border-radius: 999px;
      background: rgba(255,255,255,0.08);
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, var(--blue), #8ed0ff);
    }}
    .good {{ color: var(--green); }}
    .bad {{ color: var(--red); }}
    .muted {{ color: var(--muted); }}
    .table-head, .table-row {{
      grid-template-columns: 126px 70px 82px 120px 1fr;
      font-size: 13px;
      padding: 10px 0;
      border-bottom: 1px solid rgba(255,255,255,0.06);
    }}
    .table-head {{
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      font-size: 11px;
    }}
    .pill {{
      display: inline-block;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      background: rgba(255,255,255,0.08);
    }}
    .pill.win {{ background: rgba(63,185,80,0.16); color: var(--green); }}
    .pill.loss {{ background: rgba(248,81,73,0.14); color: var(--red); }}
    .pill.flat {{ background: rgba(210,153,34,0.14); color: var(--amber); }}
    .league-list {{
      display: grid;
      gap: 12px;
    }}
    .league-item {{
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 12px;
      padding: 12px 0;
      border-bottom: 1px solid rgba(255,255,255,0.06);
    }}
    .footer {{
      margin-top: 16px;
      font-size: 12px;
      color: var(--muted);
    }}
    @media (max-width: 900px) {{
      .span-3, .span-4, .span-5, .span-6, .span-7, .span-8, .span-12 {{
        grid-column: span 12;
      }}
      .bars .row {{ grid-template-columns: 1fr; }}
      .table-head, .table-row {{ grid-template-columns: 1fr; gap: 6px; }}
      .league-item {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <h1>0x960 Engine Dashboard</h1>
      <p>Swarm progress, internal Elo deltas, league self-play, and optional Stockfish anchors in one static page.</p>
    </div>
    <div class="grid" id="app"></div>
    <div class="footer" id="footer"></div>
  </div>
  <script type="application/json" id="dashboard-data">__DASHBOARD_JSON__</script>
  <script>
    const data = JSON.parse(document.getElementById('dashboard-data').textContent);
    const app = document.getElementById('app');
    const footer = document.getElementById('footer');

    const accepted = data.accepted_results || [];
    const league = data.league;
    const anchors = data.stockfish_anchors || [];
    const engineProgress = data.engine_progress;
    const all = data.all_results || [];

    const latestAccepted = accepted.length ? accepted[accepted.length - 1] : null;
    const bestAccepted = accepted.length
      ? accepted.reduce((best, row) => (row.score > best.score ? row : best), accepted[0])
      : null;
    const bestRejected = all.filter((row) => !row.accepted && row.score !== null).reduce((best, row) => {
      if (!best || row.score > best.score) return row;
      return best;
    }, null);

    function card(cls, inner) {{
      const el = document.createElement('section');
      el.className = `card ${cls}`;
      el.innerHTML = inner;
      return el;
    }}

    function scoreClass(value) {{
      if (value > 0.5) return 'good';
      if (value < 0.5) return 'bad';
      return 'muted';
    }}

    function eloClass(value) {{
      if (value > 0) return 'good';
      if (value < 0) return 'bad';
      return 'muted';
    }}

    const kpis = [
      {{
        label: 'Accepted Champions',
        value: String(data.accepted_count),
        sub: latestAccepted ? `Latest: ${latestAccepted.worker_name}` : 'No accepted challenger yet'
      }},
      {{
        label: 'Current Internal Score',
        value: latestAccepted ? latestAccepted.score.toFixed(3) : 'n/a',
        sub: latestAccepted ? `vs previous champion` : 'Awaiting accepted run'
      }},
      {{
        label: 'Current Internal Elo',
        value: latestAccepted ? `${latestAccepted.elo_delta_estimate.toFixed(1)}` : 'n/a',
        sub: latestAccepted ? 'delta vs prior champion' : 'Awaiting accepted run'
      }},
      {{
        label: 'League Score',
        value: league ? league.overall_score.toFixed(3) : 'n/a',
        sub: league ? `${league.total_points.toFixed(1)}/${league.total_games} points` : 'League not available'
      }}
    ];

    if (engineProgress) {{
      kpis.push({{
        label: 'Search Gain',
        value: `${{engineProgress.result.elo_delta_estimate.toFixed(1)}}`,
        sub: `${{engineProgress.result.points.toFixed(1)}}/${{engineProgress.result.total_games}} vs baseline`
      }});
    }}

    for (const kpi of kpis) {{
      app.appendChild(card('span-3', `
        <div class="kpi-label">${{kpi.label}}</div>
        <div class="kpi-value">${{kpi.value}}</div>
        <div class="kpi-sub">${{kpi.sub}}</div>
      `));
    }}

    function lineChart(rows) {{
      if (!rows.length) {{
        return '<p class="muted">No accepted results yet.</p>';
      }}
      const width = 640;
      const height = 260;
      const padding = 28;
      const xs = rows.map((_, index) => padding + (index * (width - padding * 2) / Math.max(rows.length - 1, 1)));
      const ys = rows.map((row) => {{
        const score = row.score ?? 0.5;
        return height - padding - ((score - 0.35) / 0.35) * (height - padding * 2);
      }});
      const points = xs.map((x, index) => `${{x}},${{ys[index]}}`).join(' ');
      const circles = xs.map((x, index) =>
        `<circle cx="${{x}}" cy="${{ys[index]}}" r="5" fill="#58a6ff"><title>${{rows[index].worker_name}}: ${{rows[index].score.toFixed(3)}}</title></circle>`
      ).join('');
      return `
        <svg viewBox="0 0 ${{width}} ${{height}}" class="chart" role="img" aria-label="Accepted score progression">
          <line x1="${{padding}}" y1="${{height - padding}}" x2="${{width - padding}}" y2="${{height - padding}}" stroke="rgba(255,255,255,0.18)" />
          <line x1="${{padding}}" y1="${{padding}}" x2="${{padding}}" y2="${{height - padding}}" stroke="rgba(255,255,255,0.18)" />
          <line x1="${{padding}}" y1="${{height - padding - ((0.5 - 0.35) / 0.35) * (height - padding * 2)}}" x2="${{width - padding}}" y2="${{height - padding - ((0.5 - 0.35) / 0.35) * (height - padding * 2)}}" stroke="rgba(210,153,34,0.35)" stroke-dasharray="4 4" />
          <polyline fill="none" stroke="#58a6ff" stroke-width="3" points="${{points}}" />
          ${{circles}}
        </svg>
      `;
    }}

    app.appendChild(card('span-7', `
      <h2 class="section-title">Accepted Score Progression</h2>
      ${{lineChart(accepted)}}
    `));

    const summaryRows = [
      latestAccepted ? `<div class="league-item"><div><strong>Latest winner</strong><div class="muted">${{latestAccepted.worker_name}} in ${{latestAccepted.round_name}}</div></div><div class="${{eloClass(latestAccepted.elo_delta_estimate)}}">${{latestAccepted.elo_delta_estimate.toFixed(1)}} Elo</div><div class="${{scoreClass(latestAccepted.score)}}">${{latestAccepted.score.toFixed(3)}} score</div></div>` : '',
      bestAccepted ? `<div class="league-item"><div><strong>Best accepted score</strong><div class="muted">${{bestAccepted.worker_name}}</div></div><div class="${{eloClass(bestAccepted.elo_delta_estimate)}}">${{bestAccepted.elo_delta_estimate.toFixed(1)}} Elo</div><div class="${{scoreClass(bestAccepted.score)}}">${{bestAccepted.score.toFixed(3)}} score</div></div>` : '',
      bestRejected ? `<div class="league-item"><div><strong>Best rejected try</strong><div class="muted">${{bestRejected.worker_name}} in ${{bestRejected.round_name}}</div></div><div class="${{eloClass(bestRejected.elo_delta_estimate)}}">${{bestRejected.elo_delta_estimate.toFixed(1)}} Elo</div><div class="${{scoreClass(bestRejected.score)}}">${{bestRejected.score.toFixed(3)}} score</div></div>` : ''
    ].join('');

    app.appendChild(card('span-5', `
      <h2 class="section-title">Swarm Snapshot</h2>
      <div class="league-list">${{summaryRows || '<p class="muted">No benchmark rows yet.</p>'}}</div>
    `));

    if (engineProgress) {{
      app.appendChild(card('span-12', `
        <h2 class="section-title">Engine Search Progress</h2>
        <div class="league-list">
          <div class="league-item">
            <div>
              <strong>${{engineProgress.label}}</strong>
              <div class="muted">${{engineProgress.result.wins}}-${{engineProgress.result.draws}}-${{engineProgress.result.losses}}</div>
            </div>
            <div class="${{scoreClass(engineProgress.result.score)}}">${{engineProgress.result.score.toFixed(3)}} score</div>
            <div class="${{eloClass(engineProgress.result.elo_delta_estimate)}}">${{engineProgress.result.elo_delta_estimate.toFixed(1)}} Elo</div>
          </div>
        </div>
        <div class="kpi-sub" style="margin-top: 10px;">
          Candidate search: ${{engineProgress.candidate_search_path}}<br>
          Baseline root: ${{engineProgress.baseline_root}}
        </div>
      `));
    }}

    function barRows(rows, key, formatter) {{
      if (!rows.length) {{
        return '<p class="muted">No data yet.</p>';
      }}
      const values = rows.map((row) => Math.abs(row[key] ?? 0));
      const max = Math.max(...values, 1);
      return rows.map((row) => {{
        const value = row[key] ?? 0;
        const width = Math.max(6, Math.round(Math.abs(value) / max * 100));
        const cls = value > 0 ? 'good' : value < 0 ? 'bad' : 'muted';
        const fill = value > 0 ? 'var(--green)' : value < 0 ? 'var(--red)' : 'var(--amber)';
        return `
          <div class="row">
            <div>${{row.worker_name}}</div>
            <div class="bar-track"><div class="bar-fill" style="width:${{width}}%; background:${{fill}}"></div></div>
            <div class="${{cls}}">${{formatter(value)}}</div>
          </div>
        `;
      }}).join('');
    }}

    app.appendChild(card('span-6', `
      <h2 class="section-title">Accepted Internal Elo Deltas</h2>
      <div class="bars">${{barRows(accepted, 'elo_delta_estimate', (value) => value.toFixed(1))}}</div>
    `));

    app.appendChild(card('span-6', `
      <h2 class="section-title">League Self-Play</h2>
      ${{
        league
          ? `<div class="league-list">${{league.opponents.map((opponent) => `
              <div class="league-item">
                <div>
                  <strong>${{opponent.label}}</strong>
                  <div class="muted">${{opponent.result.wins}}-${{opponent.result.draws}}-${{opponent.result.losses}}</div>
                </div>
                <div class="${{scoreClass(opponent.result.score)}}">${{opponent.result.score.toFixed(3)}} score</div>
                <div class="${{eloClass(opponent.result.elo_delta_estimate)}}">${{opponent.result.elo_delta_estimate.toFixed(1)}} Elo</div>
              </div>
            `).join('')}}</div>
            <div class="kpi-sub" style="margin-top: 10px;">Overall: ${{league.overall_score.toFixed(3)}} score, ${{league.overall_elo_delta_estimate.toFixed(1)}} Elo delta estimate</div>`
          : '<p class="muted">League benchmark not available.</p>'
      }}
    `));

    if (anchors.length) {{
      app.appendChild(card('span-12', `
        <h2 class="section-title">Stockfish Anchor Ladder</h2>
        <div class="bars">${{anchors.map((row) => `
          <div class="row">
            <div>${{row.label}}</div>
            <div class="bar-track"><div class="bar-fill" style="width:${{Math.max(6, Math.round(row.score * 100))}}%; background:${{row.score >= 0.5 ? 'var(--green)' : 'var(--blue)'}}"></div></div>
            <div class="${{scoreClass(row.score)}}">${{row.score.toFixed(3)}}</div>
          </div>
        `).join('')}}</div>
      `));
    }}

    const rows = all.slice().reverse().map((row) => {{
      const pillClass = row.accepted ? 'win' : (row.score > 0.5 ? 'flat' : 'loss');
      const pillText = row.accepted ? 'accepted' : 'rejected';
      return `
        <div class="table-row">
          <div>${{row.round_name}}</div>
          <div>${{row.worker_name}}</div>
          <div><span class="pill ${{pillClass}}">${{pillText}}</span></div>
          <div class="${{scoreClass(row.score)}}">${{row.score !== null ? row.score.toFixed(3) : 'n/a'}}</div>
          <div class="muted">${{row.summary}}</div>
        </div>
      `;
    }}).join('');

    app.appendChild(card('span-12', `
      <h2 class="section-title">Recent Swarm Results</h2>
      <div class="table-head">
        <div>Round</div>
        <div>Worker</div>
        <div>Status</div>
        <div>Score</div>
        <div>Summary</div>
      </div>
      ${{rows || '<p class="muted">No swarm results yet.</p>'}}
    `));

    footer.textContent = `Generated ${{data.generated_at}} | champion: ${{data.current_champion}}`;
  </script>
</body>
</html>
"""
    template = template.replace("{{", "{").replace("}}", "}")
    return template.replace("__DASHBOARD_JSON__", data_json)


def parse_args() -> argparse.Namespace:
    root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-file",
        default=str(root / "outputs" / "codex_swarm" / "champion_eval.py"),
        help="Candidate file to treat as the current engine in the dashboard.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(root / "outputs" / "dashboard"),
        help="Directory where index.html and dashboard_data.json will be written.",
    )
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--max-plies", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--league-positions", type=int, default=8)
    parser.add_argument("--league-accepted-limit", type=int, default=4)
    parser.add_argument("--include-engine-progress", action="store_true")
    parser.add_argument("--engine-baseline-root", default="/tmp/0x960-search-baseline")
    parser.add_argument("--engine-positions", type=int, default=8)
    parser.add_argument("--include-stockfish", action="store_true")
    parser.add_argument("--engine-command", default="stockfish")
    parser.add_argument("--stockfish-depth", type=int, default=1)
    parser.add_argument("--stockfish-positions", type=int, default=4)
    parser.add_argument("--stockfish-elo", type=int, action="append", default=[1320, 1600])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = _build_dashboard_data(args).to_json()
    (output_dir / "dashboard_data.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "index.html").write_text(_dashboard_html(payload), encoding="utf-8")

    print(f"wrote {(output_dir / 'index.html')}")
    print(f"wrote {(output_dir / 'dashboard_data.json')}")


if __name__ == "__main__":
    main()
