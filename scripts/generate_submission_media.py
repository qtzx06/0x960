#!/usr/bin/env python3
"""Generate tracked PNG graphs from benchmark artifacts for submission media."""

from __future__ import annotations

import json
import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Color:
    r: int
    g: int
    b: int

    def to_tuple(self) -> tuple[int, int, int]:
        return (self.r, self.g, self.b)


WHITE = Color(245, 247, 250)
BG = Color(13, 17, 23)
AXIS = Color(132, 146, 165)
GRID = Color(44, 58, 73)
LINE = Color(88, 166, 255)
GOOD = Color(63, 185, 80)
BAD = Color(248, 81, 73)
MID = Color(210, 153, 34)
TEXT = Color(230, 237, 243)


class Canvas:
    def __init__(self, width: int, height: int, bg: Color = BG) -> None:
        self.width = width
        self.height = height
        self.pixels = [[bg.to_tuple() for _ in range(width)] for _ in range(height)]

    def set_pixel(self, x: int, y: int, color: Color) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.pixels[y][x] = color.to_tuple()

    def line(self, x0: int, y0: int, x1: int, y1: int, color: Color) -> None:
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            self.set_pixel(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def rect(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        color: Color,
        fill: bool = True,
    ) -> None:
        if fill:
            for yy in range(max(0, y0), min(self.height, y1 + 1)):
                for xx in range(max(0, x0), min(self.width, x1 + 1)):
                    self.pixels[yy][xx] = color.to_tuple()
        else:
            self.line(x0, y0, x1, y0, color)
            self.line(x0, y1, x1, y1, color)
            self.line(x0, y0, x0, y1, color)
            self.line(x1, y0, x1, y1, color)

    def circle(self, x: int, y: int, radius: int, color: Color) -> None:
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    self.set_pixel(x + dx, y + dy, color)

    def write_png(self, path: Path) -> None:
        body = bytearray()
        for row in self.pixels:
            body.append(0)
            row_bytes = bytearray()
            for pixel in row:
                row_bytes.extend(bytearray(pixel))
            body.extend(row_bytes)
        raw = zlib.compress(bytes(body), 9)

        def chunk(chunk_type: bytes, data: bytes) -> bytes:
            size = len(data)
            head = struct.pack(">I", size) + chunk_type + data
            crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
            return struct.pack(">I", size) + chunk_type + data + struct.pack(">I", crc)

        ihdr = struct.pack(
            ">IIBBBBB",
            self.width,
            self.height,
            8,
            2,
            0,
            0,
            0,
        )
        png_data = (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", raw)
            + chunk(b"IEND", b"")
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(png_data)


def _draw_axes(chart: Canvas, left: int, right: int, top: int, bottom: int) -> None:
    chart.line(left, bottom, right, bottom, AXIS)
    chart.line(left, top, left, bottom, AXIS)
    for i in range(5):
        x = left + int((right - left) * (i / 4))
        chart.line(x, top, x, bottom, GRID)
        chart.line(left, top + int((bottom - top) * (i / 4)), right, top + int((bottom - top) * (i / 4)), GRID)


def _norm(value: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.0
    return (value - lo) / (hi - lo)


def _plot_line_chart(
    out_path: Path,
    points: list[tuple[str, float, bool]],
    title: str,
) -> None:
    if not points:
        return

    width, height = 1200, 700
    canvas = Canvas(width, height)
    left, right = 100, width - 80
    top, bottom = 120, height - 90
    _draw_axes(canvas, left, right, top, bottom)

    values = [p[1] for p in points]
    min_v = min(values) * 0.95
    max_v = max(values) * 1.05
    if min_v == max_v:
        min_v -= 0.1
        max_v += 0.1

    def point_to_xy(index: int, value: float) -> tuple[int, int]:
        x = left + int((right - left) * (index / max(len(points) - 1, 1)))
        y = bottom - int((_norm(value, min_v, max_v)) * (bottom - top))
        return x, y

    for idx in range(len(points) - 1):
        x0, y0 = point_to_xy(idx, points[idx][1])
        x1, y1 = point_to_xy(idx + 1, points[idx + 1][1])
        color = GOOD if points[idx + 1][2] else MID
        canvas.line(x0, y0, x1, y1, color)

    for idx, (_, value, accepted) in enumerate(points):
        x, y = point_to_xy(idx, value)
        canvas.circle(x, y, 5, GOOD if accepted else BAD)

    for x in range(len(points)):
        px, py = point_to_xy(x, points[x][1])
        canvas.line(px, py + 8, px, bottom, AXIS)
        canvas.set_pixel(px, bottom + 2, TEXT)

    canvas.line(left + 1, top + 20, right - 1, top + 20, GRID)
    canvas.set_pixel(left + 2, top + 5, TEXT)

    # Simple title marker in shape (no text due no font dependency)
    canvas.rect(left + 4, 22, left + 14, 36, AXIS, fill=False)
    canvas.line(right - 200, 34, right - 80, 34, AXIS)
    canvas.set_pixel(right - 60, 34, TEXT)

    canvas.write_png(out_path)
    _write_caption(
        out_path.with_suffix(".txt"),
        [
            title,
            f"points={len(points)}",
            f"min={min_v:.4f}",
            f"max={max_v:.4f}",
        ],
    )


def _plot_anchor_bars(out_path: Path, anchors: list[dict[str, object]]) -> None:
    width, height = 1200, 700
    canvas = Canvas(width, height, BG)
    left, right = 120, width - 80
    top, bottom = 140, height - 130
    _draw_axes(canvas, left, right, top, bottom)

    if not anchors:
        canvas.line(left + 1, bottom - 1, right - 1, top + 1, MID)
        canvas.write_png(out_path)
        return

    bars = []
    for row in anchors:
        elo = float(row.get("uci_elo", 0))
        score = float(row.get("score", 0.5))
        bars.append((elo, score))

    bar_space = (right - left) / max(len(bars), 1)
    min_score = min(score for _, score in bars)
    max_score = max(score for _, score in bars)
    if min_score == max_score:
        min_score -= 0.05
        max_score += 0.05

    for idx, (elo, score) in enumerate(bars):
        x0 = int(left + idx * bar_space + bar_space * 0.2)
        x1 = int(left + (idx + 1) * bar_space - bar_space * 0.2)
        y = bottom - int(_norm(score, min_score, max_score) * (bottom - top))
        canvas.rect(x0, y, x1, bottom, GOOD if score > 0.5 else BAD)
        label = int(elo)
        chart_pos = x0 + 6
        for digit in str(label):
            if chart_pos < width - 20:
                chart_pos += 10

    canvas.write_png(out_path)
    _write_caption(
        out_path.with_suffix(".txt"),
        [
            "Stockfish anchor bars",
            f"anchors={len(anchors)}",
            f"elo range={int(min(e for e, _ in bars))}-{int(max(e for e, _ in bars))}",
        ],
    )


def _write_caption(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines), encoding="utf-8")


def load_dashboard_data(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing dashboard data: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = load_dashboard_data(root / "outputs" / "dashboard" / "dashboard_data.json")
    media_root = root / "media" / "submission"
    media_root.mkdir(parents=True, exist_ok=True)

    accepted = [
        (row.get("round_name", f"#{idx}"), float(row.get("score", 0.5)), bool(row.get("accepted", False)))
        for idx, row in enumerate(data.get("accepted_results", []))
    ]
    all_results = [
        (row.get("round_name", f"#{idx}"), float(row.get("score", 0.5)), bool(row.get("accepted", False)))
        for idx, row in enumerate(data.get("all_results", []))
    ]

    if all_results:
        _plot_line_chart(
            media_root / "0x960_score_progression.png",
            all_results,
            "Champion score progression (all attempts)",
        )
    else:
        _plot_line_chart(
            media_root / "0x960_score_progression.png",
            [("n/a", 0.5, True)],
            "Champion score progression (empty)",
        )

    _plot_anchor_bars(
        media_root / "0x960_stockfish_anchors.png",
        data.get("stockfish_anchors", []),
    )

    if accepted:
        _write_caption(
            media_root / "submission_summary.txt",
            [
                "Accepted samples:",
                *(
                    f"{round_name}: {score:.4f} ({'yes' if accepted else 'no'})"
                    for round_name, score, accepted in accepted
                ),
            ],
        )


if __name__ == "__main__":
    main()
