"""TUI chart rendering for the LLM usage dashboard.

Produces Rich Text objects with vertical bar charts, stacked bar charts,
line charts, and a combined dashboard view.
"""

from __future__ import annotations

import math
import shutil
from datetime import UTC, datetime
from typing import Any

from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

_NICE_BASES = [1, 2, 5]


def _nice_ticks(max_val: float, num_ticks: int = 5) -> list[float]:
    """Compute human-friendly evenly-spaced Y axis ticks from 0 to >= max_val.

    Uses a 1-2-5 nice step algorithm.  Always starts at 0.
    """
    if max_val <= 0:
        # Fallback: produce a small scale so the chart isn't degenerate.
        return [float(i) for i in range(num_ticks + 1)]

    raw_step = max_val / max(num_ticks - 1, 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    # Pick the smallest nice step >= raw_step
    nice_step = magnitude  # fallback
    for base in _NICE_BASES:
        candidate = base * magnitude
        if candidate >= raw_step - 1e-12:
            nice_step = candidate
            break

    # Build tick list from 0 upward
    ticks: list[float] = []
    val = 0.0
    while val < max_val + nice_step * 0.01:
        ticks.append(val)
        val += nice_step
    # Ensure the last tick >= max_val
    if ticks[-1] < max_val - 1e-9:
        ticks.append(ticks[-1] + nice_step)
    return ticks


def _nice_range_ticks(lo: float, hi: float, num_ticks: int = 5) -> list[float]:
    """Nice ticks for an arbitrary range (not necessarily from 0)."""
    if abs(hi - lo) < 1e-12:
        # Flat data: create a range around the value
        if abs(lo) < 1e-12:
            lo, hi = -1.0, 1.0
        else:
            spread = abs(lo) * 0.1
            lo -= spread
            hi += spread

    span = hi - lo
    raw_step = span / max(num_ticks - 1, 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    nice_step = magnitude
    for base in _NICE_BASES:
        candidate = base * magnitude
        if candidate >= raw_step - 1e-12:
            nice_step = candidate
            break

    # Round lo down and hi up to nice boundaries
    nice_lo = math.floor(lo / nice_step) * nice_step
    nice_hi = math.ceil(hi / nice_step) * nice_step

    ticks: list[float] = []
    val = nice_lo
    while val <= nice_hi + nice_step * 0.01:
        ticks.append(round(val, 10))
        val += nice_step
    return ticks


def _fmt_axis_label(val: float) -> str:
    """Format a numeric value with K/M suffixes for Y axis display."""
    abs_val = abs(val)
    if abs_val >= 1_000_000:
        formatted = f"{val / 1_000_000:.1f}M"
    elif abs_val >= 1_000:
        formatted = f"{val / 1_000:.1f}K"
    elif val == int(val):
        formatted = str(int(val))
    else:
        formatted = f"{val:.1f}"
    return formatted


def _auto_unit(max_val: float) -> str:
    """Return 'M', 'K', or '' based on magnitude."""
    if max_val >= 1_000_000:
        return "M"
    if max_val >= 1_000:
        return "K"
    return ""


def _local_tz_offset_hours() -> float:
    """Return local timezone offset in hours."""
    now = datetime.now(UTC)
    local_now = now.astimezone()
    offset = local_now.utcoffset()
    if offset is None:
        return 0.0
    return offset.total_seconds() / 3600


def _local_tz_label() -> str:
    """Return timezone label like 'UTC+8' or 'UTC-5'."""
    hours = _local_tz_offset_hours()
    if hours == int(hours):
        h = int(hours)
        return f"UTC{h:+d}" if h != 0 else "UTC"
    return f"UTC{hours:+.1f}"


# ---------------------------------------------------------------------------
# X axis helpers
# ---------------------------------------------------------------------------


def _compute_x_label_step(num_buckets: int, available_width: int, label_width: int) -> int:
    """Decide how many buckets to skip between X axis labels."""
    if num_buckets == 0:
        return 1
    # Each label needs at least label_width+1 chars to not overlap
    labels_that_fit = max(available_width // (label_width + 1), 1)
    step = max(math.ceil(num_buckets / labels_that_fit), 1)
    return step


# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------


def render_bar_chart(
    *,
    buckets: list[str],
    values: list[float],
    y_label: str,
    chart_height: int = 15,
    chart_width: int = 0,
    bar_style: str = "green",
    y_axis_width: int = 0,
) -> Text:
    """Render a vertical bar chart as a Rich Text object.

    Uses full-block and half-block characters for sub-cell Y precision.
    """
    if chart_width <= 0:
        chart_width = shutil.get_terminal_size(fallback=(80, 24)).columns

    n = len(buckets)
    if n == 0:
        return Text(f"  {y_label}: (no data)\n")

    y_max_raw = max(values) if values else 0.0
    ticks = _nice_ticks(y_max_raw)
    y_max = ticks[-1]

    # Y axis label width
    tick_labels = [_fmt_axis_label(t) for t in ticks]
    if y_axis_width <= 0:
        y_axis_width = max(len(lbl) for lbl in tick_labels) + 1

    bar_area_width = chart_width - y_axis_width - 1  # -1 for axis line
    if bar_area_width < n:
        bar_area_width = n

    # Each bucket gets equal space in the bar area
    chars_per_bucket = bar_area_width / n

    result = Text()
    result.append(f"  {y_label}\n")

    # Build rows top to bottom
    for row in range(chart_height, 0, -1):
        # Value range for this row
        row_top = (row / chart_height) * y_max
        row_bot = ((row - 1) / chart_height) * y_max
        row_mid = (row_bot + row_top) / 2

        # Y axis tick label
        tick_str = ""
        for i, t in enumerate(ticks):
            if abs(t - row_top) < (y_max / chart_height) * 0.4:
                tick_str = tick_labels[i]
                break
        line_prefix = tick_str.rjust(y_axis_width - 1) + " " + "\u2502"

        result.append(line_prefix)

        # Bar cells
        for idx in range(n):
            cell_width = round(chars_per_bucket * (idx + 1)) - round(chars_per_bucket * idx)
            val = values[idx]
            if val >= row_top:
                # Full block
                result.append("\u2588" * cell_width, style=bar_style)
            elif val > row_mid:
                # Half block (top half filled via bottom-half char)
                result.append("\u2584" * cell_width, style=bar_style)
            else:
                result.append(" " * cell_width)

        result.append("\n")

    # X axis line
    x_line = "\u2500" * bar_area_width
    result.append(" " * y_axis_width + "\u2514" + x_line + "\n")

    # X axis labels
    label_width = max(len(b) for b in buckets) if buckets else 2
    step = _compute_x_label_step(n, bar_area_width, label_width)

    label_line = [" "] * (y_axis_width + 1)  # offset for Y axis + corner
    for idx in range(0, n, step):
        pos = y_axis_width + 1 + round(chars_per_bucket * idx)
        lbl = buckets[idx]
        for ci, ch in enumerate(lbl):
            target = pos + ci
            if target < len(label_line):
                label_line[target] = ch
            else:
                while len(label_line) <= target:
                    label_line.append(" ")
                label_line[target] = ch

    result.append("".join(label_line).rstrip() + "\n")
    return result


# ---------------------------------------------------------------------------
# Stacked bar chart
# ---------------------------------------------------------------------------


def render_stacked_bar_chart(
    *,
    buckets: list[str],
    series_a: list[float],
    label_a: str,
    series_b: list[float],
    label_b: str,
    y_label: str,
    chart_height: int = 15,
    chart_width: int = 0,
    style_a: str = "cyan",
    style_b: str = "yellow",
    y_axis_width: int = 0,
) -> Text:
    """Render a two-series stacked vertical bar chart.

    series_a on bottom (full block), series_b stacked on top (light shade).
    """
    if chart_width <= 0:
        chart_width = shutil.get_terminal_size(fallback=(80, 24)).columns

    n = len(buckets)
    if n == 0:
        return Text(f"  {y_label}: (no data)\n")

    # Compute stacked totals
    stacked = [a + b for a, b in zip(series_a, series_b, strict=True)]
    y_max_raw = max(stacked) if stacked else 0.0
    ticks = _nice_ticks(y_max_raw)
    y_max = ticks[-1]

    tick_labels = [_fmt_axis_label(t) for t in ticks]
    if y_axis_width <= 0:
        y_axis_width = max(len(lbl) for lbl in tick_labels) + 1

    bar_area_width = chart_width - y_axis_width - 1
    if bar_area_width < n:
        bar_area_width = n

    chars_per_bucket = bar_area_width / n

    result = Text()
    # Legend header
    result.append(f"  {y_label}  ")
    result.append("\u2588", style=style_a)
    result.append(f" {label_a}  ")
    result.append("\u2591", style=style_b)
    result.append(f" {label_b}\n")

    for row in range(chart_height, 0, -1):
        row_top = (row / chart_height) * y_max
        row_bot = ((row - 1) / chart_height) * y_max
        row_mid = (row_bot + row_top) / 2

        tick_str = ""
        for i, t in enumerate(ticks):
            if abs(t - row_top) < (y_max / chart_height) * 0.4:
                tick_str = tick_labels[i]
                break
        line_prefix = tick_str.rjust(y_axis_width - 1) + " " + "\u2502"
        result.append(line_prefix)

        for idx in range(n):
            cell_width = round(chars_per_bucket * (idx + 1)) - round(chars_per_bucket * idx)
            a_val = series_a[idx]
            total_val = stacked[idx]

            if total_val >= row_top:
                # This cell is fully within the stacked bar
                if a_val >= row_top:
                    # Entirely series_a
                    result.append("\u2588" * cell_width, style=style_a)
                elif a_val > row_bot:
                    # Transition zone: partially a, partially b
                    result.append("\u2588" * cell_width, style=style_a)
                else:
                    # Entirely series_b
                    result.append("\u2591" * cell_width, style=style_b)
            elif total_val > row_mid:
                # Half block at the top of the bar
                if a_val >= row_bot:
                    result.append("\u2584" * cell_width, style=style_a)
                else:
                    result.append("\u2584" * cell_width, style=style_b)
            elif a_val >= row_top:
                result.append("\u2588" * cell_width, style=style_a)
            elif a_val > row_mid:
                result.append("\u2584" * cell_width, style=style_a)
            else:
                result.append(" " * cell_width)

        result.append("\n")

    # X axis
    x_line = "\u2500" * bar_area_width
    result.append(" " * y_axis_width + "\u2514" + x_line + "\n")

    label_width = max(len(b) for b in buckets) if buckets else 2
    step = _compute_x_label_step(n, bar_area_width, label_width)
    label_line = [" "] * (y_axis_width + 1)
    for idx in range(0, n, step):
        pos = y_axis_width + 1 + round(chars_per_bucket * idx)
        lbl = buckets[idx]
        for ci, ch in enumerate(lbl):
            target = pos + ci
            while len(label_line) <= target:
                label_line.append(" ")
            label_line[target] = ch
    result.append("".join(label_line).rstrip() + "\n")
    return result


# ---------------------------------------------------------------------------
# Line chart
# ---------------------------------------------------------------------------


def render_line_chart(
    *,
    buckets: list[str],
    values: list[float | None],
    y_label: str,
    chart_height: int = 8,
    chart_width: int = 0,
    line_style: str = "magenta",
    warn_below: float | None = None,
    y_axis_width: int = 0,
    y_floor: float | None = None,
    y_ceil: float | None = None,
) -> Text:
    """Render a line chart with adaptive Y range.

    Y range does NOT start from 0; uses _nice_range_ticks for scaling.
    None values leave gaps in the line.
    *y_floor* / *y_ceil* clamp the Y axis so ticks never go below / above.
    """
    if chart_width <= 0:
        chart_width = shutil.get_terminal_size(fallback=(80, 24)).columns

    n = len(buckets)
    if n == 0:
        return Text(f"  {y_label}: (no data)\n")

    # Filter non-None values
    real_vals = [v for v in values if v is not None]
    if not real_vals:
        return Text(f"  {y_label}: (no data)\n")

    v_min = min(real_vals)
    v_max = max(real_vals)

    # Add padding
    span = v_max - v_min
    padding = max(abs(v_min) * 0.1, 1.0) if span < 1e-12 else span * 0.1
    ticks = _nice_range_ticks(v_min - padding, v_max + padding)

    # Clamp ticks to floor/ceil bounds
    if y_floor is not None:
        ticks = [t for t in ticks if t >= y_floor - 1e-9]
        if not ticks or ticks[0] > y_floor + 1e-9:
            ticks.insert(0, y_floor)
    if y_ceil is not None:
        ticks = [t for t in ticks if t <= y_ceil + 1e-9]
        if not ticks or ticks[-1] < y_ceil - 1e-9:
            ticks.append(y_ceil)

    y_lo = ticks[0]
    y_hi = ticks[-1]
    y_range = y_hi - y_lo
    if y_range < 1e-12:
        y_range = 1.0

    tick_labels = [_fmt_axis_label(t) for t in ticks]
    if y_axis_width <= 0:
        y_axis_width = max(len(lbl) for lbl in tick_labels) + 1

    plot_area_width = chart_width - y_axis_width - 1
    if plot_area_width < n:
        plot_area_width = n

    # Build 2D grid [row][col], row 0 = top
    grid: list[list[str]] = [[" "] * plot_area_width for _ in range(chart_height)]

    # Map each data point to a column position and row
    def _col_for(idx: int) -> int:
        if n == 1:
            return plot_area_width // 2
        return round(idx * (plot_area_width - 1) / (n - 1))

    def _row_for(val: float) -> int:
        frac = (val - y_lo) / y_range
        frac = max(0.0, min(1.0, frac))
        row = chart_height - 1 - round(frac * (chart_height - 1))
        return row

    # Place data points and connecting lines
    prev_col: int | None = None
    prev_row: int | None = None

    for idx in range(n):
        val = values[idx]
        if val is None:
            prev_col = None
            prev_row = None
            continue

        col = _col_for(idx)
        row = _row_for(val)
        grid[row][col] = "\u00b7"  # middle dot for data point

        # Draw connecting line from previous point
        if prev_col is not None and prev_row is not None:
            _draw_line_segment(grid, prev_row, prev_col, row, col)

        prev_col = col
        prev_row = row

    # Render
    result = Text()
    result.append(f"  {y_label}\n")

    for row_idx in range(chart_height):
        row_val = y_hi - (row_idx / max(chart_height - 1, 1)) * y_range

        tick_str = ""
        for i, t in enumerate(ticks):
            if abs(t - row_val) < (y_range / chart_height) * 0.5:
                tick_str = tick_labels[i]
                break
        line_prefix = tick_str.rjust(y_axis_width - 1) + " " + "\u2502"
        result.append(line_prefix)

        for col_idx in range(plot_area_width):
            ch = grid[row_idx][col_idx]
            if ch != " ":
                result.append(ch, style=line_style)
            else:
                result.append(ch)

        result.append("\n")

    # X axis
    x_line = "\u2500" * plot_area_width
    result.append(" " * y_axis_width + "\u2514" + x_line + "\n")

    label_width = max(len(b) for b in buckets) if buckets else 2
    step = _compute_x_label_step(n, plot_area_width, label_width)
    label_line = [" "] * (y_axis_width + 1)
    for idx in range(0, n, step):
        pos = y_axis_width + 1 + _col_for(idx)
        lbl = buckets[idx]
        for ci, ch in enumerate(lbl):
            target = pos + ci
            while len(label_line) <= target:
                label_line.append(" ")
            label_line[target] = ch
    result.append("".join(label_line).rstrip() + "\n")
    return result


def _draw_line_segment(
    grid: list[list[str]],
    r1: int,
    c1: int,
    r2: int,
    c2: int,
) -> None:
    """Draw a line between two points on the grid using /, \\, and horizontal chars."""
    dc = c2 - c1
    dr = r2 - r1

    if dc == 0:
        # Vertical - just fill intermediate cells
        step = 1 if r2 > r1 else -1
        for r in range(r1 + step, r2, step):
            if grid[r][c1] == " ":
                grid[r][c1] = "|"
        return

    # Bresenham-like: step through columns
    for c in range(c1 + 1, c2):
        # Interpolate row
        frac = (c - c1) / dc
        r = r1 + frac * dr
        ri = round(r)
        ri = max(0, min(len(grid) - 1, ri))

        if grid[ri][c] == " ":
            if dr < 0:
                grid[ri][c] = "/"
            elif dr > 0:
                grid[ri][c] = "\\"
            else:
                grid[ri][c] = "\u2500"  # horizontal


# ---------------------------------------------------------------------------
# Cost breakdown
# ---------------------------------------------------------------------------

# Anthropic API pricing: (input, cache_read, cache_write, output) in $/MTok
_PRICING: dict[str, tuple[float, float, float, float]] = {
    "opus": (15.0, 1.50, 18.75, 75.0),
    "sonnet": (3.0, 0.30, 3.75, 15.0),
    "haiku": (0.80, 0.08, 1.00, 4.0),
}


def _match_pricing(model: str) -> tuple[float, float, float, float]:
    """Match a model string to its pricing tier. Falls back to sonnet."""
    model_lower = model.lower()
    for tier, prices in _PRICING.items():
        if tier in model_lower:
            return prices
    return _PRICING["sonnet"]


def _compute_cost(
    input_tokens: int,
    cache_read_tokens: int,
    cache_create_tokens: int,
    output_tokens: int,
    pricing: tuple[float, float, float, float],
) -> float:
    """Compute cost in USD from token counts and pricing rates."""
    p_in, p_cr, p_cw, p_out = pricing
    return (
        input_tokens * p_in
        + cache_read_tokens * p_cr
        + cache_create_tokens * p_cw
        + output_tokens * p_out
    ) / 1_000_000


def _fmt_cost(cost: float) -> str:
    if cost >= 100:
        return f"${cost:.2f}"
    if cost >= 1:
        return f"${cost:.3f}"
    return f"${cost:.4f}"


def _token_cell(tokens: int, rate: float) -> Text:
    """Format a token count with its unit price as a two-line cell."""
    t = Text()
    t.append(_fmt_axis_label(float(tokens)))
    t.append(f"\n@${rate}/MTok", style="dim")
    return t


def render_cost_table(model_usage: list[dict[str, Any]]) -> Table:
    """Render a per-model cost breakdown table."""
    table = Table(title="Cost Breakdown (USD)", show_edge=True, pad_edge=True)
    table.add_column("Model", style="bold", min_width=12)
    table.add_column("Calls", justify="right")
    table.add_column("Input", justify="right")
    table.add_column("Cache Read", justify="right")
    table.add_column("Cache Write", justify="right")
    table.add_column("Output", justify="right")
    table.add_column("Hit%", justify="right")
    table.add_column("Cost", justify="right", style="bold green")

    total_cost = 0.0
    total_calls = 0

    for row in model_usage:
        model = row["model"]
        calls = row["calls"]
        inp = row["input_tokens"]
        cr = row["cache_read_tokens"]
        cw = row["cache_create_tokens"]
        out = row["output_tokens"]
        pricing = _match_pricing(model)
        p_in, p_cr, p_cw, p_out = pricing
        cost = _compute_cost(inp, cr, cw, out, pricing)
        total_cost += cost
        total_calls += calls

        model_total_in = inp + cr + cw
        hit_pct = (
            f"{min(100.0, max(0.0, cr / model_total_in * 100)):.0f}%"
            if model_total_in > 0 else "n/a"
        )

        table.add_row(
            model,
            str(calls),
            _token_cell(inp, p_in),
            _token_cell(cr, p_cr),
            _token_cell(cw, p_cw),
            _token_cell(out, p_out),
            hit_pct,
            _fmt_cost(cost),
        )

    table.add_section()
    table.add_row(
        "Total", str(total_calls),
        "", "", "", "", "",
        _fmt_cost(total_cost),
        style="bold",
    )

    return table


# ---------------------------------------------------------------------------
# Y-axis width helpers (for cross-chart alignment)
# ---------------------------------------------------------------------------


def _y_width_for_bar(values: list[float]) -> int:
    """Compute Y-axis label width for a bar chart's data."""
    y_max = max(values) if values else 0.0
    ticks = _nice_ticks(y_max)
    return max(len(_fmt_axis_label(t)) for t in ticks) + 1


def _y_width_for_line(
    values: list[float | None],
    y_floor: float | None = None,
    y_ceil: float | None = None,
) -> int:
    """Compute Y-axis label width for a line chart's data."""
    real = [v for v in values if v is not None]
    if not real:
        return 4
    v_min, v_max = min(real), max(real)
    span = v_max - v_min
    pad = max(abs(v_min) * 0.1, 1.0) if span < 1e-12 else span * 0.1
    ticks = _nice_range_ticks(v_min - pad, v_max + pad)
    if y_floor is not None:
        ticks = [t for t in ticks if t >= y_floor - 1e-9]
    if y_ceil is not None:
        ticks = [t for t in ticks if t <= y_ceil + 1e-9]
    if not ticks:
        ticks = [0.0]
    return max(len(_fmt_axis_label(t)) for t in ticks) + 1


# ---------------------------------------------------------------------------
# Dashboard composer
# ---------------------------------------------------------------------------


def render_dashboard(
    *,
    title: str,
    summary: dict[str, Any],
    timeseries: list[dict[str, Any]],
    all_buckets: list[str],
    chart_width: int = 0,
) -> Text:
    """Compose header + summary + 3 charts into a single Rich Text."""
    if chart_width <= 0:
        chart_width = shutil.get_terminal_size(fallback=(80, 24)).columns

    tz_label = _local_tz_label()

    # Build lookup from timeseries
    ts_map: dict[str, dict[str, Any]] = {}
    for row in timeseries:
        ts_map[row["bucket"]] = row

    # Extract per-bucket values with defaults
    calls: list[float] = []
    input_tokens: list[float] = []
    output_tokens: list[float] = []
    cache_hit_pcts: list[float | None] = []

    for b in all_buckets:
        row = ts_map.get(b)
        if row:
            calls.append(float(row.get("calls", 0)))
            inp = row.get("input_tokens", 0)
            cr = row.get("cache_read_tokens", 0)
            cc = row.get("cache_create_tokens", 0)
            out = row.get("output_tokens", 0)
            total_in = inp + cr + cc
            input_tokens.append(float(total_in))
            output_tokens.append(float(out))
            if total_in > 0:
                cache_hit_pcts.append(min(100.0, max(0.0, cr / total_in * 100)))
            else:
                cache_hit_pcts.append(None)
        else:
            calls.append(0.0)
            input_tokens.append(0.0)
            output_tokens.append(0.0)
            cache_hit_pcts.append(None)

    # Header
    result = Text()
    header_text = f" {title} ({tz_label}) "
    pad_total = max(chart_width - len(header_text), 4)
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    result.append("\u2550" * pad_left + header_text + "\u2550" * pad_right + "\n", style="bold")

    # Summary line
    total_calls = summary.get("total_calls", 0)
    total_input = summary.get("total_input_tokens", 0)
    total_output = summary.get("total_output_tokens", 0)
    cache_read = summary.get("cache_read_tokens", 0)
    avg_s = summary.get("avg_elapsed_s", 0)

    cache_create = summary.get("cache_create_tokens", 0)
    input_only = summary.get("input_tokens", 0)

    cache_pct: float | None = None
    if total_input > 0:
        cache_pct = min(100.0, max(0.0, cache_read / total_input * 100))
    cache_pct_str = "n/a" if cache_pct is None else f"{cache_pct:.0f}%"

    summary_text = (
        f"Calls: {total_calls} \u2502 "
        f"Input: {_fmt_axis_label(float(total_input))} \u2502 "
        f"Output: {_fmt_axis_label(float(total_output))} \u2502 "
        f"Cache hit: {cache_pct_str} \u2502 "
        f"Avg: {avg_s:.1f}s"
    )
    result.append(summary_text + "\n")

    # Cache hit rate detail
    if cache_pct is not None:
        non_cached_pct = min(100.0, max(0.0, input_only / total_input * 100))
        create_pct = min(100.0, max(0.0, cache_create / total_input * 100))
        detail = (
            f"  Cache detail: "
            f"non-cached {_fmt_axis_label(float(input_only))} ({non_cached_pct:.0f}%) | "
            f"read {_fmt_axis_label(float(cache_read))} ({cache_pct:.0f}%) | "
            f"create {_fmt_axis_label(float(cache_create))} ({create_pct:.0f}%)"
        )
        result.append(detail + "\n", style="dim")
    result.append("\n")

    # Pre-compute max Y-axis width so all charts align vertically
    stacked_totals = [i + o for i, o in zip(input_tokens, output_tokens, strict=True)]
    y_width = max(
        _y_width_for_bar(calls),
        _y_width_for_bar(stacked_totals),
        _y_width_for_line(cache_hit_pcts, y_floor=0, y_ceil=100),
    )

    # Chart 1: calls bar chart
    calls_chart = render_bar_chart(
        buckets=all_buckets,
        values=calls,
        y_label="calls",
        chart_height=10,
        chart_width=chart_width,
        bar_style="green",
        y_axis_width=y_width,
    )
    result.append_text(calls_chart)
    result.append("\n")

    # Chart 2: tokens stacked bar chart
    tokens_chart = render_stacked_bar_chart(
        buckets=all_buckets,
        series_a=input_tokens,
        label_a="input tokens",
        series_b=output_tokens,
        label_b="output tokens",
        y_label="tokens",
        chart_height=10,
        chart_width=chart_width,
        y_axis_width=y_width,
    )
    result.append_text(tokens_chart)
    result.append("\n")

    # Chart 3: cache hit % line chart (Y axis clamped to 0-100)
    cache_chart = render_line_chart(
        buckets=all_buckets,
        values=cache_hit_pcts,
        y_label="cache hit %",
        chart_height=8,
        chart_width=chart_width,
        line_style="magenta",
        y_axis_width=y_width,
        y_floor=0,
        y_ceil=100,
    )
    result.append_text(cache_chart)

    return result
