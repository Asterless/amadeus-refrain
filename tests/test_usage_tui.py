"""Tests for usage TUI chart rendering."""

from __future__ import annotations

from src.llm.usage_tui import (
    _nice_ticks,
    render_bar_chart,
    render_dashboard,
    render_line_chart,
    render_stacked_bar_chart,
)


def test_nice_ticks_small() -> None:
    ticks = _nice_ticks(12)
    assert ticks[0] == 0
    assert ticks[-1] >= 12
    assert len(ticks) >= 3
    # Ticks should be evenly spaced
    diffs = [ticks[i + 1] - ticks[i] for i in range(len(ticks) - 1)]
    assert all(abs(d - diffs[0]) < 1e-9 for d in diffs)


def test_nice_ticks_large() -> None:
    ticks = _nice_ticks(37000)
    assert ticks[0] == 0
    assert ticks[-1] >= 37000
    assert len(ticks) >= 3
    # Step should be a nice number
    step = ticks[1] - ticks[0]
    assert step > 0


def test_nice_ticks_zero() -> None:
    ticks = _nice_ticks(0)
    assert ticks[0] == 0
    assert ticks[-1] > 0  # Must produce non-zero scale even for all-zero data
    assert len(ticks) >= 2


def test_render_bar_chart_output() -> None:
    buckets = [f"{h:02d}" for h in range(24)]
    values = [float(i * 10) for i in range(24)]
    result = render_bar_chart(
        buckets=buckets,
        values=values,
        y_label="calls",
        chart_height=10,
        chart_width=80,
        bar_style="green",
    )
    text = result.plain
    assert "calls" in text
    # Should contain at least some bucket labels
    assert any(b in text for b in buckets)
    # Should contain block characters
    assert "\u2588" in text  # full block


def test_render_bar_chart_all_zero() -> None:
    buckets = ["A", "B", "C"]
    values = [0.0, 0.0, 0.0]
    result = render_bar_chart(
        buckets=buckets,
        values=values,
        y_label="count",
        chart_height=5,
        chart_width=40,
    )
    # Should not raise, should produce output
    assert result.plain.strip() != ""


def test_render_stacked_bar_chart_output() -> None:
    buckets = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    series_a = [100.0, 200.0, 150.0, 300.0, 250.0]
    series_b = [50.0, 80.0, 60.0, 100.0, 90.0]
    result = render_stacked_bar_chart(
        buckets=buckets,
        series_a=series_a,
        label_a="input",
        series_b=series_b,
        label_b="output",
        y_label="tokens",
        chart_height=10,
        chart_width=60,
    )
    text = result.plain
    assert "tokens" in text
    assert "input" in text
    assert "output" in text


def test_render_line_chart_output() -> None:
    buckets = [f"{h:02d}" for h in range(24)]
    values: list[float | None] = [float(50 + i * 2) for i in range(24)]
    result = render_line_chart(
        buckets=buckets,
        values=values,
        y_label="cache hit %",
        chart_height=8,
        chart_width=80,
    )
    text = result.plain
    assert "cache hit" in text
    # Should contain at least one line character
    assert any(c in text for c in ("\u00b7", "\u2500", "/", "\\"))


def test_render_line_chart_constant() -> None:
    buckets = ["A", "B", "C", "D"]
    values: list[float | None] = [75.0, 75.0, 75.0, 75.0]
    result = render_line_chart(
        buckets=buckets,
        values=values,
        y_label="pct",
        chart_height=6,
        chart_width=40,
    )
    # Should not raise, should produce output
    assert result.plain.strip() != ""


def test_render_line_chart_with_nones() -> None:
    buckets = ["A", "B", "C", "D", "E"]
    values: list[float | None] = [10.0, None, 30.0, None, 50.0]
    result = render_line_chart(
        buckets=buckets,
        values=values,
        y_label="val",
        chart_height=6,
        chart_width=40,
    )
    # Should not raise, should produce output
    assert result.plain.strip() != ""


def test_render_dashboard() -> None:
    all_buckets = [f"{h:02d}" for h in range(24)]
    timeseries = [
        {
            "bucket": f"{h:02d}",
            "calls": h * 2,
            "input_tokens": h * 1000,
            "cache_read_tokens": h * 800,
            "cache_create_tokens": h * 100,
            "output_tokens": h * 200,
        }
        for h in range(0, 24, 3)  # sparse data: every 3 hours
    ]
    result = render_dashboard(
        title="Today 2026-04-02",
        summary={
            "total_calls": 100,
            "total_input_tokens": 500000,
            "total_output_tokens": 80000,
            "cache_read_tokens": 400000,
            "avg_elapsed_s": 3.5,
        },
        timeseries=timeseries,
        all_buckets=all_buckets,
        chart_width=80,
    )
    text = result.plain.lower()
    assert "calls" in text
    assert "tokens" in text
    assert "cache hit" in text
