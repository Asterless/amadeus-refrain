"""CLI for querying LLM usage stats. Run: uv run python -m src.llm.usage_cli"""

from __future__ import annotations

import argparse
import asyncio
import calendar
from datetime import UTC, datetime, timedelta
from typing import Any

from rich.console import Console

from src.llm.usage import UsageTracker
from src.llm.usage_tui import _local_tz_offset_hours, render_cost_table, render_dashboard

_DB_PATH = "storage/usage.db"


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _cache_hit_pct(data: dict[str, Any]) -> str:
    total = data.get("total_input_tokens", 0)
    if total == 0:
        return "n/a"
    return f"{data.get('cache_read_tokens', 0) / total * 100:.0f}%"


def _print_summary(title: str, data: dict[str, Any]) -> None:
    print(f"\n=== {title} ===")
    print(f"  Total calls:      {data.get('total_calls', 0)}")

    print(f"  Input tokens:     {_fmt_tokens(data.get('total_input_tokens', 0))}")
    print(f"    Non-cached:     {_fmt_tokens(data.get('input_tokens', 0))}")
    print(f"    Cache read:     {_fmt_tokens(data.get('cache_read_tokens', 0))}")
    print(f"    Cache create:   {_fmt_tokens(data.get('cache_create_tokens', 0))}")
    print(f"  Cache hit rate:   {_cache_hit_pct(data)}")
    print(f"  Output tokens:    {_fmt_tokens(data.get('total_output_tokens', 0))}")

    print(f"  Chat calls:       {data.get('chat_calls', 0)}")
    print(f"  Proactive:        {data.get('proactive_calls', 0)}")
    print(f"  Compact:          {data.get('compact_calls', 0)}")
    print(f"  Dream:            {data.get('dream_calls', 0)}")
    print(f"  Tool rounds:      {data.get('total_tool_rounds', 0)}")
    print(f"  Errors:           {data.get('error_count', 0)}")
    print(f"  Avg latency:      {data.get('avg_elapsed_s', 0):.1f}s")


def _print_top(title: str, rows: list[dict[str, Any]], id_key: str) -> None:
    print(f"\n=== {title} ===")
    if not rows:
        print("  (no data)")
        return
    print(f"  {'ID':<15} {'Calls':>6} {'Input':>10} {'Output':>10}")
    print(f"  {'-'*15} {'-'*6} {'-'*10} {'-'*10}")
    for row in rows:
        print(
            f"  {row[id_key]!s:<15} {row['calls']:>6} "
            f"{_fmt_tokens(row['total_input']):>10} {_fmt_tokens(row['total_output']):>10}"
        )


def _summarize_timeseries(ts: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive summary stats directly from timeseries rows.

    This avoids timezone mismatches between the summary query (UTC LIKE)
    and the timeseries query (which applies a tz offset).
    """
    total_calls = sum(r["calls"] for r in ts)
    input_tokens = sum(r["input_tokens"] for r in ts)
    cache_read = sum(r["cache_read_tokens"] for r in ts)
    cache_create = sum(r["cache_create_tokens"] for r in ts)
    output_tokens = sum(r["output_tokens"] for r in ts)
    return {
        "total_calls": total_calls,
        "input_tokens": input_tokens,
        "cache_read_tokens": cache_read,
        "cache_create_tokens": cache_create,
        "total_input_tokens": input_tokens + cache_read + cache_create,
        "total_output_tokens": output_tokens,
        "avg_elapsed_s": 0.0,  # not available from timeseries
    }


async def _run_tui(args: argparse.Namespace) -> None:
    tracker = UsageTracker(db_path=_DB_PATH)
    await tracker.init()
    try:
        tz_offset = _local_tz_offset_hours()
        period = args.tui_period
        date: str | None = args.date

        if period == "day":
            ts = await tracker.timeseries(period="day", date=date, tz_offset_hours=tz_offset)
            if date:
                all_buckets = [f"{h:02d}" for h in range(24)]
                title = f"Daily Usage: {date}"
            else:
                now = datetime.now(UTC).astimezone()
                cur = now.hour
                all_buckets = [f"{(cur + 1 + i) % 24:02d}" for i in range(24)]
                title = "Usage: Last 24 Hours"

        elif period == "week":
            if not date:
                date = datetime.now(UTC).astimezone().strftime("%Y-%m-%d")
            ts = await tracker.timeseries(period="week", date=date, tz_offset_hours=tz_offset)
            end = datetime.strptime(date, "%Y-%m-%d")
            all_buckets = [(end - timedelta(days=6 - i)).strftime("%m-%d") for i in range(7)]
            title = f"Usage: Last 7 Days (ending {date})"

        elif period == "month":
            ts = await tracker.timeseries(period="month", date=date, tz_offset_hours=tz_offset)
            if date:
                year, month = map(int, date.split("-"))
                num_days = calendar.monthrange(year, month)[1]
                all_buckets = [f"{d:02d}" for d in range(1, num_days + 1)]
                title = f"Monthly Usage: {date}"
            else:
                now = datetime.now(UTC).astimezone()
                all_buckets = [(now - timedelta(days=29 - i)).strftime("%m-%d") for i in range(30)]
                title = "Usage: Last 30 Days"

        else:
            msg = f"unknown tui period: {period!r}"
            raise ValueError(msg)

        model_data = await tracker.usage_by_model(
            period=period, date=date, tz_offset_hours=tz_offset,
        )

        summary = _summarize_timeseries(ts)
        dashboard = render_dashboard(title=title, summary=summary, timeseries=ts, all_buckets=all_buckets)
        console = Console()
        console.print(dashboard)
        if model_data:
            console.print(render_cost_table(model_data))
    finally:
        await tracker.close()


async def _run(args: argparse.Namespace) -> None:
    if args.command == "tui":
        await _run_tui(args)
        return

    tracker = UsageTracker(db_path=_DB_PATH)
    await tracker.init()
    try:
        if args.command == "today":
            data = await tracker.summary_today()
            _print_summary("Today", data)
        elif args.command == "month":
            data = await tracker.summary_month(args.month)
            _print_summary(f"Month: {args.month or 'current'}", data)
        elif args.command == "top-users":
            rows = await tracker.top_users(days=args.days)
            _print_top(f"Top Users (last {args.days} days)", rows, "user_id")
        elif args.command == "top-groups":
            rows = await tracker.top_groups(days=args.days)
            _print_top(f"Top Groups (last {args.days} days)", rows, "group_id")
    finally:
        await tracker.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Usage Stats")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("today", help="Today's usage summary")

    month_p = sub.add_parser("month", help="Monthly usage summary")
    month_p.add_argument("month", nargs="?", default=None, help="YYYY-MM (default: current)")

    users_p = sub.add_parser("top-users", help="Top users by token consumption")
    users_p.add_argument("--days", type=int, default=7, help="Lookback days (default: 7)")

    groups_p = sub.add_parser("top-groups", help="Top groups by token consumption")
    groups_p.add_argument("--days", type=int, default=7, help="Lookback days (default: 7)")

    tui_p = sub.add_parser("tui", help="Rich TUI dashboard with charts")
    tui_sub = tui_p.add_subparsers(dest="tui_period", required=True)

    day_p = tui_sub.add_parser("day", help="Last 24 hours (by hour)")
    day_p.add_argument("date", nargs="?", default=None, help="YYYY-MM-DD (default: last 24h)")

    month_tui = tui_sub.add_parser("month", help="Last 30 days (by day)")
    month_tui.add_argument("date", nargs="?", default=None, help="YYYY-MM (default: last 30 days)")

    week_p = tui_sub.add_parser("week", help="Last 7 days (by day)")
    week_p.add_argument("date", nargs="?", default=None, help="YYYY-MM-DD end date (default: last 7 days)")

    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
