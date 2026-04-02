"""CLI for querying LLM usage stats. Run: uv run python -m src.llm.usage_cli"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

from src.llm.usage import UsageTracker

_DB_PATH = "storage/usage.db"


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _print_summary(title: str, data: dict[str, Any]) -> None:
    print(f"\n=== {title} ===")
    print(f"  Total calls:    {data.get('total_calls', 0)}")
    print(f"  Input tokens:   {_fmt_tokens(data.get('total_input_tokens', 0))}")
    print(f"  Output tokens:  {_fmt_tokens(data.get('total_output_tokens', 0))}")
    print(f"  Chat calls:     {data.get('chat_calls', 0)}")
    print(f"  Proactive:      {data.get('proactive_calls', 0)}")
    print(f"  Compact:        {data.get('compact_calls', 0)}")
    print(f"  Errors:         {data.get('error_count', 0)}")
    print(f"  Avg latency:    {data.get('avg_elapsed_s', 0):.1f}s")


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


async def _run(args: argparse.Namespace) -> None:
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

    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
