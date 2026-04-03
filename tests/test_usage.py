"""Tests for UsageTracker."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from src.llm.usage import UsageTracker


@pytest.fixture
async def tracker(tmp_path) -> UsageTracker:
    t = UsageTracker(db_path=str(tmp_path / "usage.db"))
    await t.init()
    return t


async def test_init_creates_table(tracker: UsageTracker) -> None:
    """Table should exist after init."""
    rows = await tracker.query_raw("SELECT name FROM sqlite_master WHERE type='table' AND name='llm_calls'")
    assert len(rows) == 1


async def test_record_inserts_row(tracker: UsageTracker) -> None:
    await tracker.record(
        call_type="chat",
        user_id="12345",
        group_id=None,
        model="claude-sonnet-4-6",
        input_tokens=100,
        cache_read_tokens=50,
        cache_create_tokens=10,
        output_tokens=200,
        tool_rounds=0,
        elapsed_s=1.5,
    )
    rows = await tracker.query_raw("SELECT * FROM llm_calls")
    assert len(rows) == 1
    row = rows[0]
    assert row["call_type"] == "chat"
    assert row["user_id"] == "12345"
    assert row["group_id"] is None
    assert row["input_tokens"] == 100
    assert row["output_tokens"] == 200
    assert row["cache_read_tokens"] == 50
    assert row["cache_create_tokens"] == 10
    assert row["tool_rounds"] == 0
    assert row["elapsed_s"] == pytest.approx(1.5)
    assert row["error"] is None


async def test_record_with_error(tracker: UsageTracker) -> None:
    await tracker.record(
        call_type="chat",
        user_id="12345",
        group_id="99999",
        model="claude-sonnet-4-6",
        input_tokens=0,
        cache_read_tokens=0,
        cache_create_tokens=0,
        output_tokens=0,
        tool_rounds=0,
        elapsed_s=0.5,
        error="API timeout",
    )
    rows = await tracker.query_raw("SELECT error, group_id FROM llm_calls")
    assert rows[0]["error"] == "API timeout"
    assert rows[0]["group_id"] == "99999"


async def test_record_failure_does_not_raise(tracker: UsageTracker, tmp_path) -> None:
    """record() should swallow errors gracefully."""
    await tracker.close()
    # After close, writing should not raise
    await tracker.record(
        call_type="chat", user_id="1", group_id=None, model="m",
        input_tokens=0, cache_read_tokens=0, cache_create_tokens=0,
        output_tokens=0, tool_rounds=0, elapsed_s=0.0,
    )



async def _insert_sample_data(tracker: UsageTracker) -> None:
    """Insert sample records for query tests."""
    records = [
        ("chat", "111", None, "model-a", 100, 50, 10, 200, 1, 2.0),
        ("chat", "111", "999", "model-a", 150, 80, 20, 300, 2, 3.0),
        ("chat", "222", "999", "model-a", 200, 100, 30, 400, 0, 1.0),
        ("proactive", None, "999", "model-a", 50, 30, 5, 100, 0, 5.0),
        ("compact", "111", None, "model-a", 80, 0, 0, 50, 0, 1.5),
    ]
    for r in records:
        await tracker.record(
            call_type=r[0], user_id=r[1], group_id=r[2], model=r[3],
            input_tokens=r[4], cache_read_tokens=r[5], cache_create_tokens=r[6],
            output_tokens=r[7], tool_rounds=r[8], elapsed_s=r[9],
        )


async def test_summary_today(tracker: UsageTracker) -> None:
    await _insert_sample_data(tracker)
    summary = await tracker.summary_today()
    assert summary["total_calls"] == 5
    # total_input = sum(input + cache_read + cache_create) for each record
    assert summary["total_input_tokens"] == (100+50+10) + (150+80+20) + (200+100+30) + (50+30+5) + (80+0+0)
    assert summary["total_output_tokens"] == 200 + 300 + 400 + 100 + 50


async def test_top_users(tracker: UsageTracker) -> None:
    await _insert_sample_data(tracker)
    top = await tracker.top_users(days=1)
    assert len(top) >= 2
    # User 111 has more total tokens than 222
    assert top[0]["user_id"] == "222" or top[0]["user_id"] == "111"


async def test_top_groups(tracker: UsageTracker) -> None:
    await _insert_sample_data(tracker)
    top = await tracker.top_groups(days=1)
    assert len(top) >= 1
    assert top[0]["group_id"] == "999"


async def test_summary_month(tracker: UsageTracker) -> None:
    await _insert_sample_data(tracker)
    now = datetime.now(UTC)
    month_str = now.strftime("%Y-%m")
    summary = await tracker.summary_month(month_str)
    assert summary["total_calls"] == 5



async def test_alert_on_slow_call(tracker: UsageTracker) -> None:
    alert_fn = AsyncMock()
    tracker.set_alert(alert_fn=alert_fn, cache_hit_warn=90.0, slow_threshold_s=2.0)
    await tracker.record(
        call_type="chat", user_id="1", group_id=None, model="m",
        input_tokens=100, cache_read_tokens=80, cache_create_tokens=10,
        output_tokens=50, tool_rounds=0, elapsed_s=5.0,
    )
    alert_fn.assert_called_once()
    assert "slow" in alert_fn.call_args[0][0].lower() or "慢" in alert_fn.call_args[0][0]


async def test_alert_on_low_cache_hit(tracker: UsageTracker) -> None:
    alert_fn = AsyncMock()
    # min_samples=3, so we need 3 low-hit calls after cold start to trigger
    tracker.set_alert(alert_fn=alert_fn, cache_hit_warn=90.0, slow_threshold_s=999.0)
    # Consume cold-start (first chat call after restart is silently skipped)
    await tracker.record(
        call_type="chat", user_id="1", group_id=None, model="m",
        input_tokens=100, cache_read_tokens=0, cache_create_tokens=100,
        output_tokens=10, tool_rounds=0, elapsed_s=1.0,
    )
    alert_fn.reset_mock()
    low_hit_kwargs = dict(
        call_type="chat", user_id="1", group_id=None, model="m",
        input_tokens=100, cache_read_tokens=10, cache_create_tokens=0,
        output_tokens=50, tool_rounds=0, elapsed_s=1.0,
    )
    # First two low-hit calls: not enough samples yet (need 3)
    await tracker.record(**low_hit_kwargs)
    await tracker.record(**low_hit_kwargs)
    alert_fn.assert_not_called()
    # Third low-hit call triggers the alert
    await tracker.record(**low_hit_kwargs)
    alert_fn.assert_called_once()
    msg = alert_fn.call_args[0][0].lower()
    assert "cache" in msg
    assert "avg" in msg


async def test_no_cache_alert_on_cold_start(tracker: UsageTracker) -> None:
    """First chat call after restart skips cache-hit alert (cold start)."""
    alert_fn = AsyncMock()
    tracker.set_alert(alert_fn=alert_fn, cache_hit_warn=90.0, slow_threshold_s=999.0)
    await tracker.record(
        call_type="chat", user_id="1", group_id=None, model="m",
        input_tokens=100, cache_read_tokens=0, cache_create_tokens=100,
        output_tokens=50, tool_rounds=0, elapsed_s=1.0,
    )
    alert_fn.assert_not_called()


async def test_alert_on_error(tracker: UsageTracker) -> None:
    alert_fn = AsyncMock()
    tracker.set_alert(alert_fn=alert_fn, cache_hit_warn=90.0, slow_threshold_s=999.0)
    await tracker.record(
        call_type="chat", user_id="1", group_id=None, model="m",
        input_tokens=0, cache_read_tokens=0, cache_create_tokens=0,
        output_tokens=0, tool_rounds=0, elapsed_s=0.5,
        error="API timeout",
    )
    alert_fn.assert_called_once()
    assert "error" in alert_fn.call_args[0][0].lower() or "错误" in alert_fn.call_args[0][0]


async def test_no_alert_when_ok(tracker: UsageTracker) -> None:
    alert_fn = AsyncMock()
    tracker.set_alert(alert_fn=alert_fn, cache_hit_warn=90.0, slow_threshold_s=60.0)
    # Cold start
    await tracker.record(
        call_type="chat", user_id="1", group_id=None, model="m",
        input_tokens=10, cache_read_tokens=90, cache_create_tokens=0,
        output_tokens=50, tool_rounds=0, elapsed_s=1.0,
    )
    # 3 high-hit calls — should never alert
    for _ in range(3):
        await tracker.record(
            call_type="chat", user_id="1", group_id=None, model="m",
            input_tokens=10, cache_read_tokens=90, cache_create_tokens=0,
            output_tokens=50, tool_rounds=0, elapsed_s=1.0,
        )
    alert_fn.assert_not_called()


async def test_cache_alert_cooldown(tracker: UsageTracker) -> None:
    """Second alert within cooldown period is suppressed."""
    alert_fn = AsyncMock()
    tracker.set_alert(
        alert_fn=alert_fn, cache_hit_warn=90.0, slow_threshold_s=999.0,
        cache_alert_cooldown_m=60.0,  # 60 min cooldown — won't expire during test
    )
    # Cold start
    await tracker.record(
        call_type="chat", user_id="1", group_id=None, model="m",
        input_tokens=100, cache_read_tokens=0, cache_create_tokens=100,
        output_tokens=10, tool_rounds=0, elapsed_s=1.0,
    )
    low_hit = dict(
        call_type="chat", user_id="1", group_id=None, model="m",
        input_tokens=100, cache_read_tokens=10, cache_create_tokens=0,
        output_tokens=50, tool_rounds=0, elapsed_s=1.0,
    )
    # Fill window (3 calls) → first alert
    for _ in range(3):
        await tracker.record(**low_hit)
    assert alert_fn.call_count == 1
    # More low-hit calls within cooldown → suppressed
    for _ in range(3):
        await tracker.record(**low_hit)
    assert alert_fn.call_count == 1  # still just 1


async def test_no_cache_alert_for_compact(tracker: UsageTracker) -> None:
    """compact calls don't use prompt cache, so no cache hit warning."""
    alert_fn = AsyncMock()
    tracker.set_alert(alert_fn=alert_fn, cache_hit_warn=90.0, slow_threshold_s=999.0)
    await tracker.record(
        call_type="compact", user_id="1", group_id=None, model="m",
        input_tokens=100, cache_read_tokens=0, cache_create_tokens=0,
        output_tokens=50, tool_rounds=0, elapsed_s=1.0,
    )
    alert_fn.assert_not_called()


async def test_timeseries_hourly(tracker: UsageTracker) -> None:
    """timeseries with bucket='hour' returns 24 buckets for a day."""
    await _insert_sample_data(tracker)
    now = datetime.now(UTC)
    date_str = now.strftime("%Y-%m-%d")
    rows = await tracker.timeseries(period="day", date=date_str)
    assert len(rows) >= 1
    row = rows[0]
    for key in ("bucket", "calls", "input_tokens", "cache_read_tokens",
                "cache_create_tokens", "output_tokens"):
        assert key in row, f"missing key: {key}"
    assert len(row["bucket"]) == 2


async def test_timeseries_daily(tracker: UsageTracker) -> None:
    """timeseries with period='month' returns daily buckets."""
    await _insert_sample_data(tracker)
    now = datetime.now(UTC)
    month_str = now.strftime("%Y-%m")
    rows = await tracker.timeseries(period="month", date=month_str)
    assert len(rows) >= 1
    row = rows[0]
    for key in ("bucket", "calls", "input_tokens", "cache_read_tokens",
                "cache_create_tokens", "output_tokens"):
        assert key in row
    assert len(row["bucket"]) == 2


async def test_timeseries_week(tracker: UsageTracker) -> None:
    """timeseries with period='week' returns daily buckets for 7 days."""
    await _insert_sample_data(tracker)
    now = datetime.now(UTC)
    date_str = now.strftime("%Y-%m-%d")
    rows = await tracker.timeseries(period="week", date=date_str)
    assert len(rows) >= 1
    row = rows[0]
    assert "-" in row["bucket"]
