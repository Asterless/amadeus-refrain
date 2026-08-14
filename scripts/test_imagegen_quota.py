"""Smoke test for the image generation quota tracker (no API calls)."""

from __future__ import annotations

import asyncio
import pathlib
import tempfile

from src.tools.imagegen_usage import ImageGenQuota


async def main() -> None:
    base = pathlib.Path(tempfile.gettempdir())

    # 1. Cooldown
    f1 = base / "igq_cooldown.json"
    f1.unlink(missing_ok=True)
    q1 = ImageGenQuota(str(f1))
    await q1.record(user_id="u1", group_id="g1")
    ok, reason = await q1.check(
        user_id="u1", group_id="g1",
        global_limit=10, user_limit=10, group_limit=10, cooldown_s=60,
    )
    assert not ok and reason
    ok, _ = await q1.check(
        user_id="u1", group_id="g1",
        global_limit=10, user_limit=10, group_limit=10, cooldown_s=0,
    )
    assert ok

    # 2. Per-user daily limit
    f2 = base / "igq_user.json"
    f2.unlink(missing_ok=True)
    q2 = ImageGenQuota(str(f2))
    await q2.record(user_id="u1", group_id="g9")
    await q2.record(user_id="u1", group_id="g9")
    ok, reason = await q2.check(
        user_id="u1", group_id="g9",
        global_limit=10, user_limit=2, group_limit=10, cooldown_s=0,
    )
    assert not ok and reason

    # 3. Per-group daily limit
    f3 = base / "igq_group.json"
    f3.unlink(missing_ok=True)
    q3 = ImageGenQuota(str(f3))
    await q3.record(user_id="u1", group_id="g1")
    await q3.record(user_id="u2", group_id="g1")
    ok, reason = await q3.check(
        user_id="u3", group_id="g1",
        global_limit=10, user_limit=10, group_limit=2, cooldown_s=0,
    )
    assert not ok and reason

    # 4. Global daily limit
    f4 = base / "igq_global.json"
    f4.unlink(missing_ok=True)
    q4 = ImageGenQuota(str(f4))
    await q4.record(user_id="u1", group_id="g1")
    await q4.record(user_id="u2", group_id="g2")
    ok, reason = await q4.check(
        user_id="u9", group_id="g9",
        global_limit=2, user_limit=10, group_limit=10, cooldown_s=0,
    )
    assert not ok and reason

    # 5. Persistence across instances + summary + reset
    f5 = base / "igq_persist.json"
    f5.unlink(missing_ok=True)
    q5a = ImageGenQuota(str(f5))
    await q5a.record(user_id="u1", group_id="g1")
    await q5a.record(user_id="u1", group_id="g1")
    q5b = ImageGenQuota(str(f5))
    summary = await q5b.summary(
        user_id="u1", group_id="g1",
        global_limit=20, user_limit=5, group_limit=15, cooldown_s=0,
    )
    print(summary)
    assert "2/20" in summary and "2/5" in summary and "2/15" in summary
    q5b.reset_today()
    ok, _ = await q5b.check(
        user_id="u1", group_id="g1",
        global_limit=20, user_limit=5, group_limit=15, cooldown_s=0,
    )
    assert ok

    for f in (f1, f2, f3, f4, f5):
        f.unlink(missing_ok=True)
    print("ALL QUOTA TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
