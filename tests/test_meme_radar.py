"""Trend provider and background radar tests."""

from typing import Any

from src.meme.radar import MemeRadar, UapiTrendProvider
from src.meme.store import MemeStore, TrendItem


class _FakeProvider:
    def __init__(self) -> None:
        self.closed = False

    async def fetch(self, platform: str, limit: int) -> list[TrendItem]:
        if platform == "broken":
            raise RuntimeError("upstream unavailable")
        return [TrendItem(platform=platform, title=f"{platform}热点", rank=1)]

    async def close(self) -> None:
        self.closed = True


async def test_radar_keeps_partial_success_and_invalidates_prompt(tmp_path) -> None:
    store = MemeStore(str(tmp_path / "memes.json"))
    provider = _FakeProvider()
    changes = 0

    def on_change() -> None:
        nonlocal changes
        changes += 1

    radar = MemeRadar(
        store,
        provider,
        platforms=["weibo", "broken", "bilibili"],
        on_change=on_change,
    )
    fetched = await radar.refresh_once()

    assert fetched == 2
    assert store.count == 2
    assert changes == 1
    await radar.refresh_once()
    assert changes == 1
    await radar.stop()
    assert provider.closed is True


async def test_uapi_provider_normalizes_response(monkeypatch) -> None:
    provider = UapiTrendProvider("https://example.com/hotboard")

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "list": [
                    {"index": 2, "title": "测试热点", "url": "https://example.com/1", "hot_value": 123},
                ]
            }

    async def fake_get(*args, **kwargs) -> _Response:
        return _Response()

    monkeypatch.setattr(provider._client, "get", fake_get)
    rows = await provider.fetch("weibo", 10)
    await provider.close()

    assert rows[0].platform == "weibo"
    assert rows[0].rank == 2
    assert rows[0].hot_value == "123"


async def test_uapi_provider_rejects_invalid_payload(monkeypatch) -> None:
    provider = UapiTrendProvider("https://example.com/hotboard")

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"unexpected": True}

    async def fake_get(*args, **kwargs) -> _Response:
        return _Response()

    monkeypatch.setattr(provider._client, "get", fake_get)
    try:
        try:
            await provider.fetch("weibo", 10)
        except ValueError as exc:
            assert "does not contain a list" in str(exc)
        else:
            raise AssertionError("invalid payload should fail")
    finally:
        await provider.close()
