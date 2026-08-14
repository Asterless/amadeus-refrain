"""NetEase Cloud Music client tests without live network access."""

import json
from pathlib import Path

import httpx

from src.music.client import NeteaseMusicClient


async def _client(tmp_path: Path, handler: httpx.MockTransport) -> NeteaseMusicClient:
    client = NeteaseMusicClient("http://music.test", cookie_file=str(tmp_path / "cookie.json"))
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=handler)
    return client


async def test_search_normalizes_song_results(tmp_path: Path) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/cloudsearch"
        assert request.url.params["keywords"] == "红莲华"
        return httpx.Response(
            200,
            json={
                "code": 200,
                "result": {
                    "songs": [
                        {
                            "id": 123,
                            "name": "红莲华",
                            "ar": [{"name": "LiSA"}],
                            "al": {"name": "LEO-NiNE"},
                        }
                    ]
                },
            },
        )

    client = await _client(tmp_path, httpx.MockTransport(handle))
    rows = await client.search("红莲华")
    await client.close()

    assert rows[0].id == 123
    assert rows[0].label == "红莲华 - LiSA"
    assert rows[0].album == "LEO-NiNE"


async def test_get_song_validates_share_target(tmp_path: Path) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/song/detail"
        assert request.url.params["ids"] == "123"
        return httpx.Response(
            200,
            json={
                "code": 200,
                "songs": [{"id": 123, "name": "红莲华", "ar": [{"name": "LiSA"}], "al": {"name": "专辑"}}],
            },
        )

    client = await _client(tmp_path, httpx.MockTransport(handle))
    track = await client.get_song(123)
    await client.close()
    assert track is not None
    assert track.label == "红莲华 - LiSA"


async def test_qr_login_keeps_key_private_and_persists_cookie(tmp_path: Path) -> None:
    png = "data:image/png;base64,aW1hZ2U="

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/login/qr/key":
            return httpx.Response(200, json={"code": 200, "data": {"unikey": "secret-key"}})
        if request.url.path == "/login/qr/create":
            assert request.url.params["key"] == "secret-key"
            return httpx.Response(200, json={"code": 200, "data": {"qrimg": png}})
        if request.url.path == "/login/qr/check":
            assert request.url.params["key"] == "secret-key"
            return httpx.Response(200, json={"code": 803, "cookie": "MUSIC_U=token123; Path=/;"})
        raise AssertionError(request.url)

    client = await _client(tmp_path, httpx.MockTransport(handle))
    assert await client.login_qr("admin") == png
    assert await client.check_qr("admin") == "logged_in"
    assert await client.check_qr("admin") == "missing"
    await client.close()

    saved = json.loads((tmp_path / "cookie.json").read_text(encoding="utf-8"))
    assert saved["MUSIC_U"] == "token123"


async def test_qr_expiry_clears_pending_login(tmp_path: Path) -> None:
    calls = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        if request.url.path == "/login/qr/key":
            return httpx.Response(200, json={"code": 200, "data": {"unikey": "key"}})
        if request.url.path == "/login/qr/create":
            return httpx.Response(200, json={"code": 200, "data": {"qrimg": "data:image/png;base64,eA=="}})
        calls += 1
        return httpx.Response(200, json={"code": 800})

    client = await _client(tmp_path, httpx.MockTransport(handle))
    await client.login_qr("admin")
    assert await client.check_qr("admin") == "expired"
    assert await client.check_qr("admin") == "missing"
    assert calls == 1
    await client.close()
