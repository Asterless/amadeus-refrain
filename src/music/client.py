"""Small async client for a NeteaseCloudMusicApi-compatible service."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel


class MusicTrack(BaseModel):
    id: int
    name: str
    artists: str = ""
    album: str = ""
    url: str = ""

    @property
    def label(self) -> str:
        suffix = f" - {self.artists}" if self.artists else ""
        return f"{self.name}{suffix}"


class NeteaseMusicClient:
    def __init__(
        self,
        base_url: str,
        *,
        cookie_file: str = "storage/netease_cookie.json",
        timeout: float = 15.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.cookie_file = Path(cookie_file)
        self.client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)
        self._pending_login: dict[str, str] = {}
        self._service_process: asyncio.subprocess.Process | None = None
        self._load_cookies()

    def _load_cookies(self) -> None:
        try:
            values = json.loads(self.cookie_file.read_text(encoding="utf-8"))
            if isinstance(values, dict):
                self.client.cookies.update({str(k): str(v) for k, v in values.items()})
        except (OSError, ValueError, TypeError):
            return

    def _save_cookies(self) -> None:
        self.cookie_file.parent.mkdir(parents=True, exist_ok=True)
        values = {cookie.name: cookie.value for cookie in self.client.cookies.jar}
        self.cookie_file.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")

    async def close(self) -> None:
        await self.client.aclose()
        if self._service_process is not None and self._service_process.returncode is None:
            self._service_process.terminate()
            try:
                await asyncio.wait_for(self._service_process.wait(), timeout=5)
            except TimeoutError:
                self._service_process.kill()
                await self._service_process.wait()
        self._service_process = None

    async def start_local_service(self, service_app: str, *, node_executable: str = "node") -> bool:
        """Start a configured local API as a loopback-only child process."""
        parsed = urlparse(self.base_url)
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("自动启动仅允许 loopback API 地址")
        try:
            response = await self.client.get(f"{self.base_url}/", timeout=2)
            if response.status_code < 500:
                return True
        except httpx.HTTPError:
            pass

        app = Path(service_app).resolve()
        if not app.is_file():
            raise FileNotFoundError(f"网易云 API 入口不存在: {app}")
        env = os.environ.copy()
        env["HOST"] = "127.0.0.1"
        env["PORT"] = str(parsed.port or 3000)
        self._service_process = await asyncio.create_subprocess_exec(
            node_executable,
            str(app),
            cwd=str(app.parent),
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        for _ in range(40):
            if self._service_process.returncode is not None:
                return False
            try:
                response = await self.client.get(f"{self.base_url}/", timeout=1)
                if response.status_code < 500:
                    return True
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.5)
        return False

    async def _get(self, path: str, **params: Any) -> dict[str, Any]:
        response = await self.client.get(f"{self.base_url}{path}", params=params)
        response.raise_for_status()
        payload: Any = response.json()
        if not isinstance(payload, dict) or int(payload.get("code", 200)) not in (200, 201):
            message = (
                str(payload.get("msg", "网易云接口返回错误"))
                if isinstance(payload, dict)
                else "网易云接口返回错误"
            )
            raise RuntimeError(message)
        return payload

    async def search(self, keyword: str, limit: int = 8) -> list[MusicTrack]:
        payload = await self._get("/cloudsearch", keywords=keyword[:80], limit=min(max(limit, 1), 20), type=1)
        rows = payload.get("result", {}).get("songs", [])
        result: list[MusicTrack] = []
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict) or not row.get("id"):
                continue
            artists = ", ".join(str(a.get("name", "")) for a in row.get("ar", []) if isinstance(a, dict))
            album = str((row.get("al") or {}).get("name", ""))
            result.append(MusicTrack(id=int(row["id"]), name=str(row.get("name", "")), artists=artists, album=album))
        return result

    async def get_song(self, song_id: int) -> MusicTrack | None:
        payload = await self._get("/song/detail", ids=str(song_id))
        rows = payload.get("songs", [])
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
            return None
        row = rows[0]
        artists = ", ".join(str(a.get("name", "")) for a in row.get("ar", []) if isinstance(a, dict))
        album = str((row.get("al") or {}).get("name", ""))
        return MusicTrack(
            id=int(row["id"]),
            name=str(row.get("name", "")),
            artists=artists,
            album=album,
        )

    async def login_qr(self, session_key: str) -> str:
        timestamp = int(time.time() * 1000)
        key_payload = await self._get("/login/qr/key", timestamp=timestamp)
        key = str(key_payload.get("data", {}).get("unikey", ""))
        if not key:
            raise RuntimeError("网易云未返回二维码 key")
        qr_payload = await self._get("/login/qr/create", key=key, qrimg="true", timestamp=timestamp)
        qrimg = str(qr_payload.get("data", {}).get("qrimg", ""))
        if not qrimg:
            raise RuntimeError("网易云未返回二维码图片")
        self._pending_login[session_key] = key
        return qrimg

    async def check_qr(self, session_key: str) -> str:
        key = self._pending_login.get(session_key)
        if not key:
            return "missing"
        response = await self.client.get(
            f"{self.base_url}/login/qr/check",
            params={"key": key, "timestamp": int(time.time() * 1000)},
        )
        response.raise_for_status()
        payload: Any = response.json()
        code = int(payload.get("code", 0)) if isinstance(payload, dict) else 0
        if code == 803:
            raw_cookie = str(payload.get("cookie", ""))
            parsed = SimpleCookie()
            parsed.load(raw_cookie)
            for name, morsel in parsed.items():
                self.client.cookies.set(name, morsel.value)
            self._save_cookies()
            self._pending_login.pop(session_key, None)
            return "logged_in"
        if code == 800:
            self._pending_login.pop(session_key, None)
        return {800: "expired", 801: "waiting", 802: "scanned"}.get(code, "waiting")

    async def is_logged_in(self) -> bool:
        try:
            payload = await self._get("/login/status", timestamp=int(time.time() * 1000))
            return int(payload.get("data", {}).get("account", {}).get("id", 0)) > 0
        except (httpx.HTTPError, RuntimeError, TypeError, ValueError):
            return False

    @staticmethod
    def qr_data_url(qrimg: str) -> str:
        if qrimg.startswith("data:image"):
            return qrimg
        return f"data:image/png;base64,{base64.b64encode(qrimg.encode()).decode()}"

    @staticmethod
    def qr_bytes(qrimg: str) -> bytes:
        if "," not in qrimg:
            raise ValueError("无效二维码图片")
        return base64.b64decode(qrimg.split(",", 1)[1], validate=True)
