"""Host-side control service for the QQ bot.

Lets the bot (running in Docker) start/stop the GPT-SoVITS TTS API and the
bot container itself. Run on the host with:

    powershell -ExecutionPolicy Bypass -File scripts/start_control_server.ps1

or manually:

    .venv\\Scripts\\python.exe scripts/control_server.py

Endpoints:
    GET  /status                -> {"ok": true, "data": {tts_api, bot}}
    POST /control               -> {"action": "start_tts" | "stop_tts" |
                                     "start_bot" | "stop_bot" | "restart_bot" |
                                     "set_config"} with "section", "key", "value"

Auth: every request must carry header "X-Auth-Token: <token>", where <token>
is [control].token from config.toml. Bind address defaults to 0.0.0.0 so the
Docker bot container can reach it via host.docker.internal.
"""

from __future__ import annotations

import json
import logging
import re
import socket
import subprocess
import sys
import threading
import tomllib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_DIR / "config.toml"
LOGS_DIR = PROJECT_DIR / "storage" / "logs"
COMPOSE_FILE = PROJECT_DIR / "docker-compose.yml"

GPT_ROOT = Path(r"D:\GPT-SoVITS\GPT-SoVITS-v4-20250529\GPT-SoVITS-v4-20250529")
GPT_PYTHON = GPT_ROOT / "runtime" / "python.exe"
TTS_API_REL_CONFIG = "GPT_SoVITS/configs/tts_infer_shigeju.yaml"
TTS_PORT = 9880

ACTIONS = {
    "start_tts", "stop_tts",
    "start_music", "stop_music",
    "start_bot", "stop_bot", "restart_bot",
    "set_config",
}

_log = logging.getLogger("control_server")
_action_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_control_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise RuntimeError(f"config.toml not found: {CONFIG_PATH}")
    with CONFIG_PATH.open("rb") as fh:
        data = tomllib.load(fh)
    return data.get("control", {})


def _load_music_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise RuntimeError(f"config.toml not found: {CONFIG_PATH}")
    with CONFIG_PATH.open("rb") as fh:
        data = tomllib.load(fh)
    return data.get("music", {})


# ---------------------------------------------------------------------------
# TTS API control
# ---------------------------------------------------------------------------

def _port_listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _listener_pid(port: int) -> int | None:
    try:
        out = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15,
        ).stdout
    except Exception:
        return None
    for line in out.splitlines():
        if f":{port}" in line and "LISTENING" in line:
            parts = line.split()
            if parts:
                try:
                    return int(parts[-1])
                except ValueError:
                    continue
    return None


def _pid_is_tts_api(pid: int) -> bool:
    try:
        out = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        return "api_v2.py" in (out.stdout or "")
    except Exception:
        _log.warning("could not inspect pid=%s command line, assuming it is the API", pid)
        return True


def start_tts() -> dict[str, Any]:
    if _port_listening(TTS_PORT):
        return {"state": "already_running", "message": "已在运行（9880 端口监听中）"}
    if not GPT_PYTHON.exists():
        return {"state": "error", "message": f"GPT-SoVITS runtime python not found: {GPT_PYTHON}"}
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with (
        open(LOGS_DIR / "tts_api.stdout.log", "ab") as stdout,
        open(LOGS_DIR / "tts_api.stderr.log", "ab") as stderr,
    ):
        proc = subprocess.Popen(
            [
                str(GPT_PYTHON),
                "api_v2.py",
                "-a",
                "0.0.0.0",
                "-p",
                str(TTS_PORT),
                "-c",
                TTS_API_REL_CONFIG,
            ],
            cwd=str(GPT_ROOT),
            stdout=stdout,
            stderr=stderr,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    return {
        "state": "starting",
        "pid": proc.pid,
        "message": "正在启动，模型加载约需 1-2 分钟",
    }


def stop_tts() -> dict[str, Any]:
    pid = _listener_pid(TTS_PORT)
    if pid is None:
        return {"state": "not_running", "message": "未在运行"}
    if not _pid_is_tts_api(pid):
        return {
            "state": "error",
            "message": f"端口 {TTS_PORT} 的进程 {pid} 不是 GPT-SoVITS API，拒绝结束",
        }
    subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, timeout=30)
    return {"state": "stopped", "message": "已关闭"}


# ---------------------------------------------------------------------------
# NetEase Cloud Music API control
# ---------------------------------------------------------------------------

def _music_port() -> int:
    cfg = _load_music_config()
    try:
        return urlparse(str(cfg.get("api_base_url") or "")).port or 3000
    except ValueError:
        return 3000


def _pid_is_music_api(pid: int) -> bool:
    try:
        out = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        command = out.stdout or ""
        return "NeteaseCloudMusicApi" in command or "app.js" in command
    except Exception:
        _log.warning("could not inspect pid=%s command line, assuming it is the music API", pid)
        return True


def start_music() -> dict[str, Any]:
    port = _music_port()
    if _port_listening(port):
        return {"state": "already_running", "message": f"已在运行（{port} 端口监听中）"}
    cfg = _load_music_config()
    service_app = str(cfg.get("service_app") or "").strip()
    node_executable = str(cfg.get("node_executable") or "node").strip()
    if not service_app:
        return {"state": "error", "message": "[music].service_app 未配置"}
    app = Path(service_app).resolve()
    if not app.is_file():
        return {"state": "error", "message": f"网易云 API 入口不存在: {app}"}
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    env = __import__("os").environ.copy()
    env["HOST"] = "0.0.0.0"
    env["PORT"] = str(port)
    with (
        open(LOGS_DIR / "music_api.stdout.log", "ab") as stdout,
        open(LOGS_DIR / "music_api.stderr.log", "ab") as stderr,
    ):
        proc = subprocess.Popen(
            [node_executable, str(app)],
            cwd=str(app.parent),
            env=env,
            stdout=stdout,
            stderr=stderr,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    return {"state": "starting", "pid": proc.pid, "message": f"正在启动（端口 {port}）"}


def stop_music() -> dict[str, Any]:
    port = _music_port()
    pid = _listener_pid(port)
    if pid is None:
        return {"state": "not_running", "message": "未在运行"}
    if not _pid_is_music_api(pid):
        return {
            "state": "error",
            "message": f"端口 {port} 的进程 {pid} 不是网易云 API，拒绝结束",
        }
    subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, timeout=30)
    return {"state": "stopped", "message": "已关闭"}


# ---------------------------------------------------------------------------
# Bot container control
# ---------------------------------------------------------------------------

def _docker_compose(action: str) -> dict[str, Any]:
    """Run `docker compose <action> bot` and return a short summary."""
    try:
        proc = subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FILE), action, "bot"],
            cwd=str(PROJECT_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
    except FileNotFoundError:
        return {"state": "error", "message": "docker 命令不存在，请确认 docker 已安装且在 PATH 中"}
    except subprocess.TimeoutExpired:
        return {"state": "error", "message": f"docker compose {action} 超时"}
    combined = (proc.stdout or "") + (proc.stderr or "")
    tail = [line for line in combined.splitlines() if line.strip()][-3:]
    ok = proc.returncode == 0
    return {
        "state": action,
        "ok": ok,
        "message": "；".join(tail) if tail else ("ok" if ok else "failed"),
    }


def _bot_status() -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FILE), "ps", "bot", "--format", "json"],
            cwd=str(PROJECT_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        rows = [json.loads(line) for line in proc.stdout.splitlines() if line.strip().startswith("{")]
        if rows:
            row = rows[0]
            return {
                "state": row.get("State", "unknown"),
                "status": row.get("Status", ""),
                "name": row.get("Name", "qq-bot"),
            }
        return {"state": "unknown", "status": (proc.stdout + proc.stderr).strip()[:200]}
    except Exception as exc:
        return {"state": "error", "status": str(exc)[:200]}


def status() -> dict[str, Any]:
    return {
        "tts_api": {"running": _port_listening(TTS_PORT), "port": TTS_PORT},
        "music_api": {"running": _port_listening(_music_port()), "port": _music_port()},
        "bot": _bot_status(),
    }


def _toml_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    raise ValueError(f"unsupported value type: {type(value).__name__}")


def set_config(section: str, key: str, value: Any) -> dict[str, Any]:
    """Persist a config value by rewriting the matching line in config.toml."""
    if not CONFIG_PATH.exists():
        return {"state": "error", "message": "config.toml not found"}
    try:
        literal = _toml_literal(value)
    except ValueError as exc:
        return {"state": "error", "message": str(exc)}
    lines = CONFIG_PATH.read_text(encoding="utf-8").splitlines()
    current: str | None = None
    target: int | None = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped[1:-1].strip()
            continue
        if current == section and re.match(rf"^\s*{re.escape(key)}\s*=", line):
            target = i
            break
    if target is None:
        insert_at: int | None = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if (
                stripped.startswith("[")
                and stripped.endswith("]")
                and stripped[1:-1].strip() == section
            ):
                insert_at = i + 1
                break
        if insert_at is None:
            return {"state": "error", "message": f"config section not found: {section}"}
        lines.insert(insert_at, f"{key} = {literal}")
        message = f"{section}.{key} = {literal}（新增）"
    else:
        lines[target] = f"{key} = {literal}"
        message = f"{section}.{key} = {literal}"
    CONFIG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"state": "ok", "message": message}


def run_action(action: str, **kwargs: Any) -> dict[str, Any]:
    with _action_lock:
        if action == "start_tts":
            return start_tts()
        if action == "stop_tts":
            return stop_tts()
        if action == "start_music":
            return start_music()
        if action == "stop_music":
            return stop_music()
        if action in {"start_bot", "stop_bot", "restart_bot"}:
            return _docker_compose(action.removesuffix("_bot"))
        if action == "set_config":
            return set_config(
                str(kwargs.get("section") or "").strip(),
                str(kwargs.get("key") or "").strip(),
                kwargs.get("value"),
            )
        return status()


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    server_version = "AmadeusControl/1.0"

    def log_message(self, format: str, *args: object) -> None:
        _log.info("%s - %s", self.client_address[0], format % args)

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        token: str = getattr(self.server, "token", "")  # type: ignore[attr-defined]
        return bool(token) and self.headers.get("X-Auth-Token") == token

    def _path(self) -> str:
        return self.path.split("?", 1)[0]

    def do_GET(self) -> None:
        if self._path() != "/status":
            self._send_json(404, {"ok": False, "error": "not found"})
            return
        if not self._authorized():
            self._send_json(401, {"ok": False, "error": "unauthorized"})
            return
        try:
            with _action_lock:
                payload = status()
        except Exception as exc:  # pragma: no cover - defensive
            _log.exception("status failed")
            self._send_json(500, {"ok": False, "error": str(exc)})
            return
        self._send_json(200, {"ok": True, "data": payload})

    def do_POST(self) -> None:
        if self._path() != "/control":
            self._send_json(404, {"ok": False, "error": "not found"})
            return
        if not self._authorized():
            self._send_json(401, {"ok": False, "error": "unauthorized"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            body = json.loads(raw or b"{}")
            action = str(body.get("action") or "").strip()
        except Exception as exc:
            self._send_json(400, {"ok": False, "error": f"bad request: {exc}"})
            return
        if action not in ACTIONS:
            self._send_json(400, {"ok": False, "error": f"unknown action: {action!r}"})
            return
        extra: dict[str, Any] = {}
        if action == "set_config":
            section = str(body.get("section") or "").strip()
            key = str(body.get("key") or "").strip()
            if not section or not key:
                self._send_json(400, {"ok": False, "error": "set_config requires section and key"})
                return
            extra = {"section": section, "key": key, "value": body.get("value")}
        _log.info("action=%s from=%s", action, self.client_address[0])
        try:
            result = run_action(action, **extra)
        except Exception as exc:  # pragma: no cover - defensive
            _log.exception("action=%s failed", action)
            self._send_json(500, {"ok": False, "error": str(exc)})
            return
        self._send_json(200, {"ok": True, "data": result})


def main() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOGS_DIR / "control_server.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    try:
        control = _load_control_config()
    except Exception as exc:
        _log.error("cannot load [control] from config.toml: %s", exc)
        sys.exit(2)

    token = str(control.get("token") or "").strip()
    if not token:
        _log.error("[control].token is empty in config.toml; refusing to start")
        sys.exit(2)

    port = int(control.get("port") or 8765)
    server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    server.token = token  # type: ignore[attr-defined]
    _log.info("control server listening on 0.0.0.0:%s (auth enabled)", port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
