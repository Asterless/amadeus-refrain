"""Admin control commands wired to the host control service.

Slash commands (admin only for control, /使用帮助 is public):
    /语音启动 / 语音关闭 / 控制状态
    /bot重启 / bot关闭 / bot启动

The host side runs scripts/control_server.py; see [control] in config.toml.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import aiohttp
from loguru import logger
from nonebot import on_command
from nonebot.adapters.onebot.v11 import Bot, Message, MessageEvent
from nonebot.params import CommandArg
from nonebot.permission import SUPERUSER

from src.config_loader import load_config

_tts_start = on_command(
    "语音启动", aliases={"开启语音", "启动语音api"}, permission=SUPERUSER,
    priority=0, block=True,
)
_tts_stop = on_command(
    "语音关闭", aliases={"关闭语音"}, permission=SUPERUSER,
    priority=0, block=True,
)
_music_start = on_command(
    "音乐启动", aliases={"音乐api启动", "网易云启动"}, permission=SUPERUSER,
    priority=0, block=True,
)
_music_stop = on_command(
    "音乐关闭", aliases={"音乐api关闭", "网易云关闭"}, permission=SUPERUSER,
    priority=0, block=True,
)
_bot_restart = on_command(
    "bot重启", aliases={"重启bot", "重启机器人"}, permission=SUPERUSER,
    priority=0, block=True,
)
_bot_stop = on_command(
    "bot关闭", aliases={"关闭bot", "关闭机器人"}, permission=SUPERUSER,
    priority=0, block=True,
)
_bot_start = on_command(
    "bot启动", aliases={"启动bot", "启动机器人"}, permission=SUPERUSER,
    priority=0, block=True,
)
_control_status = on_command(
    "控制状态", aliases={"状态查询"}, permission=SUPERUSER,
    priority=0, block=True,
)
_usage_help = on_command(
    "使用帮助", aliases={"帮助", "help", "怎么用"},
    priority=0, block=True,
)
_config_set = on_command(
    "设置", aliases={"配置", "改配置"}, permission=SUPERUSER,
    priority=0, block=True,
)
_config_view = on_command(
    "查看配置", aliases={"配置详情", "全部配置"}, permission=SUPERUSER,
    priority=0, block=True,
)
_imagegen_quota = on_command(
    "生图额度", aliases={"生图剩余", "图片额度"},
    priority=0, block=True,
)
_imagegen_cancel = on_command(
    "取消生图", aliases={"取消生成", "取消画图"},
    priority=0, block=True,
)


@dataclass(frozen=True)
class SettingSpec:
    name: str
    section: str
    key: str
    kind: str
    hint: str


SETTINGS: dict[str, SettingSpec] = {
    spec.name: spec
    for spec in (
        SettingSpec("表情速率", "sticker", "send_probability", "float01", "表情包发送概率 0~1，如 0.8=80%"),
        SettingSpec("自动收藏表情", "sticker", "auto_collect", "bool", "开/关"),
        SettingSpec("仅收藏表情包", "sticker", "auto_collect_only_stickers", "bool", "开/关"),
        SettingSpec("收藏冷却秒", "sticker", "auto_collect_cooldown_seconds", "int", "秒"),
        SettingSpec("收到表情回复", "sticker", "reply_on_receive", "bool", "开/关"),
        SettingSpec("仅@回复", "group", "at_only", "bool", "开/关"),
        SettingSpec("消息合并条数", "group", "batch_size", "int", "条"),
        SettingSpec("静默等待秒", "group", "debounce_seconds", "float", "秒"),
        SettingSpec("主动冷却秒", "group", "proactive_cooldown_seconds", "float", "秒"),
        SettingSpec("每小时主动上限", "group", "proactive_max_replies_per_hour", "int", "次/小时"),
        SettingSpec("语音最大字数", "tts", "max_chars", "int", "字"),
        SettingSpec("识图开关", "vision", "enabled", "bool", "开/关"),
        SettingSpec("识图模式", "vision", "describe_mode", "describe", "only_at / always"),
        SettingSpec("生图开关", "imagegen", "enabled", "bool", "开/关"),
        SettingSpec("生图每日总限额", "imagegen", "daily_global_limit", "int", "全 bot 每天最多生成张数，0=不限制"),
        SettingSpec("生图单人每日上限", "imagegen", "daily_user_limit", "int", "单个 QQ 用户每天最多张数，0=不限制"),
        SettingSpec("生图群每日上限", "imagegen", "daily_group_limit", "int", "单个群每天最多张数，0=不限制"),
        SettingSpec("生图冷却秒", "imagegen", "cooldown_seconds", "float", "同一用户两次生图最小间隔秒数"),
    )
}


class ControlClient:
    def __init__(self, base_url: str, token: str, timeout_seconds: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    async def status(self) -> dict[str, Any]:
        return await self._request("GET", "/status")

    async def action(self, action: str) -> dict[str, Any]:
        return await self._request("POST", "/control", payload={"action": action})

    async def set_config(self, section: str, key: str, value: Any) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/control",
            payload={"action": "set_config", "section": section, "key": key, "value": value},
        )

    async def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        headers = {"X-Auth-Token": self._token}
        async with (
            aiohttp.ClientSession(timeout=self._timeout, headers=headers) as session,
            session.request(method, f"{self._base_url}{path}", json=payload) as resp,
        ):
            data = await resp.json(content_type=None)
        if not isinstance(data, dict):
            raise RuntimeError(f"unexpected control server response: {data!r}")
        return data


_client: ControlClient | None = None
_client_sig: tuple[str, str, float] | None = None
_background_tasks: set[asyncio.Task[None]] = set()


def _get_client() -> ControlClient | None:
    global _client, _client_sig
    cfg = load_config().control
    if not cfg.enabled or not cfg.token or not cfg.base_url:
        _client = None
        _client_sig = None
        return None
    sig = (cfg.base_url, cfg.token, cfg.timeout_seconds)
    if _client is None or _client_sig != sig:
        _client = ControlClient(*sig)
        _client_sig = sig
    return _client


async def _safe_action(client: ControlClient, action: str) -> None:
    try:
        await client.action(action)
    except Exception:
        logger.warning("control action={} failed", action, exc_info=True)


def _fire_and_forget(client: ControlClient, action: str) -> None:
    task = asyncio.create_task(_safe_action(client, action))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


def _fmt_result(label: str, data: dict[str, Any]) -> str:
    if not data.get("ok"):
        return f"{label} 操作失败：{data.get('error', '未知错误')}"
    inner = data.get("data") or {}
    message = inner.get("message") or inner.get("state") or "完成"
    return f"{label}：{message}"


async def _disabled_reply(bot: Bot, event: MessageEvent) -> None:
    await bot.send(event, "控制功能未启用或未配置 token（config.toml [control]）")


def _parse_value(kind: str, raw: str) -> Any:
    text = raw.strip()
    if kind == "bool":
        if text.lower() in {"开", "on", "true", "yes", "1", "是"}:
            return True
        if text.lower() in {"关", "off", "false", "no", "0", "否"}:
            return False
        raise ValueError(f"无法识别布尔值：{raw}（用 开/关）")
    if kind == "int":
        try:
            return int(text)
        except ValueError:
            raise ValueError(f"请输入整数：{raw}") from None
    if kind == "float":
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"请输入数字：{raw}") from None
    if kind == "float01":
        try:
            value = float(text)
        except ValueError:
            raise ValueError(f"请输入 0~1 的数字：{raw}") from None
        if not 0 <= value <= 1:
            raise ValueError("取值范围 0~1")
        return value
    if kind == "describe":
        if text in {"only_at", "仅@"}:
            return "only_at"
        if text in {"always", "总是"}:
            return "always"
        raise ValueError("识图模式只能填 only_at（仅@）或 always（总是）")
    return text


def _apply_setting(section: str, key: str, value: Any) -> bool:
    import src.plugins.chat as chat

    if section == "sticker":
        if key == "send_probability":
            if chat._sticker_sender is not None:
                chat._sticker_sender._send_probability = value
                return True
        elif key == "auto_collect":
            chat._sticker_auto_collect = value
            return True
        elif key == "auto_collect_only_stickers":
            chat._sticker_auto_collect_only_stickers = value
            return True
        elif key == "auto_collect_cooldown_seconds":
            chat._sticker_auto_collect_cooldown = value
            return True
        elif key == "reply_on_receive":
            chat._scheduler._reply_on_sticker = value
            return True
    elif section == "group":
        setattr(chat._group_config, key, value)
        return True
    elif section == "vision":
        if key == "enabled":
            chat._vision_enabled = value
            return True
        elif key == "describe_mode":
            chat._scheduler._always_describe_images = value == "always"
            return True
    elif section == "tts" and key == "max_chars" and chat._voice_sender is not None:
        chat._voice_sender._max_chars = value
        return True
    elif section == "imagegen":
        imagegen_tools = [
            tool for tool in (chat._imagegen_tool, chat._imagegen_edit_tool) if tool is not None
        ]
        if not imagegen_tools:
            return False
        if key == "enabled":
            for tool in imagegen_tools:
                tool._enabled = bool(value)
            return True
        if key == "daily_global_limit":
            for tool in imagegen_tools:
                tool._daily_global_limit = int(value)
            return True
        if key == "daily_user_limit":
            for tool in imagegen_tools:
                tool._daily_user_limit = int(value)
            return True
        if key == "daily_group_limit":
            for tool in imagegen_tools:
                tool._daily_group_limit = int(value)
            return True
        if key == "cooldown_seconds":
            for tool in imagegen_tools:
                tool._cooldown_seconds = float(value)
            return True
    return False


def _current_value(section: str, key: str) -> Any:
    import src.plugins.chat as chat

    if section == "sticker":
        if key == "send_probability":
            return chat._sticker_sender._send_probability if chat._sticker_sender else None
        if key == "auto_collect":
            return chat._sticker_auto_collect
        if key == "auto_collect_only_stickers":
            return chat._sticker_auto_collect_only_stickers
        if key == "auto_collect_cooldown_seconds":
            return chat._sticker_auto_collect_cooldown
        if key == "reply_on_receive":
            return chat._scheduler._reply_on_sticker
    if section == "group":
        return getattr(chat._group_config, key, None)
    if section == "vision":
        if key == "enabled":
            return chat._vision_enabled
        if key == "describe_mode":
            return "always" if chat._scheduler._always_describe_images else "only_at"
    if section == "tts" and key == "max_chars":
        return chat._voice_sender._max_chars if chat._voice_sender else None
    if section == "imagegen":
        tool = chat._imagegen_tool
        if tool is None:
            return None
        if key == "enabled":
            return tool._enabled
        if key == "daily_global_limit":
            return tool._daily_global_limit
        if key == "daily_user_limit":
            return tool._daily_user_limit
        if key == "daily_group_limit":
            return tool._daily_group_limit
        if key == "cooldown_seconds":
            return tool._cooldown_seconds
    return None


_SENSITIVE_SUFFIXES = ("_key", "token", "secret", "password")


def _redact_value(key: str, value: Any) -> str:
    lowered = key.lower()
    if lowered.endswith(_SENSITIVE_SUFFIXES) or lowered in {"cookie"}:
        return "***"
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return ", ".join(f"{k}: {v}" for k, v in value.items())
    return str(value)


def _flatten_paths(prefix: str, values: Any, out: list[str]) -> None:
    if isinstance(values, dict):
        for k, v in values.items():
            _flatten_paths(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif values not in (None, "", [], {}):
        out.append(f"{prefix} = {_redact_value(prefix, values)}")


def _format_config() -> str:
    data = load_config().model_dump()
    lines: list[str] = []
    _flatten_paths("", data, lines)
    return "\n".join(lines).strip()


def _resolve_path(path: str) -> tuple[str, str, Any] | None:
    """Resolve a dotted config path to (section, key, current value)."""
    parts = path.strip().split(".")
    if len(parts) < 2:
        return None
    node: Any = load_config().model_dump()
    for part in parts:
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, dict):
            try:
                node = node[int(part)]
            except (KeyError, TypeError, ValueError):
                return None
        else:
            return None
    return ".".join(parts[:-1]), parts[-1], node


def _parse_by_type(current: Any, raw: str) -> Any:
    text = raw.strip()
    if isinstance(current, bool):
        return _parse_value("bool", text)
    if isinstance(current, int):
        try:
            return int(text)
        except ValueError:
            raise ValueError(f"请输入整数：{raw}") from None
    if isinstance(current, float):
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"请输入数字：{raw}") from None
    if isinstance(current, list):
        items: list[Any] = []
        for item in [p.strip() for p in text.split(",") if p.strip()]:
            try:
                items.append(int(item))
            except ValueError:
                try:
                    items.append(float(item))
                except ValueError:
                    items.append(item)
        if not items:
            raise ValueError("列表不能为空，用逗号分隔")
        return items
    if isinstance(current, str):
        return text
    raise ValueError(f"不支持修改此类型的配置：{type(current).__name__}")


@_tts_start.handle()
async def handle_tts_start(bot: Bot, event: MessageEvent) -> None:
    client = _get_client()
    if client is None:
        await _disabled_reply(bot, event)
        return
    try:
        data = await client.action("start_tts")
        await bot.send(event, _fmt_result("语音 API", data))
    except Exception:
        logger.warning("control start_tts failed", exc_info=True)
        await bot.send(event, "启动语音 API 失败：无法连接宿主控制服务")


@_tts_stop.handle()
async def handle_tts_stop(bot: Bot, event: MessageEvent) -> None:
    client = _get_client()
    if client is None:
        await _disabled_reply(bot, event)
        return
    try:
        data = await client.action("stop_tts")
        await bot.send(event, _fmt_result("语音 API", data))
    except Exception:
        logger.warning("control stop_tts failed", exc_info=True)
        await bot.send(event, "关闭语音 API 失败：无法连接宿主控制服务")


@_music_start.handle()
async def handle_music_start(bot: Bot, event: MessageEvent) -> None:
    client = _get_client()
    if client is None:
        await _disabled_reply(bot, event)
        return
    try:
        data = await client.action("start_music")
        await bot.send(event, _fmt_result("网易云 API", data))
    except Exception:
        logger.warning("control start_music failed", exc_info=True)
        await bot.send(event, "启动网易云 API 失败：无法连接宿主控制服务")


@_music_stop.handle()
async def handle_music_stop(bot: Bot, event: MessageEvent) -> None:
    client = _get_client()
    if client is None:
        await _disabled_reply(bot, event)
        return
    try:
        data = await client.action("stop_music")
        await bot.send(event, _fmt_result("网易云 API", data))
    except Exception:
        logger.warning("control stop_music failed", exc_info=True)
        await bot.send(event, "关闭网易云 API 失败：无法连接宿主控制服务")


@_bot_restart.handle()
async def handle_bot_restart(bot: Bot, event: MessageEvent) -> None:
    client = _get_client()
    if client is None:
        await _disabled_reply(bot, event)
        return
    await bot.send(event, "收到，正在重启 bot…约 20 秒后恢复")
    _fire_and_forget(client, "restart_bot")


@_bot_stop.handle()
async def handle_bot_stop(bot: Bot, event: MessageEvent) -> None:
    client = _get_client()
    if client is None:
        await _disabled_reply(bot, event)
        return
    await bot.send(event, "收到，正在关闭 bot（NapCat 保持在线，可用宿主脚本重新启动）…")
    _fire_and_forget(client, "stop_bot")


@_bot_start.handle()
async def handle_bot_start(bot: Bot, event: MessageEvent) -> None:
    client = _get_client()
    if client is None:
        await _disabled_reply(bot, event)
        return
    try:
        data = await client.action("start_bot")
        await bot.send(event, _fmt_result("bot", data))
    except Exception:
        logger.warning("control start_bot failed", exc_info=True)
        await bot.send(event, "启动 bot 失败：无法连接宿主控制服务")


@_control_status.handle()
async def handle_status(bot: Bot, event: MessageEvent) -> None:
    client = _get_client()
    if client is None:
        await _disabled_reply(bot, event)
        return
    try:
        data = await client.status()
        if not data.get("ok"):
            await bot.send(event, f"查询失败：{data.get('error', '未知错误')}")
            return
        inner = data.get("data") or {}
        tts = inner.get("tts_api") or {}
        music = inner.get("music_api") or {}
        bot_state = inner.get("bot") or {}
        tts_state = "运行中" if tts.get("running") else "未运行"
        music_state = "运行中" if music.get("running") else "未运行"
        await bot.send(
            event,
            f"语音 API：{tts_state}（端口 {tts.get('port', 9880)}）\n"
            f"网易云 API：{music_state}（端口 {music.get('port', 3000)}）\n"
            f"bot 容器：{bot_state.get('status') or bot_state.get('state') or '未知'}",
        )
    except Exception:
        logger.warning("control status failed", exc_info=True)
        await bot.send(event, "查询失败：无法连接宿主控制服务")


@_config_set.handle()
async def handle_config_set(
    bot: Bot, event: MessageEvent, cmd_arg: Message = CommandArg(),  # noqa: B008
) -> None:
    parts = cmd_arg.extract_plain_text().strip().split()
    if not parts:
        lines = ["当前可调配置（/设置 <名称> <值>；也可以用 /查看配置 里的完整路径，如 /设置 llm.max_tokens 2048）："]
        for spec in SETTINGS.values():
            lines.append(f"· {spec.name}：{_current_value(spec.section, spec.key)}（{spec.hint}）")
        await bot.send(event, "\n".join(lines))
        return
    if len(parts) < 2:
        await bot.send(event, "用法：/设置 <名称或路径> <值>；单独发 /设置 查看可调配置列表")
        return
    name, raw = parts[0], " ".join(parts[1:])
    spec = SETTINGS.get(name)
    if spec is not None:
        try:
            value = _parse_value(spec.kind, raw)
        except ValueError as exc:
            await bot.send(event, f"参数错误：{exc}")
            return
        client = _get_client()
        if client is None:
            await _disabled_reply(bot, event)
            return
        try:
            data = await client.set_config(spec.section, spec.key, value)
            if not data.get("ok"):
                await bot.send(event, f"设置失败：{data.get('error', '未知错误')}")
                return
            _apply_setting(spec.section, spec.key, value)
            await bot.send(event, f"已设置 {spec.name} = {value}，已写入 config.toml 并立即生效")
        except Exception:
            logger.warning("control set_config failed", exc_info=True)
            await bot.send(event, "设置失败：无法连接宿主控制服务")
        return

    resolved = _resolve_path(name)
    if resolved is None:
        await bot.send(event, f"未知配置项：{name}；发 /设置 查看列表，或用 /查看配置 里的完整路径")
        return
    section, key, current = resolved
    try:
        value = _parse_by_type(current, raw)
    except ValueError as exc:
        await bot.send(event, f"参数错误：{exc}")
        return
    client = _get_client()
    if client is None:
        await _disabled_reply(bot, event)
        return
    try:
        data = await client.set_config(section, key, value)
        if not data.get("ok"):
            await bot.send(event, f"设置失败：{data.get('error', '未知错误')}")
            return
        if _apply_setting(section, key, value):
            await bot.send(event, f"已设置 {name} = {value}，已写入 config.toml 并立即生效")
        else:
            await bot.send(event, f"已设置 {name} = {value}，已写入 config.toml，重启 bot 后生效")
    except Exception:
        logger.warning("control set_config failed", exc_info=True)
        await bot.send(event, "设置失败：无法连接宿主控制服务")


@_config_view.handle()
async def handle_config_view(bot: Bot, event: MessageEvent) -> None:
    text = _format_config()
    if not text:
        await bot.send(event, "配置为空")
        return
    chunks = [text[i : i + 1500] for i in range(0, len(text), 1500)]
    for index, chunk in enumerate(chunks):
        prefix = "当前配置（敏感字段已打码）：\n" if index == 0 else ""
        await bot.send(event, prefix + chunk)


@_usage_help.handle()
async def handle_usage_help(bot: Bot, event: MessageEvent) -> None:
    await bot.send(
        event,
        "使用帮助（发送斜杠命令，群内或私聊均可）：\n\n"
        "普通功能：\n"
        "· 聊天：直接 @我 说话即可\n"
        "· 语音：说\"用语音说/念出来 + 内容\"\n"
        "· 生图：说\"画一张/生成图片 + 描述\"，我会先把描述润色成详细画面描述再和你确认，"
        "回复「确认」才生成；想改直接发新描述；回复「取消」或使用 /取消生图 可取消\n"
        "· 图生图：发一张图并说要怎么改（如\"把这张图变成油画风\"），"
        "我会先确认修改方案再出图\n"
        "· 表情包：说\"发个表情包\"或让我收藏/使用表情包\n"
        "· 音乐：说\"搜歌/推荐歌 + 关键词\"\n\n"
        "斜杠命令：\n"
        "· /使用帮助 或 /help：查看本帮助\n"
        "· /语音启动 / /语音关闭：启停语音合成 API\n"
        "· /音乐启动 / /音乐关闭：启停网易云 API\n"
        "· /控制状态：查看语音/网易云 API 与 bot 状态\n"
        "· /设置：查看/修改配置（支持 /查看配置 里的完整路径，如 /设置 group.debounce_seconds 5）\n"
        "· /生图额度：查看今日生图剩余额度与冷却时间\n"
        "· /取消生图：取消当前待确认的文生图或图生图请求\n"
        "· /查看配置：输出当前完整配置（敏感字段打码）\n"
        "· /bot重启 / /bot关闭 / /bot启动：管理 bot 容器（仅管理员）",
    )


@_imagegen_quota.handle()
async def handle_imagegen_quota(bot: Bot, event: MessageEvent) -> None:
    import src.plugins.chat as chat

    tool = chat._imagegen_tool
    if tool is None:
        await bot.send(event, "生图功能未启用（config.toml [imagegen].enabled）")
        return
    user_id = str(getattr(event, "user_id", "") or "")
    group_id = str(getattr(event, "group_id", "") or "") or None
    text = await tool.usage.summary(
        user_id=user_id,
        group_id=group_id,
        global_limit=tool._daily_global_limit,
        user_limit=tool._daily_user_limit,
        group_limit=tool._daily_group_limit,
        cooldown_s=tool._cooldown_seconds,
    )
    await bot.send(event, text)


@_imagegen_cancel.handle()
async def handle_imagegen_cancel(bot: Bot, event: MessageEvent) -> None:
    import src.plugins.chat as chat

    tool = chat._imagegen_tool
    if tool is None:
        await bot.send(event, "生图功能未启用（config.toml [imagegen].enabled）")
        return
    user_id = str(getattr(event, "user_id", "") or "")
    group_id = str(getattr(event, "group_id", "") or "") or None
    if tool.cancel_pending(user_id, group_id):
        await bot.send(event, "已取消生成。")
    else:
        await bot.send(event, "当前没有待确认的生图请求。")
