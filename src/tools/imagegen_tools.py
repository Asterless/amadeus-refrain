"""Image generation tool backed by Zhipu BigModel CogView (OpenAI-compatible)."""

from __future__ import annotations

import base64
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import aiohttp
from loguru import logger

from src.tools.base import Tool
from src.tools.context import ToolContext
from src.tools.imagegen_usage import ImageGenQuota, ImageGenReservation

_CONFIRM_TIMEOUT_S = 120.0
_CONFIRM_RE = re.compile(
    r"^(?:确认|同意|可以|好的|好|行|ok|okay|没问题|生成|生成吧|来|发吧|就这个|就这张|嗯+|是|对|就这样|就这么办|开始|开始生成|可以了|妥)[!！。.~～\s]*$",
    re.IGNORECASE,
)
_CANCEL_RE = re.compile(
    r"^(?:/?(?:取消生图|取消生成|取消画图)|取消|算了|不用了|不要了|不生成了?|撤销)[!！。.~～\s]*$"
)
_REVISION_RE = re.compile(
    r"^(?:修改|调整|改成|改为|换成|不要|去掉|删除|增加|添加|加上|再加|补上|"
    r"把.+?(?:改成|改为|换成|去掉|删除))[：:\s]?",
)


@dataclass
class _PendingPrompt:
    prompt: str
    created_at: float
    user_id: str
    group_id: str | None
    image_path: str | None = None  # None = 文生图；有值 = 图生图
    revisions: tuple[str, ...] = ()
    confirmation_message_id: int | None = None

    def effective_prompt(self) -> str:
        if not self.revisions:
            return self.prompt
        revision_lines = "\n".join(
            f"{index}. {revision}" for index, revision in enumerate(self.revisions, start=1)
        )
        return (
            f"{self.prompt}\n"
            "在保留上述未被修改内容的基础上，按顺序应用以下修改；如有冲突，以后面的要求为准：\n"
            f"{revision_lines}"
        )


# (user_id, group_id) -> pending confirmation request
_pending_prompts: dict[tuple[str, str | None], _PendingPrompt] = {}


class GenerateImageTool(Tool):
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        size: str,
        timeout_seconds: float,
        max_prompt_chars: int,
        proxy: str = "",
        daily_global_limit: int = 0,
        daily_user_limit: int = 0,
        daily_group_limit: int = 0,
        cooldown_seconds: float = 0.0,
        usage_file: str = "storage/imagegen_usage.json",
        usage: ImageGenQuota | None = None,
        prompt_rewriter: Callable[[str, str], Awaitable[str]] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._size = size
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._max_prompt_chars = max_prompt_chars
        self._proxy = proxy
        self._enabled = True
        self._daily_global_limit = daily_global_limit
        self._daily_user_limit = daily_user_limit
        self._daily_group_limit = daily_group_limit
        self._cooldown_seconds = cooldown_seconds
        self.usage = usage if usage is not None else ImageGenQuota(usage_file)
        self._prompt_rewriter = prompt_rewriter

    def set_prompt_rewriter(self, rewriter: Callable[[str, str], Awaitable[str]]) -> None:
        self._prompt_rewriter = rewriter

    @property
    def name(self) -> str:
        return "generate_image"

    @property
    def description(self) -> str:
        return (
            "根据用户的生图需求生成一张图片并发送到当前群或私聊。"
            "仅在用户明确要求画画、生成图片/表情图/立绘/壁纸/海报等时调用。"
            "调用时必须先把用户的需求润色成高质量的画面描述（prompt），润色规则：\n"
            "1. 结构：场景/背景 → 主体与动作 → 细节 → 约束，写成一段通顺自然的中文，不要用标签或列表；\n"
            "2. 用户描述已经具体 → 只整理结构、保留全部指定元素，不要凭空添加内容；\n"
            "3. 用户描述太简单 → 可以补充风格/媒介、构图/视角、光线/氛围、配色、质感等，"
            "但只补能实质提升出图质量的部分；\n"
            "4. 绝不添加用户未要求的人物、物体、品牌、标语或剧情；\n"
            "5. 画面需要出现文字时，明确写出文字内容，并说明字体风格和位置；\n"
            "6. 长度控制在 30~120 字。\n"
            "注意：调用后不会立刻出图，会先把润色后的描述发给用户确认，"
            "用户回复「确认」后才真正生成。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": f"画面描述，最大 {self._max_prompt_chars} 字",
                },
            },
            "required": ["prompt"],
        }

    async def execute(self, ctx: ToolContext, **kwargs: Any) -> str:
        if not self._enabled:
            return "生图功能已关闭（管理员可用 /设置 生图开关 开 重新开启）"
        prompt = str(kwargs.get("prompt") or "").strip()
        if not prompt:
            return "没有可生成画面的描述。"
        if len(prompt) > self._max_prompt_chars:
            return f"描述过长，最大 {self._max_prompt_chars} 字。"
        if not ctx.bot or (not ctx.group_id and not ctx.user_id):
            return "没有可发送图片的聊天目标。"
        if not self._api_key:
            return "未配置生图 API key（管理员用 /设置 imagegen.api_key <key> 填写后重启生效）"

        user_id = str(ctx.user_id or "")
        group_id = str(ctx.group_id) if ctx.group_id else None
        allowed, reason = await self.usage.check(
            user_id=user_id,
            group_id=group_id,
            global_limit=self._daily_global_limit,
            user_limit=self._daily_user_limit,
            group_limit=self._daily_group_limit,
            cooldown_s=self._cooldown_seconds,
        )
        if not allowed:
            logger.info("imagegen quota blocked | user={} group={} reason={!r}", user_id, group_id, reason)
            return reason

        return await self._request_confirmation(ctx, prompt, user_id, group_id)

    def has_pending(self, user_id: str, group_id: str | None) -> bool:
        key, pending = self._find_pending(user_id, group_id)
        if pending is None:
            return False
        if time.time() - pending.created_at > _CONFIRM_TIMEOUT_S:
            self._remove_pending(key)
            return False
        return True

    def can_handle_confirmation(
        self,
        user_id: str,
        group_id: str | None,
        text: str,
        reply_message_id: int | None = None,
    ) -> bool:
        """Return whether a message is unambiguously part of confirmation flow."""
        key, pending = self._find_pending(user_id, group_id)
        if pending is None:
            return False
        if time.time() - pending.created_at > _CONFIRM_TIMEOUT_S:
            self._remove_pending(key)
            return False
        if _CONFIRM_RE.match(text) or _CANCEL_RE.match(text) or _REVISION_RE.match(text):
            return True
        return bool(
            reply_message_id is not None
            and pending.confirmation_message_id is not None
            and reply_message_id == pending.confirmation_message_id
        )

    def cancel_pending(self, user_id: str, group_id: str | None) -> bool:
        """Cancel a pending request for this user or a proactive request in the group."""
        key, pending = self._find_pending(user_id, group_id)
        if pending is None:
            return False
        self._remove_pending(key)
        return True

    async def try_confirm(
        self,
        *,
        bot: Any,
        user_id: str,
        group_id: str | None,
        text: str,
        reply_message_id: int | None = None,
    ) -> bool:
        """Handle the user's reply to a pending confirmation. Returns True if consumed."""
        key, pending = self._find_pending(user_id, group_id)
        if pending is None:
            return False
        now = time.time()
        if now - pending.created_at > _CONFIRM_TIMEOUT_S:
            self._remove_pending(key)
            await self._send_text(bot, user_id, group_id, "确认请求已过期，想生成的话重新说「画一张…」就好。")
            return True

        if _CANCEL_RE.match(text):
            self._remove_pending(key)
            await self._send_text(bot, user_id, group_id, "已取消生成。")
            return True

        if _CONFIRM_RE.match(text):
            effective_prompt = pending.effective_prompt()
            self._remove_pending(key)
            reservation, reason = await self.usage.reserve(
                user_id=user_id,
                group_id=group_id,
                global_limit=self._daily_global_limit,
                user_limit=self._daily_user_limit,
                group_limit=self._daily_group_limit,
                cooldown_s=self._cooldown_seconds,
            )
            if reservation is None:
                await self._send_text(bot, user_id, group_id, reason)
                return True
            if pending.image_path:
                await self._edit_and_send(
                    bot, effective_prompt, pending.image_path, user_id, group_id, reservation,
                )
            else:
                await self._generate_and_send(bot, effective_prompt, user_id, group_id, reservation)
            return True

        if not self.can_handle_confirmation(user_id, group_id, text, reply_message_id):
            return False

        if not text:
            # 只发了表情/图片没有文字：重新提醒一次
            await self._send_text(
                bot, user_id, group_id,
                f"还没收到确认。回复「确认」开始生成，回复「取消」或使用「/取消生图」取消，"
                f"也可以继续发修改要求：\n{pending.effective_prompt()}",
            )
            return True

        if self._prompt_rewriter is None:
            await self._send_text(bot, user_id, group_id, "暂时无法合并修改，请稍后再试或取消后重新描述。")
            return True
        try:
            effective_prompt = (await self._prompt_rewriter(pending.effective_prompt(), text)).strip()
        except Exception:
            logger.warning("imagegen prompt rewrite failed", exc_info=True)
            await self._send_text(bot, user_id, group_id, "修改合并失败，原描述已保留，请稍后再试。")
            return True
        if not effective_prompt:
            await self._send_text(bot, user_id, group_id, "修改合并失败，原描述已保留，请换种说法。")
            return True

        updated = _PendingPrompt(
            prompt=effective_prompt,
            created_at=now,
            user_id=pending.user_id,
            group_id=pending.group_id,
            image_path=pending.image_path,
            confirmation_message_id=pending.confirmation_message_id,
        )
        if len(effective_prompt) > self._max_prompt_chars:
            await self._send_text(
                bot, user_id, group_id,
                f"累计后的画面描述过长（{len(effective_prompt)}/{self._max_prompt_chars} 字）。"
                "请把这次修改说得更精简，或先取消后重新描述。",
            )
            return True
        self._remove_pending(key)
        _pending_prompts[key] = updated
        await self._send_text(
            bot, user_id, group_id,
            f"已在原画面描述上追加修改，当前完整要求为：\n{effective_prompt}\n\n"
            "回复「确认」开始生成；回复「取消」或使用"
            "「/取消生图」取消；还要改就继续发修改要求。",
        )
        return True

    def _find_pending(
        self, user_id: str, group_id: str | None,
    ) -> tuple[tuple[str, str | None], _PendingPrompt | None]:
        key = (user_id, group_id)
        pending = _pending_prompts.get(key)
        if pending is None and user_id:
            # 主动插话触发的生图没有指定用户，群内任何成员都可以确认
            key = ("", group_id)
            pending = _pending_prompts.get(key)
        return key, pending

    def _lookup_pending(self, user_id: str, group_id: str | None) -> _PendingPrompt | None:
        return self._find_pending(user_id, group_id)[1]

    @staticmethod
    def _remove_pending(key: tuple[str, str | None]) -> None:
        _pending_prompts.pop(key, None)

    async def _request_confirmation(
        self,
        ctx: ToolContext,
        prompt: str,
        user_id: str,
        group_id: str | None,
        image_path: str | None = None,
    ) -> str:
        from nonebot.adapters.onebot.v11 import MessageSegment

        _pending_prompts[(user_id, group_id)] = _PendingPrompt(
            prompt=prompt, created_at=time.time(), user_id=user_id, group_id=group_id,
            image_path=image_path,
        )
        if image_path:
            text = (
                f"我会基于你发的那张图这样修改：\n{prompt}\n\n"
                f"回复「确认」开始生成；回复「取消」或使用「/取消生图」取消；"
                f"想改就把新要求直接发我（{_CONFIRM_TIMEOUT_S:.0f} 秒内有效）。"
            )
        else:
            text = (
                f"我想生成这样一张图：\n{prompt}\n\n"
                f"回复「确认」开始生成；回复「取消」或使用「/取消生图」取消；"
                f"想改就把新描述直接发我（{_CONFIRM_TIMEOUT_S:.0f} 秒内有效）。"
            )
        try:
            response: Any
            if group_id and user_id:
                response = await ctx.bot.send_group_msg(
                    group_id=int(group_id),
                    message=MessageSegment.at(user_id=int(user_id)) + text,
                )
            elif group_id:
                response = await ctx.bot.send_group_msg(group_id=int(group_id), message=text)
            else:
                response = await ctx.bot.send_private_msg(user_id=int(user_id), message=text)
            if isinstance(response, dict) and isinstance(response.get("message_id"), int):
                _pending_prompts[(user_id, group_id)].confirmation_message_id = response["message_id"]
        except Exception as exc:
            _pending_prompts.pop((user_id, group_id), None)
            logger.warning(
                "imagegen confirm request send failed | user={} group={} err={}",
                user_id, group_id, exc,
            )
            return "提示词确认消息发送失败，请稍后再试。"
        logger.info("imagegen confirmation requested | user={} group={} chars={}", user_id, group_id, len(prompt))
        return "已向用户发送提示词确认请求，等 ta 回复「确认」后再生成。"

    async def _generate_and_send(
        self, bot: Any, prompt: str, user_id: str, group_id: str | None,
        reservation: ImageGenReservation,
    ) -> None:
        try:
            image_bytes = await self._generate(prompt)
        except Exception as exc:
            await self.usage.release(reservation)
            logger.warning("imagegen failed | model={} err={}", self._model, exc)
            await self._send_text(bot, user_id, group_id, "图片生成失败，请稍后再试。")
            return
        await self.usage.commit(reservation)
        await self._send_image(bot, image_bytes, user_id, group_id, prompt)

    async def _edit_and_send(
        self, bot: Any, prompt: str, image_path: str, user_id: str, group_id: str | None,
        reservation: ImageGenReservation,
    ) -> None:
        from pathlib import Path

        path = Path(image_path)
        if not path.exists():
            await self.usage.release(reservation)
            await self._send_text(bot, user_id, group_id, "原图已失效，请重新发一张图片再试。")
            return
        try:
            input_bytes = path.read_bytes()
            try:
                input_bytes = self._prepare_input_image(input_bytes)
            except Exception:
                logger.warning("imagegen edit input compress failed, sending original bytes", exc_info=True)
            image_bytes = await self._call_edits(input_bytes, prompt)
            try:
                image_bytes = self._compress(image_bytes)
            except Exception:
                logger.warning("imagegen edit compression failed, sending original bytes", exc_info=True)
        except Exception as exc:
            await self.usage.release(reservation)
            logger.warning("imagegen edit failed | model={} err={}", self._model, exc)
            await self._send_text(bot, user_id, group_id, "图片修改失败，请稍后再试。")
            return
        await self.usage.commit(reservation)
        await self._send_image(bot, image_bytes, user_id, group_id, prompt)

    async def _send_image(
        self, bot: Any, image_bytes: bytes, user_id: str, group_id: str | None, prompt: str,
    ) -> None:
        try:
            from nonebot.adapters.onebot.v11 import MessageSegment

            segment = MessageSegment.image(image_bytes)
            if group_id:
                await bot.send_group_msg(group_id=int(group_id), message=segment)
            else:
                await bot.send_private_msg(user_id=int(user_id), message=segment)
        except Exception:
            logger.warning(
                "imagegen send failed | group={} user={}", group_id, user_id, exc_info=True,
            )
            await self._send_text(bot, user_id, group_id, "图片生成成功但发送失败，请稍后再试。")
            return

        logger.info(
            "image sent | model={} chars={} group={} user={}",
            self._model, len(prompt), group_id, user_id,
        )

    async def _call_edits(self, image_bytes: bytes, prompt: str) -> bytes:
        """Call the OpenAI-compatible images/edits endpoint (multipart)."""
        headers = {"Authorization": f"Bearer {self._api_key}"}
        async with aiohttp.ClientSession(timeout=self._timeout, headers=headers) as session:
            form = aiohttp.FormData()
            form.add_field("model", self._model)
            form.add_field("prompt", prompt)
            form.add_field("size", self._size)
            form.add_field("n", "1")
            form.add_field("image", image_bytes, filename="input.jpg", content_type="image/jpeg")
            async with session.post(
                f"{self._base_url}/images/edits",
                data=form,
                proxy=self._proxy or None,
            ) as resp:
                if resp.status != 200:
                    body = (await resp.text())[:300]
                    raise RuntimeError(f"image edit API {resp.status}: {body}")
                data = await resp.json(content_type=None)
            entries = data.get("data") or []
            entry = entries[0] if entries else {}
            url = entry.get("url")
            if url:
                async with session.get(url) as img_resp:
                    img_resp.raise_for_status()
                    return await img_resp.read()
            if entry.get("b64_json"):
                return base64.b64decode(entry["b64_json"])
            raise RuntimeError(f"no image url or b64_json in edit response: {data!r}")

    @staticmethod
    def _prepare_input_image(image_bytes: bytes, max_dimension: int = 1024, quality: int = 90) -> bytes:
        """Downscale/re-encode the input image for the edit API to stay within size limits."""
        import pyvips

        image: Any = pyvips.Image.new_from_buffer(image_bytes, "")
        max_side = max(image.width, image.height)
        if max_side > max_dimension:
            image = image.resize(max_dimension / max_side)
        return image.jpegsave_buffer(Q=quality, strip=True)

    async def _send_text(self, bot: Any, user_id: str, group_id: str | None, text: str) -> None:
        if group_id:
            await bot.send_group_msg(group_id=int(group_id), message=text)
        else:
            await bot.send_private_msg(user_id=int(user_id), message=text)

    async def _generate(self, prompt: str) -> bytes:
        """Call CogView and return compressed JPEG bytes."""
        headers = {"Authorization": f"Bearer {self._api_key}"}
        payload = {"model": self._model, "prompt": prompt, "size": self._size, "n": 1}
        async with aiohttp.ClientSession(timeout=self._timeout, headers=headers) as session:
            async with session.post(
                f"{self._base_url}/images/generations",
                json=payload,
                proxy=self._proxy or None,
            ) as resp:
                if resp.status != 200:
                    body = (await resp.text())[:300]
                    raise RuntimeError(f"image API {resp.status}: {body}")
                data = await resp.json(content_type=None)
            entries = data.get("data") or []
            entry = entries[0] if entries else {}
            url = entry.get("url")
            if url:
                async with session.get(url) as img_resp:
                    img_resp.raise_for_status()
                    image_bytes = await img_resp.read()
            elif entry.get("b64_json"):
                image_bytes = base64.b64decode(entry["b64_json"])
            else:
                raise RuntimeError(f"no image url or b64_json in response: {data!r}")
        try:
            return self._compress(image_bytes)
        except Exception:
            logger.warning("imagegen compression failed, sending original bytes", exc_info=True)
            return image_bytes

    @staticmethod
    def _compress(image_bytes: bytes, max_dimension: int = 1024, quality: int = 85) -> bytes:
        """Resize to max dimension and re-encode as JPEG to keep QQ message small."""
        import pyvips

        image: Any = pyvips.Image.new_from_buffer(image_bytes, "")
        # libvips >= 8.16 renamed has_alpha -> hasalpha; support both versions
        has_alpha_fn = getattr(image, "hasalpha", None) or getattr(image, "has_alpha", None)
        if has_alpha_fn is not None and has_alpha_fn():
            image = image.flatten(background=[255, 255, 255])
        max_side = max(image.width, image.height)
        if max_side > max_dimension:
            image = image.resize(max_dimension / max_side)
        return image.jpegsave_buffer(Q=quality, strip=True)


class EditImageTool(GenerateImageTool):
    """图生图：基于用户发送的图片按指令生成修改后的新图。"""

    @property
    def name(self) -> str:
        return "edit_image"

    @property
    def description(self) -> str:
        return (
            "基于用户发送的图片生成一张修改后的新图片并发送到当前群或私聊。"
            "仅在用户提供了一张图并明确要求基于这张图修改、换背景、换风格、加元素等时调用；"
            "把修改要求润色成清晰的画面指令（改什么、保留什么、风格/氛围），规则：\n"
            "1. 明确说明改动点，以及必须保留不变的部分；\n"
            "2. 用户要求具体 → 不凭空添加内容；要求简单 → 只补充风格、光线等必要细节；\n"
            "3. 长度控制在 20~100 字。\n"
            "image 参数填用户消息里图片对应的标签（形如 img:1），通常取用户提到的那张；"
            "用户没有发图时不要调用本工具。\n"
            "注意：调用后不会立刻出图，会先向用户确认修改方案，回复「确认」后才真正生成。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": f"修改指令，最大 {self._max_prompt_chars} 字",
                },
                "image": {
                    "type": "string",
                    "description": "图片标签，如 img:1；缺省时用用户消息里的第一张图",
                },
            },
            "required": ["prompt"],
        }

    async def execute(self, ctx: ToolContext, **kwargs: Any) -> str:
        if not self._enabled:
            return "生图功能已关闭（管理员可用 /设置 生图开关 开 重新开启）"
        prompt = str(kwargs.get("prompt") or "").strip()
        if not prompt:
            return "请说明要对图片做什么修改。"
        if len(prompt) > self._max_prompt_chars:
            return f"描述过长，最大 {self._max_prompt_chars} 字。"
        if not ctx.bot or (not ctx.group_id and not ctx.user_id):
            return "没有可发送图片的聊天目标。"
        if not self._api_key:
            return "未配置生图 API key（管理员用 /设置 imagegen.api_key <key> 填写后重启生效）"

        tags = (ctx.extra or {}).get("image_tags") or {}
        tag = str(kwargs.get("image") or "").strip()
        image_path = tags.get(tag) if tag else None
        if not image_path and tags:
            image_path = next(iter(tags.values()))
        if not image_path:
            return "没有找到可用的图片，请把图片和修改要求一起发给我。"

        user_id = str(ctx.user_id or "")
        group_id = str(ctx.group_id) if ctx.group_id else None
        allowed, reason = await self.usage.check(
            user_id=user_id,
            group_id=group_id,
            global_limit=self._daily_global_limit,
            user_limit=self._daily_user_limit,
            group_limit=self._daily_group_limit,
            cooldown_s=self._cooldown_seconds,
        )
        if not allowed:
            logger.info("imagegen quota blocked | user={} group={} reason={!r}", user_id, group_id, reason)
            return reason
        return await self._request_confirmation(ctx, prompt, user_id, group_id, image_path=image_path)
