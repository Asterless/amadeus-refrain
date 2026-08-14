"""Text-to-speech tool supporting Edge TTS and GPT-SoVITS."""

from __future__ import annotations

from typing import Any

import httpx
from loguru import logger

from src.tools.base import Tool
from src.tools.context import ToolContext


class SendVoiceTool(Tool):
    def __init__(
        self,
        *,
        provider: str,
        voice: str,
        rate: str,
        volume: str,
        proxy: str,
        base_url: str,
        ref_audio_path: str,
        prompt_text: str,
        prompt_lang: str,
        text_lang: str,
        text_split_method: str,
        media_type: str,
        timeout_seconds: float,
        max_chars: int,
    ) -> None:
        self._provider = provider
        self._voice = voice
        self._rate = rate
        self._volume = volume
        self._proxy = proxy.strip() or None
        self._base_url = base_url.rstrip("/")
        self._ref_audio_path = ref_audio_path
        self._prompt_text = prompt_text
        self._prompt_lang = prompt_lang
        self._text_lang = text_lang
        self._text_split_method = text_split_method
        self._media_type = media_type
        self._timeout_seconds = timeout_seconds
        self._max_chars = max_chars

    async def _synthesize(self, text: str) -> bytes:
        if self._provider == "gpt_sovits":
            if not self._ref_audio_path:
                raise ValueError("GPT-SoVITS ref_audio_path is empty")
            payload = {
                "text": text,
                "text_lang": self._text_lang,
                "ref_audio_path": self._ref_audio_path,
                "prompt_lang": self._prompt_lang,
                "prompt_text": self._prompt_text,
                "text_split_method": self._text_split_method,
                "media_type": self._media_type,
                "streaming_mode": False,
            }
            async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                response = await client.post(f"{self._base_url}/tts", json=payload)
                response.raise_for_status()
            if "json" in response.headers.get("content-type", ""):
                raise RuntimeError(response.text[:500])
            return response.content

        import edge_tts

        audio = bytearray()
        communicate = edge_tts.Communicate(
            text, self._voice, rate=self._rate, volume=self._volume, proxy=self._proxy,
        )
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                audio.extend(chunk.get("data", b""))
        return bytes(audio)

    @property
    def name(self) -> str:
        return "send_voice"

    @property
    def description(self) -> str:
        return (
            "把文本合成为 QQ 语音并发送到当前群或私聊。"
            "仅在用户明确要求语音、朗读或说出来时调用；发送成功后不要重复发送同样的文字。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": f"要朗读的纯文本，最多 {self._max_chars} 字"},
            },
            "required": ["text"],
        }

    async def execute(self, ctx: ToolContext, **kwargs: Any) -> str:
        text = str(kwargs.get("text") or "").strip()
        if not text:
            return "没有可朗读的文本。"
        if len(text) > self._max_chars:
            return f"语音文本过长，最多 {self._max_chars} 字。"
        if not ctx.bot or (not ctx.group_id and not ctx.user_id):
            return "没有可发送语音的聊天目标。"

        try:
            audio = await self._synthesize(text)
            if not audio:
                return "语音生成失败：没有生成音频。"

            from nonebot.adapters.onebot.v11 import MessageSegment

            segment = MessageSegment.record(audio)
            if ctx.group_id:
                await ctx.bot.send_group_msg(group_id=int(ctx.group_id), message=segment)
            else:
                await ctx.bot.send_private_msg(user_id=int(ctx.user_id), message=segment)
        except Exception:
            logger.warning("tts send failed | provider={} chars={}", self._provider, len(text), exc_info=True)
            return "语音生成或发送失败，请稍后再试。"

        logger.info(
            "voice sent | provider={} chars={} voice={} group={} user={}",
            self._provider, len(text), self._voice, ctx.group_id, ctx.user_id,
        )
        return "语音已发送"
