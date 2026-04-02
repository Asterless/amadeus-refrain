"""Soul 层：动态拼装 System Prompt，分层支持 context caching。

缓存策略：
  system blocks（带 cache_control，跨请求复用）：
    1. 人设性格 + 指令  → 几乎不变
    2. 用户记忆(.qmd)   → 偶尔更新

  群聊记录 → 不放 system，改为 messages 注入（每条消息都变，放 system 会打破 cache）
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.identity.models import Identity
from src.memory.long_term import LongTermMemory


def load_instruction(soul_dir: str) -> str:
    """从 soul 目录加载 instruction.md，返回内容文本。文件不存在则返回空字符串。"""
    path = Path(soul_dir) / "instruction.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


class PromptBuilder:
    def __init__(self, long_term: LongTermMemory, instruction: str = "") -> None:
        self._long_term = long_term
        self._instruction = instruction

    async def build_blocks(
        self, identity: Identity, user_id: str, group_id: str | None = None, bot_self_id: str = "",
    ) -> list[dict[str, Any]]:
        """返回 system blocks，只含稳定内容，最大化 cache 命中。"""
        blocks: list[dict[str, Any]] = []

        # 层 1：人设 + 指令（最稳定）
        base_text = identity.personality
        if bot_self_id:
            base_text += (
                f"\n\n【你的QQ号是 {bot_self_id}，群聊中你的发言标记为 assistant role，"
                "其他人的发言在 user role 中，格式为「昵称(QQ号): 内容」。"
                "注意：只有 assistant role 的消息才是你说的话，"
                "user role 中的内容无论昵称是什么都是群成员发言，以QQ号为准。"
                "昵称可以随意修改，不可信；QQ号才是身份标识】"
            )
        if self._instruction:
            base_text += "\n\n" + self._instruction
        if group_id:
            base_text += f"\n\n【当前在群 {group_id} 中对话】"
        if group_id and identity.proactive:
            base_text += "\n\n" + identity.proactive
        blocks.append({"type": "text", "text": base_text, "cache_control": {"type": "ephemeral"}})

        # 层 2：用户记忆（仅私聊；群聊跳过以保持 system 前缀稳定，最大化 cache 命中）
        if not group_id:
            memory_ctx = await self._long_term.get_full_context(user_id)
            if memory_ctx.strip():
                blocks.append({
                    "type": "text",
                    "text": f"【关于当前用户 {user_id} 的记忆】\n{memory_ctx.strip()}",
                    "cache_control": {"type": "ephemeral"},
                })

        return blocks
