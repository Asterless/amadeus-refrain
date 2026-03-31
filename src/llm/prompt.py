"""Soul 层：动态拼装 System Prompt，分层支持 context caching。

缓存策略：
  system blocks（带 cache_control，跨请求复用）：
    1. 人设性格 + 工具指南  → 几乎不变
    2. 用户记忆(.qmd)       → 偶尔更新

  群聊记录 → 不放 system，改为 messages 注入（每条消息都变，放 system 会打破 cache）
"""

from typing import Any

from src.identity.models import Identity
from src.memory.group_context import GroupContext
from src.memory.long_term import LongTermMemory

TOOL_GUIDE = """\
【工具使用指南】
- 当用户提到自己的信息（昵称、爱好、经历等）时，用 save_memory 记住
- 当需要回忆用户信息时，用 recall_memory 查询
- 当用户问时间日期时，用 get_datetime 获取
- 当需要查网页内容时，用 web_fetch 抓取
- 当需要调外部 API 时，用 http_api 调用
- 群管理操作（禁言、头衔、发消息）用对应的群管理工具
"""


class PromptBuilder:
    def __init__(self, long_term: LongTermMemory, group_context: GroupContext) -> None:
        self._long_term = long_term
        self._group_context = group_context

    async def build_blocks(self, identity: Identity, user_id: str, group_id: str | None = None) -> list[dict[str, Any]]:
        """返回 system blocks，只含稳定内容，最大化 cache 命中。"""
        blocks: list[dict[str, Any]] = []

        # 层 1：人设 + 工具指南（最稳定）
        base_text = identity.personality + "\n\n" + TOOL_GUIDE
        if group_id:
            base_text += f"\n【当前在群 {group_id} 中对话】"
        blocks.append({"type": "text", "text": base_text, "cache_control": {"type": "ephemeral"}})

        # 层 2：用户记忆（偶尔更新，单独一层以便 cache）
        memory_ctx = await self._long_term.get_full_context(user_id)
        if memory_ctx.strip():
            blocks.append({
                "type": "text",
                "text": f"【关于当前用户 {user_id} 的记忆】\n{memory_ctx.strip()}",
                "cache_control": {"type": "ephemeral"},
            })

        return blocks

    def build_context_message(self, group_id: str) -> str | None:
        """返回群聊记录文本，由 client 注入到 messages 开头。"""
        ctx = self._group_context.format(group_id)
        return ctx if ctx else None
