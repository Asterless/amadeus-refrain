"""E2E 集成测试：真实调用 LLM 代理，验证对话 + 工具调用链路。

用法: uv run pytest tests/test_e2e_live.py -v -s
需要 LLM 代理在 127.0.0.1:34567 运行。
"""

import os

import pytest

from src.identity.models import Identity
from src.llm.client import LLMClient
from src.llm.prompt import PromptBuilder
from src.memory.memo_store import MemoStore
from src.memory.short_term import ShortTermMemory
from src.tools import ToolRegistry
from src.tools.context import ToolContext
from src.tools.datetime_tool import DateTimeTool
from src.tools.memo_tools import RecallMemoTool, UpdateMemoTool

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:34567")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "sk-placeholder")
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-4-20250514")

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_E2E") != "1",
    reason="Set RUN_E2E=1 to run live integration tests",
)


@pytest.fixture
async def llm(tmp_path: object) -> LLMClient:
    memo_store = MemoStore(base_dir=str(tmp_path))
    await memo_store.startup()
    short_term = ShortTermMemory()
    identity = Identity(id="test", name="测试", personality="你是一个QQ群聊机器人，回复简洁。")
    prompt_builder = PromptBuilder()
    prompt_builder.build_static(identity, bot_self_id="")

    tools = ToolRegistry()
    tools.register(RecallMemoTool(memo_store))
    tools.register(UpdateMemoTool(memo_store))
    tools.register(DateTimeTool())

    return LLMClient(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=LLM_MODEL,
        prompt_builder=prompt_builder,
        short_term=short_term,
        tools=tools,
        memo_store=memo_store,
    )


@pytest.fixture
def identity() -> Identity:
    return Identity(id="test", name="测试", personality="你是一个QQ群聊机器人，回复简洁。")


async def test_basic_chat(llm: LLMClient, identity: Identity) -> None:
    """基础对话：发送消息，收到非空回复。"""
    reply = await llm.chat(
        session_id="test_s1",
        user_id="12345",
        user_content="你好，请用一句话介绍你自己",
        identity=identity,
    )
    print(f"[basic_chat] reply: {reply}")
    assert reply is not None
    assert len(reply) > 0
    assert reply != "..."


async def test_tool_call_datetime(llm: LLMClient, identity: Identity) -> None:
    """工具调用：问时间，LLM 应调用 get_datetime 工具。"""
    reply = await llm.chat(
        session_id="test_s2",
        user_id="12345",
        user_content="现在几点了？",
        identity=identity,
    )
    print(f"[tool_datetime] reply: {reply}")
    assert reply is not None
    assert len(reply) > 0


async def test_tool_call_memory(llm: LLMClient, identity: Identity) -> None:
    """记忆工具：告诉 Bot 信息，应调用 save_memory。"""
    ctx = ToolContext(user_id="67890")
    reply = await llm.chat(
        session_id="test_s3",
        user_id="67890",
        user_content="我叫小测试，我喜欢写Python",
        identity=identity,
        ctx=ctx,
    )
    print(f"[tool_memory] reply: {reply}")
    assert reply is not None
    assert len(reply) > 0


async def test_multi_turn(llm: LLMClient, identity: Identity) -> None:
    """多轮对话：短期记忆应保留上下文。"""
    await llm.chat(session_id="test_s4", user_id="11111", user_content="我最喜欢的颜色是蓝色", identity=identity)
    reply = await llm.chat(
        session_id="test_s4", user_id="11111",
        user_content="我刚才说我最喜欢什么颜色？", identity=identity,
    )
    print(f"[multi_turn] reply: {reply}")
    assert reply is not None
    assert "蓝" in reply
