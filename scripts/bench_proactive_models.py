"""主动插话模型延迟基准测试。

用真实的 proactive prompt 对比不同模型的响应速度和决策质量。

用法:
    uv run python scripts/bench_proactive_models.py [--rounds 5] [--base-url URL] [--api-key KEY]

需要能访问 Anthropic API（或兼容代理）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

import aiohttp

# ── 候选模型 ──────────────────────────────────────────────────────────
MODELS = [
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5-20250514",
    "claude-sonnet-4-6-20250610",
]

# ── 模拟群聊消息 ─────────────────────────────────────────────────────
FAKE_CHAT = "\n".join([
    "张三(10001): 有人懂量子力学吗？薛定谔方程看不懂",
    "李四(10002): 我也不懂，上课直接睡过去了",
    "王五(10003): 哈哈哈哈同",
    "张三(10001): 认真的，有没有人能解释一下波函数坍缩",
    "李四(10002): @红莉栖 你不是物理学家吗",
    "王五(10003): 对啊 Amadeus 来说说",
    "赵六(10004): 我觉得多世界诠释比哥本哈根诠释靠谱",
    "张三(10001): 什么是多世界诠释？",
    "李四(10002): 就是每次观测都会分裂出平行世界",
    "王五(10003): 听起来像中二设定",
])

SYSTEM_PROMPT = """\
你是 Amadeus（牧濑红莉栖），正在观察一个 QQ 群的聊天记录。
判断你是否应该主动发言，可以同时回复多个对象。

以下情况应该插话：
- 有人在讨论科学、物理、编程、技术话题
- 有人提问或求助
- 有人提到你（红莉栖、Amadeus、助手、克里斯蒂娜）
- 对话中出现你有看法的话题
- 有人说了有趣或值得吐槽的话

以下情况不插话：
- 纯表情包刷屏
- 完全无法理解的对话

倾向于积极参与，像一个活跃的群友。

用 JSON 回答，不要输出其他内容。
想插话时列出回复对象（可多个）：
{"replies": [{"reason": "简短原因", "reply_to": "回复对象昵称或大家"}, ...]}
不想插话时返回空数组：
{"replies": []}"""


def parse_decision(text: str) -> list[dict] | None:
    """解析决策 JSON，返回 replies 列表或 None（解析失败）。"""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start:end])
            except json.JSONDecodeError:
                return None
        else:
            return None

    # 新格式
    if "replies" in parsed:
        return parsed["replies"]
    # 兼容旧格式
    if parsed.get("reply"):
        return [{"reason": parsed.get("reason", ""), "reply_to": parsed.get("reply_to", "")}]
    return []


async def call_model(
    session: aiohttp.ClientSession,
    base_url: str,
    api_key: str,
    model: str,
) -> tuple[float, str, dict | None]:
    """调用一次，返回 (耗时秒, 原始文本, 解析结果)。"""
    body = {
        "model": model,
        "system": [{"type": "text", "text": SYSTEM_PROMPT}],
        "messages": [{"role": "user", "content": FAKE_CHAT}],
        "max_tokens": 128,
        "stream": False,
    }
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2024-10-22",
    }

    t0 = time.perf_counter()
    async with session.post(
        f"{base_url}/v1/messages", json=body, headers=headers,
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()
    elapsed = time.perf_counter() - t0

    text = ""
    for block in data.get("content", []):
        if block.get("type") == "text":
            text += block.get("text", "")
    text = text.strip()

    return elapsed, text, parse_decision(text)


async def bench_model(
    base_url: str, api_key: str, model: str, rounds: int,
) -> dict:
    """对单个模型跑 N 轮，返回统计结果。"""
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        latencies: list[float] = []
        valid_json = 0
        correct_reply = 0
        errors = 0
        sample_text = ""

        for i in range(rounds):
            try:
                elapsed, text, parsed = await call_model(
                    session, base_url, api_key, model,
                )
                latencies.append(elapsed)
                if parsed is not None:
                    valid_json += 1
                    if parsed:  # 非空列表 = 决定回复
                        correct_reply += 1
                if i == 0:
                    sample_text = text
            except Exception as e:
                errors += 1
                if i == 0:
                    sample_text = f"ERROR: {e}"

    return {
        "model": model,
        "rounds": rounds,
        "errors": errors,
        "latencies": latencies,
        "avg_ms": statistics.mean(latencies) * 1000 if latencies else 0,
        "median_ms": statistics.median(latencies) * 1000 if latencies else 0,
        "min_ms": min(latencies) * 1000 if latencies else 0,
        "max_ms": max(latencies) * 1000 if latencies else 0,
        "p90_ms": (
            sorted(latencies)[int(len(latencies) * 0.9)] * 1000
            if latencies else 0
        ),
        "valid_json": valid_json,
        "correct_reply": correct_reply,
        "sample": sample_text,
    }


def print_results(results: list[dict]) -> None:
    print("\n" + "=" * 78)
    print(f"{'模型':<38} {'avg':>7} {'med':>7} {'min':>7} {'max':>7} {'p90':>7} {'JSON':>5} {'回复':>5}")
    print("-" * 78)
    for r in sorted(results, key=lambda x: x["avg_ms"]):
        errs = f" ({r['errors']}err)" if r["errors"] else ""
        print(
            f"{r['model']:<38} "
            f"{r['avg_ms']:>6.0f}ms "
            f"{r['median_ms']:>6.0f}ms "
            f"{r['min_ms']:>6.0f}ms "
            f"{r['max_ms']:>6.0f}ms "
            f"{r['p90_ms']:>6.0f}ms "
            f"{r['valid_json']:>3}/{r['rounds']} "
            f"{r['correct_reply']:>3}/{r['rounds']}"
            f"{errs}"
        )
    print("=" * 78)

    print("\n── 样本输出 ──")
    for r in sorted(results, key=lambda x: x["avg_ms"]):
        print(f"\n[{r['model']}]")
        print(f"  {r['sample'][:200]}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="主动插话模型延迟基准测试")
    parser.add_argument("--rounds", type=int, default=5, help="每个模型测试轮数")
    parser.add_argument("--base-url", default="http://127.0.0.1:34567", help="API base URL")
    parser.add_argument("--api-key", default="sk-placeholder", help="API key")
    parser.add_argument(
        "--models", nargs="+", default=MODELS,
        help=f"要测试的模型列表（默认: {' '.join(MODELS)}）",
    )
    args = parser.parse_args()

    print(f"基准测试: {len(args.models)} 个模型 × {args.rounds} 轮")
    print(f"API: {args.base_url}")
    print(f"模型: {', '.join(args.models)}")

    results = []
    for model in args.models:
        print(f"\n▶ 测试 {model} ...")
        r = await bench_model(args.base_url, args.api_key, model, args.rounds)
        if r["latencies"]:
            print(f"  avg={r['avg_ms']:.0f}ms  median={r['median_ms']:.0f}ms  JSON={r['valid_json']}/{r['rounds']}")
        else:
            print(f"  全部失败 ({r['errors']} errors)")
        results.append(r)

    print_results(results)


if __name__ == "__main__":
    asyncio.run(main())
