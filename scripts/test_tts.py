"""Smoke test for the GPT-SoVITS /tts endpoint (mirrors SendVoiceTool payload)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

GPT_ROOT = r"D:\GPT-SoVITS\GPT-SoVITS-v4-20250529\GPT-SoVITS-v4-20250529"
OUT = Path(r"D:\Users\haoyu\Desktop\Bot\amadeus-in-shell\storage\tts_test.wav")


async def main() -> None:
    payload = {
        "text": "你好，我是诗歌剧，很高兴认识你！",
        "text_lang": "zh",
        "ref_audio_path": str(
            Path(GPT_ROOT) / "logs" / "shigeju" / "5-wav32k" / "simple.mp4_0000000000_0000136000.wav"
        ),
        "prompt_lang": "zh",
        "prompt_text": "今天来教大家如何使用帝宝和诗歌剧唱歌.",
        "text_split_method": "cut5",
        "media_type": "wav",
        "streaming_mode": False,
        "batch_size": 1,
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as client:
        resp = await client.post("http://127.0.0.1:9880/tts", json=payload)
    ctype = resp.headers.get("content-type", "")
    print("status:", resp.status_code)
    print("content-type:", ctype)
    if "json" in ctype:
        print("error body:", resp.text[:500])
        return
    OUT.write_bytes(resp.content)
    print("saved:", OUT, "bytes:", len(resp.content), "head:", resp.content[:12].hex())


if __name__ == "__main__":
    asyncio.run(main())
