"""用免费识图模型重新校对表情包库的描述与适用场景。

遍历 storage/stickers 索引里的每张表情包，调用 [vision] 配置的视觉模型
重新识别画面内容，更新 index.json 中的 description / usage_hint。
保留 send_count / last_sent / source / created_at 等字段。

容器内运行：
    .venv/bin/python scripts/refresh_stickers.py [--only stk_xxxx]

运行后需要重启 bot 让新索引生效：
    docker compose restart bot
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tomllib
from pathlib import Path

# 允许从任意工作目录运行：把项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sticker.store import StickerStore
from src.vision import VisionClient


async def main() -> None:
    parser = argparse.ArgumentParser(description="重新校对表情包描述")
    parser.add_argument("--only", default=None, help="只校对指定 sticker_id")
    args = parser.parse_args()

    cfg = tomllib.loads(Path("config.toml").read_text(encoding="utf-8"))
    vision = cfg.get("vision", {})
    if not (vision.get("base_url") and vision.get("api_key") and vision.get("model")):
        print("[vision] 未配置 base_url/api_key/model，无法识图校对")
        sys.exit(1)

    sticker_dir = cfg.get("sticker", {}).get("storage_dir", "storage/stickers")
    store = StickerStore(storage_dir=sticker_dir)
    vc = VisionClient(
        base_url=vision["base_url"],
        api_key=vision["api_key"],
        model=vision["model"],
    )

    entries = store.list_all()
    targets = [args.only] if args.only else list(entries)
    ok_count = 0

    for sid in targets:
        entry = entries.get(sid)
        if entry is None:
            print(f"skip {sid} (索引中不存在)")
            continue
        path = store.resolve_path(sid)
        if path is None or not path.exists():
            print(f"skip {sid} (文件缺失: {entry.get('file')})")
            continue

        pair = await vc.describe_sticker(str(path))
        if pair is None:
            print(f"FAIL {sid} (识别失败，跳过)")
            continue
        desc, hint = pair
        old_desc = entry.get("description", "")
        old_hint = entry.get("usage_hint", "")
        store.update(sid, description=desc, usage_hint=hint)
        ok_count += 1
        print(f"OK {sid}")
        print(f"  旧: {old_desc} | {old_hint}")
        print(f"  新: {desc} | {hint}")

    await vc.close()
    print(f"完成：{ok_count}/{len(targets)} 张已重新校对")
    print("提示：重启 bot 让新索引生效 → docker compose restart bot")


if __name__ == "__main__":
    asyncio.run(main())
