"""MemoStore: structured memo files with in-memory cache, mentions index, and atomic writes."""

import asyncio
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import aiofiles
from loguru import logger

TZ_SHANGHAI = ZoneInfo("Asia/Shanghai")

_META_RE = re.compile(r"^<!--\s*updated:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\s*\|\s*source:\s*(.+?)\s*-->$")
_USER_REF_RE = re.compile(r"@(\d+)")
_GROUP_REF_RE = re.compile(r"#(\d+)")

PENDING_HEADER = "## 待整理"

USER_MEMO_TEMPLATE = """\
@QQ号(昵称)
身份: 职业/角色，注明自称还是已确认
性格: 说话风格、行为特征
关系: 与其他用户的关系
备注:
- 值得记录的事件或印象

## 待整理"""

GROUP_MEMO_TEMPLATE = """\
### 成员
- @QQ号(昵称): 角色/特征

### 话题
- 群内常见话题或重要讨论

### 事件
- 值得记录的群事件

### 规矩
- 群内约定或潜规则

## 待整理"""


@dataclass(frozen=True)
class Memo:
    id: str
    kind: Literal["user", "group"]
    identity: str  # First content line (index summary source)
    body: str  # Full prose body (everything after metadata line)
    refs: frozenset[str]  # @/# references found in body
    updated: datetime
    source: str  # e.g. "compact:group:987654"


def parse_memo(id: str, text: str) -> Memo:
    """Parse a memo file's text content into a Memo dataclass."""
    lines = text.splitlines()

    # Parse metadata from line 1 (HTML comment)
    updated: datetime = datetime.now(tz=TZ_SHANGHAI)
    source: str = ""
    body_start = 0

    if lines and (m := _META_RE.match(lines[0].strip())):
        updated = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M").replace(tzinfo=TZ_SHANGHAI)
        source = m.group(2)
        body_start = 1

    # Body is everything after the metadata comment
    body_lines = lines[body_start:]
    body = "\n".join(body_lines)

    # First non-empty content line after metadata becomes identity
    identity = ""
    for line in body_lines:
        stripped = line.strip()
        if stripped:
            identity = stripped
            break

    # Extract refs from full text
    refs = frozenset(_USER_REF_RE.findall(text)) | frozenset(_GROUP_REF_RE.findall(text))

    # Derive kind from id prefix
    kind: Literal["user", "group"] = "user" if id.startswith("user_") else "group"

    return Memo(
        id=id,
        kind=kind,
        identity=identity,
        body=body,
        refs=refs,
        updated=updated,
        source=source,
    )


class LockManager:
    def __init__(self) -> None:
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def get(self, id: str) -> asyncio.Lock:
        return self._locks[id]


class MemoStore:
    def __init__(
        self,
        base_dir: str,
        history_enabled: bool = True,
        user_max_chars: int = 0,
        group_max_chars: int = 0,
        index_max_lines: int = 0,
    ) -> None:
        self._base_dir = Path(base_dir)
        self._history_enabled = history_enabled
        self._user_max_chars = user_max_chars
        self._group_max_chars = group_max_chars
        self._index_max_lines = index_max_lines
        self._memos: dict[str, Memo] = {}
        # Reverse index: user/group numeric id → set of memo IDs that mention it
        self._mentions: defaultdict[str, set[str]] = defaultdict(set)
        self._lock_manager = LockManager()
        # Separate lock for index.md writes to prevent concurrent rename races
        self._index_lock = asyncio.Lock()
        self._started = False

    def _id_to_path(self, id: str) -> Path:
        """Map memo ID to file path. Raises ValueError on path traversal."""
        if ".." in id or "/" in id:
            raise ValueError(f"Invalid memo ID (path traversal detected): {id!r}")
        if id.startswith("user_"):
            num = id[len("user_"):]
            return self._base_dir / "users" / f"{num}.md"
        elif id.startswith("group_"):
            num = id[len("group_"):]
            return self._base_dir / "groups" / f"{num}.md"
        else:
            raise ValueError(f"Unknown memo ID format: {id!r}")

    def _id_to_log_path(self, id: str) -> Path:
        """Map memo ID to log file path."""
        md_path = self._id_to_path(id)
        return md_path.with_suffix(".log")

    async def startup(self) -> None:
        """Load all memos from disk, build caches, clean .tmp residuals."""
        self._memos.clear()
        self._mentions.clear()

        # Clean up any leftover .tmp files
        if self._base_dir.exists():
            for tmp_file in self._base_dir.rglob("*.tmp"):
                try:
                    tmp_file.unlink()
                    logger.debug(f"Removed leftover tmp file: {tmp_file}")
                except OSError as e:
                    logger.warning(f"Failed to remove tmp file {tmp_file}: {e}")

        # Load all existing memo files
        for subdir_name, kind_prefix in [("users", "user_"), ("groups", "group_")]:
            subdir = self._base_dir / subdir_name
            if not subdir.exists():
                continue
            for md_file in subdir.glob("*.md"):
                num = md_file.stem
                memo_id = f"{kind_prefix}{num}"
                try:
                    async with aiofiles.open(md_file, encoding="utf-8") as f:
                        text = await f.read()
                    memo = parse_memo(memo_id, text)
                    self._memos[memo_id] = memo
                    self._update_mentions(memo_id, memo.refs)
                except Exception as e:
                    logger.warning(f"Failed to load memo {md_file}: {e}")

        self._started = True
        logger.info(f"MemoStore loaded {len(self._memos)} memos from {self._base_dir}")

    def _check_started(self) -> None:
        if not self._started:
            raise RuntimeError("MemoStore.startup() must be called before use")

    def _update_mentions(self, memo_id: str, refs: frozenset[str]) -> None:
        """Add memo_id to the mentions index for each ref."""
        for ref in refs:
            self._mentions[ref].add(memo_id)

    def _remove_mentions(self, memo_id: str, refs: frozenset[str]) -> None:
        """Remove memo_id from the mentions index for each ref."""
        for ref in refs:
            s = self._mentions.get(ref)
            if s:
                s.discard(memo_id)

    def read(self, id: str) -> Memo | None:
        self._check_started()
        return self._memos.get(id)

    def about(self, user_id: str) -> list[Memo]:
        """All memos mentioning @user_id. Uses reverse index for fast lookup."""
        self._check_started()
        memo_ids = self._mentions.get(user_id, set())
        return [self._memos[mid] for mid in memo_ids if mid in self._memos]

    def list_ids(self, kind: Literal["user", "group"] | None = None) -> list[str]:
        """List all memo IDs, optionally filtered by kind."""
        self._check_started()
        if kind is None:
            return list(self._memos.keys())
        return [id for id in self._memos if self._memos[id].kind == kind]

    def serialize_index(self) -> str:
        """Render index.md content from _memos.

        Format:
            # users
            - @{qq} {identity} | {#group refs}

            # groups
            - #{gid} {identity} | {@user refs}
        """
        self._check_started()
        user_lines: list[str] = []
        group_lines: list[str] = []

        for memo_id, memo in sorted(self._memos.items()):
            if memo.kind == "user":
                num = memo_id[len("user_"):]
                group_ref_nums = sorted(_GROUP_REF_RE.findall(memo.body))
                group_refs_str = " ".join(f"#{r}" for r in group_ref_nums) if group_ref_nums else ""
                suffix = f" | {group_refs_str}" if group_refs_str else ""
                user_lines.append(f"- @{num} {memo.identity}{suffix}")
            else:
                num = memo_id[len("group_"):]
                user_ref_nums = sorted(_USER_REF_RE.findall(memo.body))
                user_refs_str = " ".join(f"@{r}" for r in user_ref_nums) if user_ref_nums else ""
                suffix = f" | {user_refs_str}" if user_refs_str else ""
                group_lines.append(f"- #{num} {memo.identity}{suffix}")

        parts: list[str] = []
        if user_lines:
            parts.append("# users\n" + "\n".join(user_lines))
        if group_lines:
            parts.append("# groups\n" + "\n".join(group_lines))

        full = "\n\n".join(parts) + "\n" if parts else ""
        if self._index_max_lines > 0:
            lines = full.splitlines()
            if len(lines) > self._index_max_lines:
                full = "\n".join(lines[: self._index_max_lines]) + "\n"
        return full

    async def write(self, id: str, memo: str, source: str) -> None:
        """Atomic write with lock.

        1. Create parent dirs if needed
        2. Enforce max_chars limit
        3. Prepend metadata comment
        4. Write to .md.tmp
        5. os.rename(.md.tmp → .md) (atomic on POSIX)
        6. Append one line to .log if history_enabled
        7. Update _memos cache and _mentions index incrementally
        8. Rewrite index.md
        """
        self._check_started()
        # Validate ID (raises ValueError on path traversal)
        md_path = self._id_to_path(id)

        # Enforce max_chars
        kind: Literal["user", "group"] = "user" if id.startswith("user_") else "group"
        max_chars = self._user_max_chars if kind == "user" else self._group_max_chars
        if max_chars > 0 and len(memo) > max_chars:
            logger.warning("memo {} exceeds max_chars ({}/{}), truncating", id, len(memo), max_chars)
            memo = memo[:max_chars]

        async with self._lock_manager.get(id):
            # 1. Create parent dirs
            md_path.parent.mkdir(parents=True, exist_ok=True)

            # 2. Prepend metadata comment
            now = datetime.now(tz=TZ_SHANGHAI).strftime("%Y-%m-%d %H:%M")
            full_text = f"<!-- updated: {now} | source: {source} -->\n\n{memo}"

            # 3. Write to .tmp
            tmp_path = md_path.with_suffix(".md.tmp")
            async with aiofiles.open(tmp_path, "w", encoding="utf-8") as f:
                await f.write(full_text)

            # 4. Atomic rename
            os.rename(tmp_path, md_path)

            # 5. Append to changelog
            if self._history_enabled:
                log_path = self._id_to_log_path(id)
                async with aiofiles.open(log_path, "a", encoding="utf-8") as f:
                    await f.write(f"<!-- updated: {now} | source: {source} -->\n")

            # 6. Update cache and mentions index incrementally
            old_refs = self._memos[id].refs if id in self._memos else frozenset()
            parsed = parse_memo(id, full_text)
            self._memos[id] = parsed
            for ref in old_refs - parsed.refs:
                self._mentions[ref].discard(id)
            for ref in parsed.refs - old_refs:
                self._mentions[ref].add(id)

        # 7. Rewrite index.md (outside per-id lock to avoid deadlock)
        async with self._index_lock:
            index_content = self.serialize_index()
            index_path = self._base_dir / "index.md"
            index_tmp = self._base_dir / "index.md.tmp"
            async with aiofiles.open(index_tmp, "w", encoding="utf-8") as f:
                await f.write(index_content)
            os.rename(index_tmp, index_path)

        logger.debug(f"Wrote memo {id} from source={source!r}")

    async def append(self, id: str, note: str, source: str) -> None:
        """Append a note as a bullet item under the '## 待整理' section.

        Creates the section if missing. Creates the memo if it doesn't exist.
        The combined content is still subject to max_chars truncation.
        """
        self._check_started()
        existing = self._memos.get(id)
        if existing:
            body = existing.body.strip()
            combined = f"{body}\n- {note}" if PENDING_HEADER in body else f"{body}\n\n{PENDING_HEADER}\n- {note}"
        else:
            combined = f"{PENDING_HEADER}\n- {note}"
        await self.write(id, combined, source)
