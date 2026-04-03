# Memo 待整理 Section Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `## 待整理` staging section to memos that compact auto-appends to, and dream auto-merges into the main body.

**Architecture:** `memo_store.append()` inserts bullet items under a `## 待整理` header at the end of the memo. `dream_pre_check()` detects non-empty pending sections and flags them. The dream prompt instructs the LLM to merge pending items into the structured main body and clear the section. `soul/instruction.md` documents the templates with the reserved section.

**Tech Stack:** Python, pytest, aiofiles

---

### Task 1: `memo_store.append()` — write to pending section

**Files:**
- Modify: `src/memory/memo_store.py:298-311`
- Test: `tests/test_memo_store.py`

- [ ] **Step 1: Write tests for pending-section append behavior**

Add three tests at the end of `tests/test_memo_store.py`:

```python
async def test_append_creates_pending_section(store: MemoStore) -> None:
    """append() on a memo without 待整理 creates the section."""
    await store.startup()
    await store.write("user_100", "@100(测试)\n身份: 学生", "test")
    await store.append("user_100", "喜欢打篮球", "compact:private:user_100")
    memo = store.read("user_100")
    assert memo is not None
    assert "## 待整理" in memo.body
    assert "- 喜欢打篮球" in memo.body


async def test_append_adds_to_existing_pending(store: MemoStore) -> None:
    """Second append() adds a new bullet to the existing 待整理 section."""
    await store.startup()
    await store.write("user_100", "@100(测试)\n\n## 待整理\n- 第一条", "test")
    await store.append("user_100", "第二条", "compact:private:user_100")
    memo = store.read("user_100")
    assert memo is not None
    assert "- 第一条" in memo.body
    assert "- 第二条" in memo.body


async def test_append_new_memo_creates_pending(store: MemoStore) -> None:
    """append() on a non-existent memo creates it with just a 待整理 section."""
    await store.startup()
    await store.append("user_200", "新用户观察", "compact:private:user_200")
    memo = store.read("user_200")
    assert memo is not None
    assert "## 待整理" in memo.body
    assert "- 新用户观察" in memo.body
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_memo_store.py::test_append_creates_pending_section tests/test_memo_store.py::test_append_adds_to_existing_pending tests/test_memo_store.py::test_append_new_memo_creates_pending -v`
Expected: FAIL — current `append()` doesn't create `## 待整理` section.

- [ ] **Step 3: Implement pending-section append**

In `src/memory/memo_store.py`, add a module-level constant and rewrite `append()`:

```python
PENDING_HEADER = "## 待整理"
```

```python
async def append(self, id: str, note: str, source: str) -> None:
    """Append a note as a bullet item under the '## 待整理' section.

    Creates the section if missing. Creates the memo if it doesn't exist.
    The combined content is still subject to max_chars truncation.
    """
    self._check_started()
    existing = self._memos.get(id)
    if existing:
        body = existing.body.strip()
        if PENDING_HEADER in body:
            combined = f"{body}\n- {note}"
        else:
            combined = f"{body}\n\n{PENDING_HEADER}\n- {note}"
    else:
        combined = f"{PENDING_HEADER}\n- {note}"
    await self.write(id, combined, source)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_memo_store.py -v`
Expected: All pass including the 3 new ones.

- [ ] **Step 5: Run full suite**

Run: `uv run pytest -x -q`
Expected: All pass. The existing `test_private_compact_appends_memo` and `test_group_compact_appends_memos` in `test_client.py` may need assertion updates (body now contains `## 待整理` and `- ` prefix). Fix if needed.

- [ ] **Step 6: Commit**

```bash
git add src/memory/memo_store.py tests/test_memo_store.py
git commit -m "feat: append memo notes to '## 待整理' pending section"
```

---

### Task 2: Update compact prompts and tool description

**Files:**
- Modify: `src/llm/client.py:176-193` (tool definition)
- Modify: `src/llm/client.py:753-770` (private compact prompt)
- Modify: `src/llm/client.py:836-846` (group compact prompt)

- [ ] **Step 1: Update `_COMPACT_MEMO_TOOL` description**

In `src/llm/client.py`, update the `note` field description:

```python
_COMPACT_MEMO_TOOL: dict[str, Any] = {
    "name": "append_memo",
    "description": "向用户或群组的备忘录追加新观察（自动写入「待整理」区域）。不要重复已有内容，只记新信息。",
    "input_schema": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "目标 memo ID，如 user_QQ号 或 group_群号",
            },
            "note": {
                "type": "string",
                "description": "要追加的新观察（一句话简短结论，系统自动加 bullet 前缀）",
            },
        },
        "required": ["id", "note"],
    },
}
```

- [ ] **Step 2: Update private compact system prompt**

In the private compact prompt (around line 754), change the memo instruction to mention the pending section and bullet format:

```python
if self._memo_store and user_id:
    system = [{"type": "text", "text": (
        "你是一个对话压缩助手。请完成两个任务：\n"
        "1. 将以下对话历史压缩成简洁的中文摘要。"
        "保留关键信息：用户的问题、重要决策、关键结论、用户偏好。"
        "去掉寒暄、重复内容和过程性细节。\n"
        "2. 如果对话中出现了关于用户的新信息（性格、偏好、背景等），"
        f"用 append_memo 工具追加到 user_{user_id}。"
        "每条 note 写一句话结论，系统会自动放入「待整理」区域。"
        "没有新信息则不需要调用。\n"
        "备忘录规则：只记新的印象和结论，不记流水账。\n"
        "最终请输出纯摘要文本（不要加标题或格式）。"
    )}]
```

- [ ] **Step 3: Update group compact system prompt**

Similar change in the group compact prompt (around line 836):

```python
system = [{"type": "text", "text": (
    "你是一个对话分析助手。请完成两个任务：\n"
    "1. 将以下群聊记录压缩成简洁的中文摘要。保留关键信息。\n"
    "2. 如果对话中出现了关于用户或群组的新信息（性格、偏好、关系、群氛围等），"
    "用 append_memo 工具追加新观察。每条 note 写一句话结论，"
    "系统会自动放入「待整理」区域。没有新信息则不需要调用。\n\n"
    f"本群 ID: group_{group_id}\n"
    f"出现的用户 ID: {', '.join(f'user_{uid}' for uid in seen_user_ids)}\n\n"
    "备忘录规则：用 @QQ号 标注人物，#群号 标注群。QQ号是唯一身份标识，昵称不可信。\n"
    "只记新的印象和结论，不记流水账。\n"
    "最终请输出纯摘要文本（不要加标题或格式）。"
)}]
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_client.py -v -x`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/llm/client.py
git commit -m "feat: update compact prompts for pending-section workflow"
```

---

### Task 3: Dream pre-check and prompt update

**Files:**
- Modify: `src/llm/dream.py:12-33` (pre_check)
- Modify: `src/llm/dream.py:83-104` (dream prompt)
- Test: `tests/test_dream.py`

- [ ] **Step 1: Write test for pre-check detecting pending items**

Add to `tests/test_dream.py`:

```python
@pytest.fixture
async def pending_store(tmp_path) -> MemoStore:
    s = MemoStore(base_dir=str(tmp_path / "pending"))
    await s.startup()
    await s.write("user_100", "@100(测试)\n身份: 学生\n\n## 待整理\n- 喜欢音乐\n- 考试周", "test")
    return s


def test_pre_check_detects_pending_items(pending_store: MemoStore) -> None:
    issues = dream_pre_check(pending_store, user_max_chars=500, group_max_chars=500)
    assert any("待整理" in issue for issue in issues)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dream.py::test_pre_check_detects_pending_items -v`
Expected: FAIL — current pre_check doesn't look for pending sections.

- [ ] **Step 3: Add pending detection to `dream_pre_check()`**

In `src/llm/dream.py`, add the import and check inside the loop:

```python
from src.memory.memo_store import PENDING_HEADER
```

Inside the `for mid in all_ids:` loop, after the oversized check:

```python
if PENDING_HEADER in memo.body:
    pending_lines = [
        l for l in memo.body.split("\n")
        if l.strip().startswith("- ") and memo.body.index(l) > memo.body.index(PENDING_HEADER)
    ]
    if pending_lines:
        issues.append(f"{mid}: has {len(pending_lines)} pending items to merge")
```

Actually, a simpler approach — just check if there's content after the header:

```python
if PENDING_HEADER in memo.body:
    after_header = memo.body.split(PENDING_HEADER, 1)[1].strip()
    if after_header:
        n = sum(1 for line in after_header.splitlines() if line.strip().startswith("- "))
        issues.append(f"{mid}: has {n} pending item(s) in 待整理 to merge")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_dream.py -v`
Expected: All pass.

- [ ] **Step 5: Update dream system prompt**

In `src/llm/dream.py`, update `_run()` to include merge instructions:

```python
system_prompt = (
    "你是记忆整理助手。以下是当前索引：\n"
    f"{index_text}\n\n"
    f"预检发现以下问题：\n{issues_text}\n\n"
    "请依次处理：\n"
    "1. 有「待整理」项的备忘录：用 recall_memo 读取完整内容，"
    "将待整理的条目合并到主体对应位置，清空「## 待整理」区域，"
    "用 update_memo 写回。\n"
    "2. 其他问题（悬空引用、超长等）：自主检查并修复。\n\n"
    "备忘录模板参考：\n"
    "用户备忘录：身份/性格/关系/备注（可选字段，空则省略）\n"
    "群备忘录：成员/话题/事件/规矩（用 ### 分区）\n\n"
    f"限制：最多 {self._max_rounds} 轮工具调用。"
)
```

- [ ] **Step 6: Run full test suite**

Run: `uv run pytest -x -q`
Expected: All pass.

- [ ] **Step 7: Commit**

```bash
git add src/llm/dream.py tests/test_dream.py
git commit -m "feat: dream detects and merges pending memo items"
```

---

### Task 4: Update `soul/instruction.md` with templates

**Files:**
- Modify: `soul/instruction.md:99-141`

- [ ] **Step 1: Update the 记忆系统 section**

Replace the `### 书写规则` section (lines ~131-141) with updated template-aware rules:

```markdown
### 书写规则

- 用 @QQ号 标注人物，首次提及带昵称：@123456(小明)
- 用 #群号 标注群：#987654
- 永远不要只写昵称不写QQ号
- 记印象和结论，不记流水账
- 个人信息写用户备忘录（user_QQ号），群体事件写群备忘录（group_群号）
- 用户备忘录 300 字以内，群备忘录 500 字以内
- 不记你自己说了什么
- 不要用<br>作为换行符

### 备忘录模板

用户备忘录（user_QQ号）：

```
@QQ号(昵称)
身份: 职业/角色，注明自称还是已确认
性格: 说话风格、行为特征
关系: 与其他用户的关系
备注:
- 值得记录的事件或印象

## 待整理
```

群备忘录（group_群号）：

```
### 成员
- @QQ号(昵称): 角色/特征

### 话题
- 群内常见话题或重要讨论

### 事件
- 值得记录的群事件

### 规矩
- 群内约定或潜规则

## 待整理
```

模板说明：
- 字段可选——没有信息的字段直接省略，不要写空占位
- `## 待整理` 是系统保留区域，由 compact 自动追加，dream 自动整理合并，你不需要手动写入
- 更新备忘录时保持已有结构，把新信息放到对应字段
```

- [ ] **Step 2: Verify no syntax issues**

Run: `uv run pytest -x -q`
Expected: All pass (instruction.md changes don't affect tests).

- [ ] **Step 3: Commit**

```bash
git add soul/instruction.md
git commit -m "docs: add memo templates with pending section to instruction"
```

---

### Task 5: Lint, type-check, final verification

- [ ] **Step 1: Run lint**

Run: `uv run ruff check src/ tests/`
Expected: All checks passed.

- [ ] **Step 2: Run type check**

Run: `uv run pyright src/ tests/`
Expected: 0 errors.

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest -v`
Expected: All pass, no regressions.
