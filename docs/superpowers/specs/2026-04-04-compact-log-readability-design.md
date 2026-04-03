# Compact Log Readability

## Goal

Make compact (context compression) operations fully observable at INFO level, so the operator can see **why** compact triggered and **what** it did, without digging into DEBUG logs.

## Current State

- Compact start: `compact | session=... split=X/Y` — has split ratio but no trigger reason (token counts)
- Compact done: `compact done | session=... summary_len=N` — missing message counts, memo writes, elapsed time
- Same pattern for `compact_group`

## Changes

### 1. Add trigger-reason log before compact

At the two call sites in `chat()` where `needs_compact` returns true, log an INFO line with the actual input tokens and the threshold that was exceeded:

```
compact triggering | group=123 input_tokens=48000 threshold=40000
compact triggering | session=private_456 input_tokens=48000 threshold=40000
```

`threshold = max_context_tokens * compact_ratio`

### 2. Enrich compact-done log

Change `_compact_with_tools` to return `(summary_text, memo_writes)` instead of just `summary_text`.

In `_compact` and `_compact_group`, add timing and merge all info into the done log:

```
compact done | session=private_456 messages=20->10 summary_len=300 memo_writes=2 elapsed=1.2s
compact_group done | group=123 messages=20->10 summary_len=300 memo_writes=2 elapsed=1.2s
```

### 3. Files changed

Only `src/llm/client.py`:
- `chat()` — two trigger-reason logs (group + private paths)
- `_compact_with_tools()` — track and return `memo_writes` count
- `_compact()` — add timing, use new return value, enrich done log
- `_compact_group()` — same as above
