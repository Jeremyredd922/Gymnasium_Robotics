# Claude Code Conversation Reviewer - Specification

## Overview

A Python CLI tool that reads local Claude Code conversation logs and produces human-readable summaries. Uses **only Python standard libraries** (no pip installs).

---

## Data Source

Claude Code stores conversations as JSONL files on disk:

| File | Purpose |
|------|---------|
| `~/.claude/history.jsonl` | Global index — one JSON object per conversation turn with `sessionId`, `timestamp`, `project`, and `display` (first message preview) |
| `~/.claude/projects/<sanitized-path>/<sessionId>.jsonl` | Full conversation transcript — every message (user, assistant, tool use, tool result) as one JSON object per line |

### Key record types inside a session JSONL

| `type` field | Description |
|--------------|-------------|
| `user` | User message. Content in `message.content` (string). |
| `assistant` | Assistant response. Content in `message.content` (list of `{type, text}` and/or `{type: "tool_use", name, input}` blocks). Token usage in `message.usage`. |
| `tool_result` | Result of a tool invocation. Contains `toolUseResult` and `sourceToolAssistantUUID`. |
| `progress` | Streaming progress updates (can be ignored for summaries). |
| `file-history-snapshot` | File backup metadata (can be ignored). |

### Useful fields per record

```
timestamp          — ISO 8601 string (e.g. "2026-02-11T19:41:14.098Z")
sessionId          — UUID identifying the conversation session
uuid / parentUuid  — message threading
cwd                — working directory at time of message
gitBranch          — active git branch
version            — Claude Code version
message.role       — "user" or "assistant"
message.content    — the actual text / tool calls
message.usage      — token counts (input_tokens, output_tokens, cache_*)
```

---

## Functional Requirements

### FR-1: List recent conversations

```
python reviewer.py list [--limit N]
```

- Read `~/.claude/history.jsonl`.
- Group entries by `sessionId`.
- For each session, show:
  - Session ID (truncated to first 8 chars)
  - Date/time of first message
  - Project path
  - First user message (truncated to 80 chars)
  - Message count
- Default `--limit` is 10. Sort by most recent first.

**Example output:**
```
Recent conversations:
  #  ID        Date                 Project                          Preview
  1  75d2e461  2026-02-11 14:41     .../DePaul/VibeCoding            Help create a specification fo...
  2  b0a994a7  2026-02-10 09:12     .../DePaul/VibeCoding            Fix the login bug in...
  3  a1c3f820  2026-02-09 20:30     C:\Users\mlbpi                   What files handle routing?
```

### FR-2: Summarize a single conversation

```
python reviewer.py summary <sessionId-prefix>
```

- Locate the session JSONL file by matching the prefix against filenames in all `~/.claude/projects/*/` directories.
- Parse every record and produce:

**Section 1 — Metadata**
- Full session ID
- Project path
- Date range (first message → last message)
- Duration (human-readable, e.g. "23 minutes")
- Claude Code version
- Model used
- Git branch

**Section 2 — Token Usage**
- Total input tokens
- Total output tokens
- Total cache read tokens
- Total cache creation tokens

**Section 3 — Conversation Flow**
A numbered list of exchanges. For each user→assistant turn:
- The user's message (truncated to 120 chars)
- A one-line description of what the assistant did (responded with text, called tool X, edited file Y, etc.)
- Tools invoked (names only, as a comma-separated list)

**Section 4 — Tools Used**
- Table of tool names with invocation counts, sorted by frequency.

**Section 5 — Files Touched**
- Deduplicated list of file paths that appeared in tool calls (Read, Edit, Write, Glob targets, Bash commands referencing files).

### FR-3: Summarize all recent conversations

```
python reviewer.py summary-all [--limit N] [--days D]
```

- Produce a condensed version of FR-2 for each of the N most recent sessions (or sessions within the last D days).
- Output: one paragraph per session containing date, project, topic (derived from first user message), message count, total tokens, and tools used.

### FR-4: Search conversations

```
python reviewer.py search <query>
```

- Case-insensitive substring search across all user and assistant text content in all session files.
- Return matching sessions with the matching line highlighted (truncated with context).
- Limit to 20 results by default.

---

## Non-Functional Requirements

### NFR-1: Standard library only
Only use modules from the Python standard library: `json`, `os`, `pathlib`, `sys`, `argparse`, `datetime`, `re`, `collections`, `textwrap`, `glob`.

### NFR-2: Platform compatibility
Must work on Windows. Use `pathlib.Path` for all path operations. Resolve `~` via `Path.home()`.

### NFR-3: Performance
- Lazy-load session files — only read full JSONL files when the user requests detail on a specific session.
- `history.jsonl` is small enough to load fully into memory.

### NFR-4: Error handling
- If `~/.claude/` does not exist, print a clear error: "Claude Code data directory not found at <path>."
- If a session ID prefix matches multiple sessions, list the ambiguous matches and ask the user to be more specific.
- Malformed JSON lines should be skipped with a warning to stderr.

---

## Architecture

```
claude-conversation-reviewer/
    reviewer.py          # CLI entry point (argparse)
    conversation.py      # Data models and parsing logic
    formatter.py         # Output formatting / display
```

### Module: `conversation.py`

**Classes:**

```python
class Message:
    """Single parsed message from a session JSONL line."""
    uuid: str
    parent_uuid: str | None
    timestamp: datetime
    role: str              # "user", "assistant", "tool_result", etc.
    text_content: str      # extracted plain text (empty for tool-only messages)
    tool_calls: list[dict] # list of {name, input_summary} for tool_use blocks
    tool_result: str | None
    token_usage: dict      # {input_tokens, output_tokens, cache_read, cache_creation}
    session_id: str
    cwd: str
    git_branch: str
    model: str | None

class Session:
    """A full conversation session."""
    session_id: str
    project: str
    messages: list[Message]
    start_time: datetime
    end_time: datetime
    version: str

    def user_messages() -> list[Message]
    def assistant_messages() -> list[Message]
    def tool_summary() -> dict[str, int]     # tool_name -> count
    def files_touched() -> list[str]
    def total_tokens() -> dict
    def turn_pairs() -> list[tuple[Message, Message]]  # (user, assistant) pairs
```

**Functions:**

```python
def load_history(claude_dir: Path) -> list[dict]
    """Parse history.jsonl into a list of index records."""

def find_session_file(claude_dir: Path, session_prefix: str) -> Path
    """Locate a session JSONL file by ID prefix. Raises if ambiguous or not found."""

def parse_session(filepath: Path) -> Session
    """Parse a full session JSONL file into a Session object."""

def recent_sessions(claude_dir: Path, limit: int) -> list[dict]
    """Return the N most recent session summaries from history.jsonl."""
```

### Module: `formatter.py`

**Functions:**

```python
def format_list(sessions: list[dict], limit: int) -> str
    """Format the 'list' command output as a table."""

def format_summary(session: Session) -> str
    """Format a full session summary with all 5 sections."""

def format_summary_compact(session: Session) -> str
    """Format a one-paragraph condensed summary."""

def format_search_results(results: list[dict]) -> str
    """Format search hits with context."""

def truncate(text: str, length: int) -> str
    """Truncate text with ellipsis."""
```

### Module: `reviewer.py`

```python
def main():
    parser = argparse.ArgumentParser(description="Review Claude Code conversations")
    subparsers = parser.add_subparsers(dest="command")

    # list
    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--limit", type=int, default=10)

    # summary
    summary_parser = subparsers.add_parser("summary")
    summary_parser.add_argument("session_id", type=str)

    # summary-all
    all_parser = subparsers.add_parser("summary-all")
    all_parser.add_argument("--limit", type=int, default=5)
    all_parser.add_argument("--days", type=int, default=None)

    # search
    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("query", type=str)
```

---

## Example Outputs

### `python reviewer.py summary 75d2e461`

```
=== Conversation Summary ===

Session:    75d2e461-79c3-4e9e-99e9-f5cc494f1b05
Project:    C:\Users\mlbpi\OneDrive\Documents\DePaul\VibeCoding
Date:       2026-02-11 14:41 — 15:03 (22 minutes)
Version:    2.1.37
Model:      claude-opus-4-6
Branch:     master

--- Token Usage ---
Input:        12,450
Output:        3,210
Cache read:   10,431
Cache create:  8,669

--- Conversation Flow ---
1. [USER]  Help create a specification for a program that does thi...
   [ASST]  Explored conversation storage paths, wrote SPEC.md
   Tools:  Task(Explore), Write

--- Tools Used ---
Tool             Count
Task(Explore)        1
Write                1

--- Files Touched ---
claude-conversation-reviewer/SPEC.md
```

---

## Future Enhancements (out of scope for v1)

- Export summaries to Markdown or HTML files
- Conversation diff (compare two sessions)
- Cost estimation based on token counts and model pricing
- Interactive TUI mode using `curses`
