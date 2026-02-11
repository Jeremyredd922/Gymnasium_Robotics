# Claude Conversation Reviewer - Implementation Plan

## Goal
Build a Python CLI tool that reads local Claude Code conversation logs (`~/.claude/`) and produces human-readable summaries. Uses only Python standard libraries.

## Architecture

Three modules in `claude-conversation-reviewer/`:

| File | Purpose |
|------|---------|
| `conversation.py` | Data models (`Message`, `Session`) and parsing logic for JSONL files |
| `formatter.py` | Output formatting functions for all display modes |
| `reviewer.py` | CLI entry point using `argparse` with subcommands |

## Commands

| Command | Description |
|---------|-------------|
| `python reviewer.py list [--limit N]` | List recent conversations from `~/.claude/history.jsonl` |
| `python reviewer.py summary <id-prefix>` | Full 5-section summary of a single conversation |
| `python reviewer.py summary-all [--limit N] [--days D]` | Condensed summary of recent conversations |
| `python reviewer.py search <query>` | Case-insensitive search across all conversations |

## Key Design Decisions

- **Standard library only** — no pip dependencies (`json`, `pathlib`, `argparse`, `datetime`, `collections`)
- **Windows-compatible** — uses `pathlib.Path` everywhere, resolves `~` via `Path.home()`
- **Lazy loading** — only reads full session JSONL files when detail is requested
- **Handles both timestamp formats** — ISO 8601 strings and Unix epoch integers (milliseconds)
- **Graceful error handling** — malformed JSON skipped with warnings, missing dirs get clear messages, ambiguous session prefixes listed

## Data Flow

1. `history.jsonl` provides a lightweight index (session ID, timestamp, project, preview)
2. Individual session files (`~/.claude/projects/<path>/<sessionId>.jsonl`) contain full transcripts
3. Each JSONL line is parsed into a `Message` object; grouped into a `Session`
4. Formatter functions convert `Session` objects into display strings

## Status

- [x] `conversation.py` — Message/Session classes, parsing, search
- [x] `formatter.py` — list table, full summary, compact summary, search results
- [x] `reviewer.py` — CLI with all 4 subcommands
- [x] Tested against real conversation data — all commands working
