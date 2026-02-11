"""Data models and parsing logic for Claude Code conversation logs."""

import json
import sys
import re
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

_UTC = timezone.utc


class Message:
    """Single parsed message from a session JSONL line."""

    def __init__(self, raw: dict):
        self.uuid = raw.get("uuid", "")
        self.parent_uuid = raw.get("parentUuid")
        self.timestamp = _parse_timestamp(raw.get("timestamp", ""))
        self.type = raw.get("type", "")
        self.session_id = raw.get("sessionId", "")
        self.cwd = raw.get("cwd", "")
        self.git_branch = raw.get("gitBranch", "")
        self.version = raw.get("version", "")

        msg = raw.get("message", {})
        self.role = msg.get("role", self.type)
        self.model = msg.get("model")

        self.text_content = ""
        self.tool_calls = []
        self.tool_result = None
        self.token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read": 0,
            "cache_creation": 0,
        }

        content = msg.get("content", "")
        if isinstance(content, str):
            self.text_content = content
        elif isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        self.tool_calls.append({
                            "name": block.get("name", ""),
                            "input_summary": _summarize_tool_input(
                                block.get("name", ""), block.get("input", {})
                            ),
                        })
                elif isinstance(block, str):
                    text_parts.append(block)
            self.text_content = "\n".join(text_parts)

        if self.type == "tool_result":
            self.tool_result = raw.get("toolUseResult", "")

        usage = msg.get("usage", {})
        if usage:
            self.token_usage = {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "cache_read": usage.get("cache_read_input_tokens", 0),
                "cache_creation": usage.get("cache_creation_input_tokens", 0),
            }


class Session:
    """A full conversation session."""

    def __init__(self, session_id: str, project: str, messages: list):
        self.session_id = session_id
        self.project = project
        self.messages = messages

        timestamps = [m.timestamp for m in messages if m.timestamp]
        self.start_time = min(timestamps) if timestamps else None
        self.end_time = max(timestamps) if timestamps else None

        versions = [m.version for m in messages if m.version]
        self.version = versions[0] if versions else "unknown"

    def user_messages(self):
        return [m for m in self.messages if m.role == "user" or m.type == "user"]

    def assistant_messages(self):
        return [m for m in self.messages if m.role == "assistant" or m.type == "assistant"]

    def tool_summary(self):
        counts = Counter()
        for m in self.assistant_messages():
            for tc in m.tool_calls:
                counts[tc["name"]] += 1
        return dict(counts.most_common())

    def files_touched(self):
        files = set()
        for m in self.messages:
            for tc in m.tool_calls:
                name = tc["name"]
                inp = tc.get("input_summary", "")
                if inp:
                    files.add(inp)
            if m.type == "tool_result" and isinstance(m.tool_result, str):
                pass  # tool results don't reliably contain file paths
        return sorted(files)

    def total_tokens(self):
        totals = {"input_tokens": 0, "output_tokens": 0, "cache_read": 0, "cache_creation": 0}
        for m in self.assistant_messages():
            for key in totals:
                totals[key] += m.token_usage.get(key, 0)
        return totals

    def turn_pairs(self):
        pairs = []
        user_msgs = []
        for m in self.messages:
            if m.type in ("progress", "file-history-snapshot"):
                continue
            if m.role == "user" or m.type == "user":
                user_msgs.append(m)
            elif m.role == "assistant" or m.type == "assistant":
                if user_msgs:
                    pairs.append((user_msgs[-1], m))
                    user_msgs = []
        return pairs

    def model_used(self):
        for m in self.assistant_messages():
            if m.model:
                return m.model
        return "unknown"

    def git_branch(self):
        for m in self.messages:
            if m.git_branch:
                return m.git_branch
        return "unknown"


def _parse_timestamp(ts):
    if not ts:
        return None
    try:
        if isinstance(ts, (int, float)):
            # Unix epoch in milliseconds
            if ts > 1e12:
                ts = ts / 1000
            return datetime.fromtimestamp(ts, tz=_UTC)
        cleaned = str(ts).replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned)
    except (ValueError, TypeError, OSError):
        return None


def _summarize_tool_input(tool_name: str, inp: dict) -> str:
    if not isinstance(inp, dict):
        return ""
    if tool_name in ("Read", "Write", "Edit"):
        return inp.get("file_path", inp.get("filePath", ""))
    if tool_name == "Glob":
        return inp.get("pattern", "")
    if tool_name == "Grep":
        return inp.get("pattern", "")
    if tool_name == "Bash":
        cmd = inp.get("command", "")
        return cmd[:80] if cmd else ""
    if tool_name == "Task":
        return inp.get("description", inp.get("prompt", ""))[:80]
    if tool_name == "NotebookEdit":
        return inp.get("notebook_path", "")
    return ""


def load_history(claude_dir: Path) -> list:
    """Parse history.jsonl into a list of index records."""
    history_file = claude_dir / "history.jsonl"
    if not history_file.exists():
        return []
    records = []
    with open(history_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed JSON at history.jsonl line {line_num}", file=sys.stderr)
    return records


def find_session_file(claude_dir: Path, session_prefix: str) -> Path:
    """Locate a session JSONL file by ID prefix. Raises if ambiguous or not found."""
    projects_dir = claude_dir / "projects"
    if not projects_dir.exists():
        raise FileNotFoundError(f"Projects directory not found at {projects_dir}")

    matches = []
    for project_dir in projects_dir.iterdir():
        if not project_dir.is_dir():
            continue
        for jsonl_file in project_dir.glob("*.jsonl"):
            if jsonl_file.stem.startswith(session_prefix):
                matches.append(jsonl_file)

    if len(matches) == 0:
        raise FileNotFoundError(f"No session found matching prefix '{session_prefix}'")
    if len(matches) > 1:
        match_ids = [m.stem[:8] for m in matches]
        raise ValueError(
            f"Ambiguous prefix '{session_prefix}' matches {len(matches)} sessions: "
            + ", ".join(match_ids)
            + "\nPlease be more specific."
        )
    return matches[0]


def parse_session(filepath: Path) -> Session:
    """Parse a full session JSONL file into a Session object."""
    messages = []
    session_id = filepath.stem
    project = filepath.parent.name

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed JSON at {filepath.name} line {line_num}", file=sys.stderr)
                continue

            msg_type = raw.get("type", "")
            if msg_type in ("progress", "file-history-snapshot"):
                continue

            messages.append(Message(raw))

    return Session(session_id, project, messages)


def recent_sessions(claude_dir: Path, limit: int = 10) -> list:
    """Return the N most recent session summaries from history.jsonl."""
    records = load_history(claude_dir)
    if not records:
        return []

    # Group by sessionId
    sessions = {}
    for rec in records:
        sid = rec.get("sessionId", "")
        if not sid:
            continue
        if sid not in sessions:
            sessions[sid] = {
                "session_id": sid,
                "timestamps": [],
                "project": rec.get("project", ""),
                "display": rec.get("display", ""),
                "message_count": 0,
            }
        sessions[sid]["timestamps"].append(rec.get("timestamp", ""))
        sessions[sid]["message_count"] += 1

    # Compute first timestamp for each session and sort
    session_list = []
    for sid, info in sessions.items():
        parsed = [_parse_timestamp(t) for t in info["timestamps"] if t]
        parsed = [t for t in parsed if t is not None]
        if parsed:
            info["first_time"] = min(parsed)
            info["last_time"] = max(parsed)
        else:
            info["first_time"] = datetime.min
            info["last_time"] = datetime.min
        session_list.append(info)

    session_list.sort(key=lambda s: s["first_time"], reverse=True)
    return session_list[:limit]


def search_sessions(claude_dir: Path, query: str, limit: int = 20) -> list:
    """Search all session files for a query string. Returns matching sessions with context."""
    projects_dir = claude_dir / "projects"
    if not projects_dir.exists():
        return []

    query_lower = query.lower()
    results = []

    for project_dir in projects_dir.iterdir():
        if not project_dir.is_dir():
            continue
        for jsonl_file in project_dir.glob("*.jsonl"):
            try:
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            raw = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        msg = raw.get("message", {})
                        content = msg.get("content", "")
                        text = ""
                        if isinstance(content, str):
                            text = content
                        elif isinstance(content, list):
                            parts = []
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    parts.append(block.get("text", ""))
                                elif isinstance(block, str):
                                    parts.append(block)
                            text = "\n".join(parts)

                        if query_lower in text.lower():
                            # Find the match position for context
                            idx = text.lower().index(query_lower)
                            start = max(0, idx - 40)
                            end = min(len(text), idx + len(query) + 40)
                            context = text[start:end].replace("\n", " ")
                            if start > 0:
                                context = "..." + context
                            if end < len(text):
                                context = context + "..."

                            results.append({
                                "session_id": jsonl_file.stem,
                                "project": project_dir.name,
                                "timestamp": raw.get("timestamp", ""),
                                "role": msg.get("role", raw.get("type", "")),
                                "context": context,
                            })

                            if len(results) >= limit:
                                return results
            except (OSError, IOError):
                continue

    return results
