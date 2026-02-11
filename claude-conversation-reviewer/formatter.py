"""Output formatting and display for Claude Code conversation reviewer."""

from datetime import datetime, timedelta


def truncate(text: str, length: int) -> str:
    """Truncate text with ellipsis."""
    text = text.replace("\n", " ").strip()
    if len(text) <= length:
        return text
    return text[: length - 3] + "..."


def _format_number(n: int) -> str:
    """Format a number with comma separators."""
    return f"{n:,}"


def _format_duration(start, end) -> str:
    """Format a duration as human-readable string."""
    if not start or not end:
        return "unknown"
    delta = end - start
    total_seconds = int(delta.total_seconds())
    if total_seconds < 60:
        return f"{total_seconds} seconds"
    minutes = total_seconds // 60
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    hours = minutes // 60
    remaining_min = minutes % 60
    if remaining_min:
        return f"{hours} hour{'s' if hours != 1 else ''} {remaining_min} min"
    return f"{hours} hour{'s' if hours != 1 else ''}"


def _format_datetime(dt) -> str:
    """Format a datetime for display."""
    if not dt:
        return "unknown"
    return dt.strftime("%Y-%m-%d %H:%M")


def _shorten_project(project: str, max_len: int = 35) -> str:
    """Shorten a project path for display."""
    if len(project) <= max_len:
        return project
    return "..." + project[-(max_len - 3):]


def format_list(sessions: list, limit: int) -> str:
    """Format the 'list' command output as a table."""
    if not sessions:
        return "No conversations found."

    lines = ["Recent conversations:"]
    lines.append(f"  {'#':>3}  {'ID':<10}  {'Date':<18}  {'Project':<35}  {'Preview'}")
    lines.append("  " + "-" * 95)

    for i, s in enumerate(sessions[:limit], 1):
        sid = s["session_id"][:8]
        dt = _format_datetime(s.get("first_time"))
        project = _shorten_project(s.get("project", ""))
        preview = truncate(s.get("display", "(no preview)"), 40)
        msg_count = s.get("message_count", 0)
        lines.append(f"  {i:>3}  {sid:<10}  {dt:<18}  {project:<35}  {preview}")

    return "\n".join(lines)


def format_summary(session) -> str:
    """Format a full session summary with all 5 sections."""
    lines = []

    # Section 1 - Metadata
    lines.append("=== Conversation Summary ===")
    lines.append("")
    lines.append(f"Session:    {session.session_id}")
    lines.append(f"Project:    {session.project}")

    date_range = _format_datetime(session.start_time)
    if session.end_time and session.start_time:
        duration = _format_duration(session.start_time, session.end_time)
        date_range += f" — {_format_datetime(session.end_time)} ({duration})"
    lines.append(f"Date:       {date_range}")
    lines.append(f"Version:    {session.version}")
    lines.append(f"Model:      {session.model_used()}")
    lines.append(f"Branch:     {session.git_branch()}")

    # Section 2 - Token Usage
    tokens = session.total_tokens()
    lines.append("")
    lines.append("--- Token Usage ---")
    lines.append(f"Input:        {_format_number(tokens['input_tokens']):>10}")
    lines.append(f"Output:       {_format_number(tokens['output_tokens']):>10}")
    lines.append(f"Cache read:   {_format_number(tokens['cache_read']):>10}")
    lines.append(f"Cache create: {_format_number(tokens['cache_creation']):>10}")

    # Section 3 - Conversation Flow
    lines.append("")
    lines.append("--- Conversation Flow ---")
    pairs = session.turn_pairs()
    if not pairs:
        lines.append("(No user-assistant exchanges found)")
    else:
        for i, (user_msg, asst_msg) in enumerate(pairs, 1):
            user_text = truncate(user_msg.text_content, 120)
            lines.append(f"{i:>2}. [USER]  {user_text}")

            # Describe what assistant did
            asst_desc = _describe_assistant_action(asst_msg)
            lines.append(f"    [ASST]  {asst_desc}")

            if asst_msg.tool_calls:
                tool_names = ", ".join(tc["name"] for tc in asst_msg.tool_calls)
                lines.append(f"    Tools:  {tool_names}")
            lines.append("")

    # Section 4 - Tools Used
    tool_counts = session.tool_summary()
    lines.append("--- Tools Used ---")
    if not tool_counts:
        lines.append("(No tools used)")
    else:
        lines.append(f"{'Tool':<25} {'Count':>5}")
        for name, count in tool_counts.items():
            lines.append(f"{name:<25} {count:>5}")

    # Section 5 - Files Touched
    files = session.files_touched()
    lines.append("")
    lines.append("--- Files Touched ---")
    if not files:
        lines.append("(No files detected)")
    else:
        for f in files:
            lines.append(f"  {f}")

    return "\n".join(lines)


def _describe_assistant_action(msg) -> str:
    """Generate a one-line description of what the assistant did."""
    parts = []
    if msg.text_content.strip():
        text_preview = truncate(msg.text_content, 60)
        parts.append(f"Responded: {text_preview}")
    for tc in msg.tool_calls:
        summary = tc.get("input_summary", "")
        if summary:
            parts.append(f"{tc['name']}: {truncate(summary, 50)}")
        else:
            parts.append(f"Called {tc['name']}")
    if not parts:
        return "(empty response)"
    return "; ".join(parts)


def format_summary_compact(session) -> str:
    """Format a one-paragraph condensed summary."""
    tokens = session.total_tokens()
    total_tok = tokens["input_tokens"] + tokens["output_tokens"]
    tool_counts = session.tool_summary()
    tools_str = ", ".join(tool_counts.keys()) if tool_counts else "none"

    user_msgs = session.user_messages()
    topic = truncate(user_msgs[0].text_content, 80) if user_msgs else "(no messages)"

    dt = _format_datetime(session.start_time)
    msg_count = len(session.messages)

    return (
        f"[{session.session_id[:8]}] {dt} | {session.project} | "
        f'"{topic}" | '
        f"{msg_count} messages, {_format_number(total_tok)} tokens | "
        f"Tools: {tools_str}"
    )


def format_search_results(results: list) -> str:
    """Format search hits with context."""
    if not results:
        return "No results found."

    lines = [f"Found {len(results)} result(s):", ""]

    for i, r in enumerate(results, 1):
        sid = r["session_id"][:8]
        role = r.get("role", "?")
        ts = r.get("timestamp", "")[:16]
        context = r.get("context", "")
        lines.append(f"  {i:>2}. [{sid}] ({role}) {ts}")
        lines.append(f"      {context}")
        lines.append("")

    return "\n".join(lines)
