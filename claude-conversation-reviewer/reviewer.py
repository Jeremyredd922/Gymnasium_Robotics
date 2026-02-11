"""CLI entry point for Claude Code Conversation Reviewer."""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

from conversation import (
    load_history,
    find_session_file,
    parse_session,
    recent_sessions,
    search_sessions,
)
from formatter import (
    format_list,
    format_summary,
    format_summary_compact,
    format_search_results,
)


def get_claude_dir() -> Path:
    """Resolve the Claude Code data directory."""
    claude_dir = Path.home() / ".claude"
    if not claude_dir.exists():
        print(f"Claude Code data directory not found at {claude_dir}.", file=sys.stderr)
        sys.exit(1)
    return claude_dir


def cmd_list(args):
    """Handle the 'list' command."""
    claude_dir = get_claude_dir()
    sessions = recent_sessions(claude_dir, limit=args.limit)
    print(format_list(sessions, args.limit))


def cmd_summary(args):
    """Handle the 'summary' command."""
    claude_dir = get_claude_dir()
    try:
        filepath = find_session_file(claude_dir, args.session_id)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    session = parse_session(filepath)
    print(format_summary(session))


def cmd_summary_all(args):
    """Handle the 'summary-all' command."""
    claude_dir = get_claude_dir()
    sessions_info = recent_sessions(claude_dir, limit=args.limit)

    if args.days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)
        sessions_info = [
            s for s in sessions_info
            if s.get("first_time") and s["first_time"].tzinfo and s["first_time"] >= cutoff
        ]

    if not sessions_info:
        print("No conversations found.")
        return

    print(f"=== Summary of {len(sessions_info)} Recent Conversations ===\n")

    for info in sessions_info:
        try:
            filepath = find_session_file(claude_dir, info["session_id"][:8])
            session = parse_session(filepath)
            print(format_summary_compact(session))
            print()
        except (FileNotFoundError, ValueError) as e:
            print(f"  [{info['session_id'][:8]}] Error: {e}", file=sys.stderr)


def cmd_search(args):
    """Handle the 'search' command."""
    claude_dir = get_claude_dir()
    results = search_sessions(claude_dir, args.query, limit=20)
    print(format_search_results(results))


def main():
    parser = argparse.ArgumentParser(description="Review Claude Code conversations")
    subparsers = parser.add_subparsers(dest="command")

    # list
    list_parser = subparsers.add_parser("list", help="List recent conversations")
    list_parser.add_argument("--limit", type=int, default=10, help="Number of sessions to show")

    # summary
    summary_parser = subparsers.add_parser("summary", help="Summarize a single conversation")
    summary_parser.add_argument("session_id", type=str, help="Session ID prefix (at least 8 chars)")

    # summary-all
    all_parser = subparsers.add_parser("summary-all", help="Summarize all recent conversations")
    all_parser.add_argument("--limit", type=int, default=5, help="Number of sessions")
    all_parser.add_argument("--days", type=int, default=None, help="Only sessions within last D days")

    # search
    search_parser = subparsers.add_parser("search", help="Search conversations")
    search_parser.add_argument("query", type=str, help="Search query (case-insensitive)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "list": cmd_list,
        "summary": cmd_summary,
        "summary-all": cmd_summary_all,
        "search": cmd_search,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
