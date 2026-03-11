# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Projects

### claude-conversation-reviewer
A Python CLI tool (standard library only) that reads local Claude Code conversation logs (`~/.claude/`) and produces summaries.
- **Spec**: [claude-conversation-reviewer/SPEC.md](claude-conversation-reviewer/SPEC.md)
- **Plan**: [claude-conversation-reviewer/plan.md](claude-conversation-reviewer/plan.md)
- **Usage**: `cd claude-conversation-reviewer && python reviewer.py list`
- **Commands**: `list [--limit N]`, `summary <id-prefix>`, `summary-all [--limit N] [--days D]`, `search <query>`
- **Architecture**: `reviewer.py` (CLI entry point) -> `conversation.py` (data models/JSONL parsing) -> `formatter.py` (output formatting)

### hw1 (Scala - DePaul CSC347)
Conway's Game of Life implementation in Scala 3.5.0.
- **Requirements**: Java 21 JDK, SBT
- **Build**: `cd hw1 && sbt compile`
- **Test**: `sbt test` or `sbt testOnly fp1tests` (single suite) or `sbt ~testOnly fp1tests` (continuous)
- **IDE**: VSCode + Scala (Metals) extension
- **Note**: `build.sbt` and files in `src/main/java/` are read-only (grading infrastructure)

### flashcard-app
Interactive CLI flashcard study tool with spaced repetition scoring.
- **Usage**: `cd flashcard-app && python flashcards.py`
- **Data**: Persists decks to `decks.json`
- **Dependencies**: Python standard library only

### doom-ascii
Terminal-based Doom powered by ViZDoom with ASCII art rendering.
- **Spec**: [doom-ascii/SPEC.md](doom-ascii/SPEC.md)
- **Prompt log**: [doom-ascii/prompts.log](doom-ascii/prompts.log)
- **Usage**: `cd doom-ascii && python main.py`
- **Options**: `--width N`, `--height N`, `--tics N`, `--scenario PATH`
- **Dependencies**: `pip install vizdoom` (brings numpy; includes freedoom2.wad)
- **Architecture**: `main.py` (game loop + input) → `game.py` (ViZDoom setup) + `ascii_renderer.py` (frame → ASCII)

### Pygame games (root level)
- `python snake_game.py` - Classic Snake game
- `python platformer.py` - Platformer with physics, platforms, particles
- **Dependency**: `pip install pygame`
