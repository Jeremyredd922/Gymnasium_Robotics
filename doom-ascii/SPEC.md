# SPEC.md — doom-ascii

## Project Overview
A terminal-based Doom experience powered by ViZDoom, rendering the game's 3D
framebuffer as ASCII art in real time. The goal is a lightweight, playable Doom
that runs entirely in the terminal with no graphical window.

## Goals
1. Use ViZDoom as the Doom engine backend (headless mode)
2. Capture the screen buffer every frame and convert it to ASCII art
3. Accept keyboard input (WASD + mouse-style turning, space/attack, E/use)
4. Display game stats (health, ammo, kills) as a text HUD
5. Keep dependencies minimal: `vizdoom`, `numpy`

## Non-Goals
- Multiplayer / networking
- Saving/loading game state
- Sound support
- ANSI color rendering (stretch goal, not required for v1)

## Architecture

```
doom-ascii/
  main.py           Entry point. Game loop, input, timing.
  ascii_renderer.py Frame buffer → ASCII string conversion.
  game.py           ViZDoom initialization, scenario config, game vars.
  SPEC.md           This file.
  prompts.log       Running log of every user prompt for this project.
```

### Data flow
```
ViZDoom engine
     │
     ▼  state.screen_buffer  (numpy array, GRAY8: H×W)
ascii_renderer.py
     │  downscale + char-map
     ▼
Terminal (stdout)   + text HUD line
```

## Component Details

### game.py
- Creates a `DoomGame` instance
- Tries `freedoom2.wad` (ships with ViZDoom ≥ 1.2); falls back to `deadly_corridor.wad`
- Screen format: `GRAY8` (grayscale, avoids channel-order ambiguity)
- Screen resolution: `320×240` (low-res → faster to process)
- `window_visible = False` (headless — we own the display)
- Available buttons: MOVE_LEFT, MOVE_RIGHT, MOVE_FORWARD, MOVE_BACKWARD,
  TURN_LEFT, TURN_RIGHT, ATTACK, USE
- Available game variables: HEALTH, AMMO2, KILLCOUNT
- Mode: `PLAYER`

### ascii_renderer.py
- Detects terminal size via `os.get_terminal_size()`
- Downscales frame to `(cols, cols × 0.375)` accounting for character aspect ratio
- Maps brightness 0–255 → index into ASCII ramp (dark → light):
  `" .`-_':,;^=+!|\\1tilI?rJjfFxX2YV3nuvczUJCL0OZmwqpdbkhao*#M&8%B@"`
- Uses numpy for fast array operations (no PIL dependency)
- Renders with ANSI cursor positioning (`\033[H`) to minimize flicker
- Hides cursor during play (`\033[?25l`), restores on exit (`\033[?25h`)

### main.py
- Background thread reads keys via `msvcrt.kbhit()` / `msvcrt.getch()` (Windows)
  or `tty`/`termios` non-blocking read (Unix)
- Key map:
  | Key          | Action        |
  |--------------|---------------|
  | W            | MOVE_FORWARD  |
  | S            | MOVE_BACKWARD |
  | A            | TURN_LEFT     |
  | D            | TURN_RIGHT    |
  | Q / ←        | TURN_LEFT     |
  | E / →        | TURN_RIGHT    |
  | Space        | ATTACK        |
  | F            | USE           |
  | R            | New episode   |
  | Ctrl+C / Esc | Quit          |
- Key repeat: last key stays active for 150 ms (smooth movement)
- Game loop: `make_action(action, tics=2)` → 17.5 fps (~35 tics / 2)

## Dependencies
```
vizdoom   >= 1.2.0   (includes freedoom2.wad, numpy)
numpy               (pulled in by vizdoom)
```

Install: `pip install vizdoom`

## Running
```bash
cd doom-ascii
python main.py
# Options:
#   --width N     override terminal columns (default: auto-detect)
#   --height N    override terminal rows (default: auto-detect)
#   --tics N      ViZDoom tics per frame (default: 2)
#   --scenario S  path to .wad scenario file (default: auto)
```

## Stretch Goals (v2)
- ANSI 256-color / truecolor rendering using original RGB buffer
- Mouse support for smooth turning
- Minimap in corner using block characters
- Demo recording / playback
- Custom scenario support

## Prompt Log
All prompts used to build this project are recorded in `prompts.log`.
