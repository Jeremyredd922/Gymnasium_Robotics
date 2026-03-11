"""
main.py — doom-ascii AI spectator mode.

An AI agent plays Doom while you watch.  Choose your renderer at startup:
  • Graphics — pygame window with full RGB visuals (requires: pip install pygame)
  • ASCII    — terminal art, no extra dependencies

Press Ctrl+C / ESC / close window to quit at any time.

Usage:
    python main.py                            # interactive startup menu
    python main.py --renderer graphics        # skip menu
    python main.py --renderer ascii           # skip menu
    python main.py --renderer graphics --agent wander --scale 2
    python main.py --renderer ascii    --agent heuristic --fps 15

Options:
    --renderer  {graphics,ascii}           Renderer to use
    --agent     {random,wander,heuristic}  AI agent (default: heuristic)
    --episodes  N                          Stop after N episodes (0 = infinite)
    --tics      N                          ViZDoom tics per step (default: 2)
    --fps       N                          ASCII target FPS (default: 15)
    --scale     N                          Graphics window scale (default: 3 → 960×720)
    --width     N                          ASCII column override
    --height    N                          ASCII row override
    --scenario  PATH                       Path to a .wad file (default: auto)
"""
import argparse
import sys
import time

from game import setup_game, read_stats
from agent import build_agent, AGENTS


# ---------------------------------------------------------------------------
# Startup menu
# ---------------------------------------------------------------------------

def _prompt(question: str, options: list, default: int) -> str:
    """Print a numbered list and return the chosen option string."""
    for i, opt in enumerate(options, 1):
        marker = " [default]" if i == default else ""
        print(f"    {i}. {opt}{marker}")
    while True:
        raw = input(f"  > ").strip()
        if raw == "":
            return options[default - 1]
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print(f"  Please enter a number 1–{len(options)}.")


def show_startup_menu(args):
    """Interactively fill in args.renderer and args.agent if not set via CLI."""
    if args.renderer and args.agent:
        return  # fully specified on command line

    print()
    print("=" * 52)
    print("  DOOM ASCII — AI Spectator")
    print("=" * 52)

    if not args.renderer:
        print("\n  Renderer:")
        args.renderer = _prompt("", ["graphics", "ascii"], default=1)

    if not args.agent:
        print("\n  AI Agent:")
        args.agent = _prompt("", list(AGENTS.keys()), default=1)

    print()


# ---------------------------------------------------------------------------
# Renderer factory
# ---------------------------------------------------------------------------

def build_renderer(args):
    if args.renderer == "graphics":
        from graphics_renderer import GraphicsRenderer
        return GraphicsRenderer(scale=args.scale)
    else:
        from ascii_renderer import AsciiRenderer
        return AsciiRenderer(width=args.width, height=args.height)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="doom-ascii: watch an AI play Doom"
    )
    p.add_argument("--renderer", default=None,
                   choices=["graphics", "ascii"],
                   help="Renderer: 'graphics' (pygame) or 'ascii' (terminal)")
    p.add_argument("--agent",    default=None,
                   choices=list(AGENTS),
                   help="AI agent (default: heuristic)")
    p.add_argument("--episodes", type=int, default=0,
                   help="Episodes to run (0 = infinite)")
    p.add_argument("--tics",     type=int, default=2,
                   help="ViZDoom tics per agent step (default: 2)")
    p.add_argument("--fps",      type=int, default=15,
                   help="Target FPS for ASCII renderer (default: 15)")
    p.add_argument("--scale",    type=int, default=3,
                   help="Window scale for graphics renderer (default: 3 → 960×720)")
    p.add_argument("--width",    type=int, default=None,
                   help="ASCII column override")
    p.add_argument("--height",   type=int, default=None,
                   help="ASCII row override")
    p.add_argument("--scenario", type=str, default=None,
                   help="Path to a .wad file (default: auto-detect)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    show_startup_menu(args)

    # Defaults that menu may have left unset.
    if not args.renderer:
        args.renderer = "graphics"
    if not args.agent:
        args.agent = "heuristic"

    renderer = build_renderer(args)
    renderer.show_loading()

    color = (args.renderer == "graphics")
    try:
        game = setup_game(scenario_path=args.scenario, color=color)
    except Exception as exc:
        renderer.close()
        sys.stdout.write(f"\nFailed to start ViZDoom: {exc}\n")
        sys.stdout.write("Install with:  pip install vizdoom\n")
        sys.exit(1)

    agent = build_agent(args.agent)

    renderer.hide_cursor()
    renderer.show_agent_banner(args.agent)
    time.sleep(1.5)

    frame_duration = 1.0 / max(1, args.fps)   # used by ASCII renderer only
    episode_num    = 0
    total_kills    = 0

    try:
        while True:
            episode_num += 1
            if args.episodes and episode_num > args.episodes:
                break

            game.new_episode()
            agent.reset()

            ep_start = time.monotonic()
            ep_kills  = 0
            running   = True   # set to False when renderer signals quit

            while not game.is_episode_finished() and running:
                t0 = time.monotonic()

                state = game.get_state()
                health, ammo, kills = read_stats(state)
                ep_kills = kills

                action = agent.act(state, health, ammo, kills)
                game.make_action(action, args.tics)

                if state is not None and state.screen_buffer is not None:
                    running = renderer.render(
                        state.screen_buffer,
                        health=health,
                        ammo=ammo,
                        kills=kills,
                        episode=episode_num,
                        strategy=agent.strategy,
                    )
                    # AsciiRenderer returns None; treat as True.
                    if running is None:
                        running = True

                # ASCII renderer: pace to target FPS.
                if args.renderer == "ascii":
                    elapsed = time.monotonic() - t0
                    wait    = frame_duration - elapsed
                    if wait > 0:
                        time.sleep(wait)

            if not running:
                break  # user closed window

            # Episode summary.
            total_kills += ep_kills
            ep_time = time.monotonic() - ep_start
            renderer.render_episode_end(
                episode=episode_num,
                kills=ep_kills,
                total_kills=total_kills,
                duration=ep_time,
            )
            time.sleep(2.0)

    except KeyboardInterrupt:
        pass
    finally:
        renderer.show_cursor()
        renderer.close()
        game.close()
        print(f"\ndoom-ascii exited.  Episodes: {episode_num}  "
              f"Total kills: {total_kills}")


if __name__ == "__main__":
    main()
