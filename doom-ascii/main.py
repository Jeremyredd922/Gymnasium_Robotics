"""
main.py — doom-ascii entry point.

Choose at startup whether to watch an AI play or control Doom yourself.

Usage:

    python main.py                               # interactive startup menu
    python main.py --mode ai   --agent hpa_star  # AI spectator, skip menu
    python main.py --mode human                  # play yourself, skip menu
    python main.py --mode human --renderer ascii # play in the terminal

Options:
    --mode      {ai,human}                         Play mode (default: menu)
    --renderer  {graphics,ascii}                   Renderer (default: graphics)
    --agent     {flow_field,waypoint,hpa_star,monster}  AI agent (default: hpa_star)
    --episodes  N     Stop after N episodes (0 = infinite)
    --tics      N     ViZDoom tics per step (default: 2)
    --fps       N     ASCII target FPS (default: 15)
    --scale     N     Graphics window scale (default: 3 → 960×720)
    --width     N     ASCII column override
    --height    N     ASCII row override
    --scenario  PATH  Path to a .wad file (default: auto-detect)
"""
import argparse
import math
import sys
import time

from game import (setup_game, read_stats, read_position,
                  read_inventory, restore_inventory,
                  change_level, get_level_sequence)
from agent import build_agent, AGENTS, HumanAgent, _KeyboardReader, _ENEMY_KEYWORDS


# ---------------------------------------------------------------------------
# Enemy-drop system
# ---------------------------------------------------------------------------

_HFOV_RAD     = math.radians(90)   # Doom default horizontal field of view
_PICKUP_RADIUS = 48                 # Doom units — how close to trigger pickup


_DROP_MIN_AGE = 8   # frames a drop must exist before it can be picked up
                    # (~0.5 s at 15 fps) — prevents same-frame instant pickup


class _Drop:
    """A pending item drop sitting at a world-space position."""
    __slots__ = ('x', 'y', 'shells', 'clips', 'stimpack', 'age')

    def __init__(self, x, y, stimpack=False):
        self.x        = x
        self.y        = y
        self.shells   = True   # 4 shells
        self.clips    = True   # 10 bullets
        self.stimpack = stimpack
        self.age      = 0      # frames since creation


def _estimate_world_pos(px, py, player_angle_deg, depth_buf, cx, cy, sw, sh):
    """
    Estimate the world (x, y) of a point at screen pixel (cx, cy) using the
    depth buffer.  Returns None if depth data is unavailable or zero.

    The depth buffer stores perpendicular (z-plane) distance in Doom map units.
    We correct for the horizontal angle offset to get Euclidean distance, then
    project into world space using the player's position and facing angle.
    """
    if depth_buf is None:
        return None
    bx = max(0, min(int(cx), sw - 1))
    by = max(0, min(int(cy), sh - 1))
    z_depth = float(depth_buf[by][bx])
    if z_depth <= 0:
        return None
    angle_off  = ((cx / sw) - 0.5) * _HFOV_RAD
    euc_dist   = z_depth / max(math.cos(angle_off), 0.01)
    angle_rad  = math.radians(player_angle_deg) + angle_off
    return (px + euc_dist * math.cos(angle_rad),
            py + euc_dist * math.sin(angle_rad))


# ---------------------------------------------------------------------------
# ASCII quit watcher — lets the user press 'c' to exit in AI spectator mode
# ---------------------------------------------------------------------------

class _AsciiQuitWatcher:
    """Background keyboard listener for AI ASCII mode — handles c (quit) and r (restart)."""

    def __init__(self):
        self.quit_requested    = False
        self.restart_requested = False
        self._kb               = _KeyboardReader()

    def check(self):
        raw, _ = self._kb.get()
        if raw in (b'c', b'C', b'\x03', b'\x1b'):
            self.quit_requested = True
        elif raw in (b'r', b'R'):
            self.restart_requested = True

    def reset(self):
        self.restart_requested = False

    def stop(self):
        self._kb.stop()


# ---------------------------------------------------------------------------
# Startup menu
# ---------------------------------------------------------------------------

def _pick(options: list, default: int = 1) -> str:
    for i, opt in enumerate(options, 1):
        tag = "  [default]" if i == default else ""
        print(f"    {i}. {opt}{tag}")
    while True:
        raw = input("  > ").strip()
        if raw == "":
            return options[default - 1]
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print(f"  Please enter 1–{len(options)}.")


def _show_controls():
    """Print both control schemes, then wait for Enter."""
    print()
    print("  ── AI Spectator ─────────────────────────────────────")
    print("  C / Ctrl+C      Quit")
    print("  R               Restart episode")
    print("  (graphics mode: ESC or Q also quit)")
    print()
    print("  ── Human Player ─────────────────────────────────────")
    print("  W / S           Move forward / backward")
    print("  A / D           Turn left / right")
    print("  Q / E           Strafe left / right")
    print("  Space           Fire weapon")
    print("  F               Use / open doors")
    print("  C / Ctrl+C      Quit")
    print("  R               Restart episode")
    print()
    input("  Press Enter to return to menu... ")


def show_startup_menu(args):
    """Fill in args.mode / args.renderer / args.agent interactively if not set."""
    # Fully specified on command line — nothing to ask.
    if args.mode and args.renderer and (args.mode == "human" or args.agent):
        return

    print()
    print("=" * 52)
    print("  DOOM ASCII")
    print("=" * 52)

    while not args.mode:
        print("\n  Play mode:")
        choice = _pick(["Watch AI play", "Play yourself", "Controls"], default=1)
        if choice == "Controls":
            _show_controls()
        elif choice == "Watch AI play":
            args.mode = "ai"
        else:
            args.mode = "human"

    if not args.renderer:
        print("\n  Renderer:")
        args.renderer = _pick(["graphics", "ascii"], default=1)

    if args.mode == "ai" and not args.agent:
        print("\n  AI Agent:")
        args.agent = _pick(list(AGENTS.keys()) + ["random"], default=3)   # hpa_star is 3rd

    print()


# ---------------------------------------------------------------------------
# Renderer factory
# ---------------------------------------------------------------------------

def build_renderer(args):
    if args.renderer == "graphics":
        from graphics_renderer import GraphicsRenderer
        return GraphicsRenderer(scale=args.scale)
    from ascii_renderer import AsciiRenderer
    return AsciiRenderer(width=args.width, height=args.height)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="doom-ascii")
    p.add_argument("--mode",     default=None, choices=["ai", "human"])
    p.add_argument("--renderer", default=None, choices=["graphics", "ascii"])
    p.add_argument("--agent",    default=None, choices=list(AGENTS) + ["random"])
    p.add_argument("--episodes", type=int, default=0)
    p.add_argument("--tics",     type=int, default=2)
    p.add_argument("--fps",      type=int, default=15)
    p.add_argument("--scale",    type=int, default=3)
    p.add_argument("--width",    type=int, default=None)
    p.add_argument("--height",   type=int, default=None)
    p.add_argument("--scenario", type=str, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    show_startup_menu(args)

    # Apply defaults for anything the menu didn't fill.
    if not args.mode:     args.mode     = "ai"
    if not args.renderer: args.renderer = "graphics"
    if not args.agent:    args.agent    = "hpa_star"

    is_human = (args.mode == "human")

    renderer = build_renderer(args)
    renderer.show_loading()

    color = (args.renderer == "graphics")

    # Build level sequence before starting the game so we can pass the
    # first level's WAD to setup_game explicitly.
    level_sequence = get_level_sequence(args.scenario)
    level_idx      = 0

    try:
        if level_sequence and not args.scenario:
            first_wad, first_map, *_ = level_sequence[0]
            game = setup_game(scenario_path=first_wad, map_name=first_map, color=color)
        else:
            game = setup_game(scenario_path=args.scenario, color=color)
    except Exception as exc:
        renderer.close()
        sys.stdout.write(f"\nFailed to start ViZDoom: {exc}\n")
        sys.stdout.write("Install with:  pip install vizdoom\n")
        sys.exit(1)

    if is_human:
        agent = build_agent("human", renderer_type=args.renderer)
    else:
        agent = build_agent(args.agent)

    quit_watcher = _AsciiQuitWatcher() if (not is_human and args.renderer == "ascii") else None

    renderer.hide_cursor()
    renderer.show_banner(args.agent if not is_human else "human",
                         player_mode=is_human)
    time.sleep(1.5)

    # Frames of no new kills required to declare a level cleared.
    # At default fps=15, 150 steps ≈ 10 seconds with no enemy activity.
    _CLEAR_FRAMES = 150

    frame_duration   = 1.0 / max(1, args.fps)
    episode_num      = 0
    total_kills      = 0
    carried_weapons  = set()   # weapon slots that persist across levels
    carried_ammo     = {}      # ammo counts that persist across levels

    try:
        while True:
            episode_num += 1
            if args.episodes and episode_num > args.episodes:
                break

            game.new_episode()
            # Restore weapons/ammo carried from the previous level.
            if carried_weapons:
                restore_inventory(game, carried_weapons, carried_ammo)
                carried_weapons = set()
                carried_ammo    = {}
            agent.reset()
            if quit_watcher is not None:
                quit_watcher.reset()
            if hasattr(renderer, 'restart_requested'):
                renderer.restart_requested = False

            kill_threshold = (level_sequence[level_idx][3]
                              if level_sequence else 0)
            ep_start            = time.monotonic()
            ep_kills            = 0
            prev_kills          = 0
            last_valid_state    = None
            running             = True
            restarting          = False
            all_cleared         = False
            last_kills          = 0
            no_kills_change_t   = 0
            # Drop tracking — reset each episode.
            pending_drops    = []          # list of _Drop
            enemy_world_pos  = {}         # {object_id: (wx, wy)}
            prev_label_ids   = set()      # enemy object_ids visible last frame

            while not game.is_episode_finished() and running:
                t0 = time.monotonic()

                try:
                    state = game.get_state()
                    if state is not None:
                        last_valid_state = state
                    health, ammo, kills = read_stats(state)
                    ep_kills = kills

                    # Level-clear detection.
                    # Threshold levels (continuous-spawn): advance once the
                    # player racks up kill_threshold kills.
                    # Stagnation levels (finite enemies): advance once the kill
                    # count hasn't risen for _CLEAR_FRAMES consecutive steps.
                    if kill_threshold > 0:
                        if kills >= kill_threshold:
                            all_cleared = True
                            break
                    else:
                        if kills > last_kills:
                            last_kills        = kills
                            no_kills_change_t = 0
                        elif kills > 0:
                            no_kills_change_t += 1
                            if no_kills_change_t >= _CLEAR_FRAMES:
                                all_cleared = True
                                break

                    # --- Enemy drop tracking -----------------------------------
                    # 1. Age existing drops and check whether the player has
                    #    walked over any that are old enough to be picked up.
                    #    (Pickup check runs BEFORE new drops are created so a
                    #    drop cannot be created and collected in the same frame.)
                    if state is not None and pending_drops:
                        px, py, _ = read_position(state)
                        for drop in pending_drops[:]:
                            drop.age += 1
                            if drop.age >= _DROP_MIN_AGE and \
                                    math.hypot(px - drop.x, py - drop.y) \
                                    <= _PICKUP_RADIUS:
                                if drop.shells:
                                    game.send_game_command("give Shell")
                                if drop.clips:
                                    game.send_game_command("give Clip")
                                if drop.stimpack:
                                    game.send_game_command("give Stimpack")
                                pending_drops.remove(drop)

                    # 2. Update last-known world positions for all visible enemies.
                    cur_label_ids = set()
                    if state is not None and state.labels:
                        buf  = state.screen_buffer
                        sw   = buf.shape[1]
                        sh   = buf.shape[0]
                        px, py, pang = read_position(state)
                        for lbl in state.labels:
                            if not any(kw in lbl.object_name
                                       for kw in _ENEMY_KEYWORDS):
                                continue
                            oid = lbl.object_id
                            cur_label_ids.add(oid)
                            cx = lbl.x + lbl.width  / 2.0
                            cy = lbl.y + lbl.height / 2.0
                            wp = _estimate_world_pos(
                                px, py, pang,
                                state.depth_buffer, cx, cy, sw, sh)
                            if wp is not None:
                                enemy_world_pos[oid] = wp

                    # 3. When kill count rises, place drops at the positions of
                    #    the enemies that just vanished from the labels buffer.
                    if kills > prev_kills:
                        just_died = prev_label_ids - cur_label_ids
                        for oid in sorted(just_died)[:kills - prev_kills]:
                            pos = enemy_world_pos.pop(oid, None)
                            if pos is not None:
                                pending_drops.append(
                                    _Drop(*pos, stimpack=(health < 50)))
                        prev_kills = kills
                    prev_label_ids = cur_label_ids
                    # -----------------------------------------------------------

                    action = agent.act(state, health, ammo, kills)

                    # Check quit — human pressing c/ESC, or AI watcher pressing c.
                    if is_human and isinstance(agent, HumanAgent) and agent.quit_requested:
                        running = False
                        break
                    if quit_watcher is not None:
                        quit_watcher.check()
                        if quit_watcher.quit_requested:
                            running = False
                            break

                    # Check restart — r key from any input source.
                    restart_signalled = (
                        (is_human and isinstance(agent, HumanAgent) and agent.restart_requested)
                        or (quit_watcher is not None and quit_watcher.restart_requested)
                        or getattr(renderer, 'restart_requested', False)
                    )
                    if restart_signalled:
                        restarting = True
                        break

                    game.make_action(action, args.tics)

                    if state is not None and state.screen_buffer is not None:
                        result = renderer.render(
                            state.screen_buffer,
                            health=health,
                            ammo=ammo,
                            kills=kills,
                            episode=episode_num,
                            strategy=agent.strategy,
                            player_mode=is_human,
                        )
                        if result is False:   # pygame window closed / ESC
                            running = False

                except Exception as exc:
                    sys.stdout.write(f"\n[episode {episode_num}] error: {exc}\n")
                    break

                if args.renderer == "ascii":
                    elapsed = time.monotonic() - t0
                    wait    = frame_duration - elapsed
                    if wait > 0:
                        time.sleep(wait)

            if restarting:
                episode_num -= 1   # don't count the restarted episode
                continue

            if not running:
                break

            player_died = game.is_player_dead()
            # Reaching the exit (episode finished, player alive) is a valid
            # completion — advance the level just like kill-all does.
            if game.is_episode_finished() and not player_died:
                all_cleared = True

            ep_time  = time.monotonic() - ep_start
            cur_name = level_sequence[level_idx][2] if level_sequence else None

            if player_died:
                # Wipe inventory — death means starting over from scratch.
                carried_weapons = set()
                carried_ammo    = {}
                total_kills += ep_kills
                restart_name = level_sequence[0][2] if level_sequence else None
                renderer.render_episode_end(
                    episode=episode_num,
                    kills=ep_kills,
                    total_kills=total_kills,
                    duration=ep_time,
                    map_name=cur_name,
                    completed=False,
                    next_map=restart_name,
                )
                time.sleep(2.0)
                if level_sequence and level_idx != 0:
                    level_idx = 0
                    change_level(game, level_sequence[0][0], level_sequence[0][1])
                episode_num -= 1
                continue

            if not all_cleared:
                # Unexpected end — silent restart of same level.
                episode_num -= 1
                continue

            # Level cleared — save inventory then advance to the next level.
            carried_weapons, carried_ammo = read_inventory(last_valid_state)
            total_kills += ep_kills
            next_wad = next_map_id = next_name = None
            if level_sequence:
                level_idx = (level_idx + 1) % len(level_sequence)
                next_wad, next_map_id, next_name, _ = level_sequence[level_idx]

            renderer.render_episode_end(
                episode=episode_num,
                kills=ep_kills,
                total_kills=total_kills,
                duration=ep_time,
                map_name=cur_name,
                completed=True,
                next_map=next_name,
            )
            time.sleep(2.0)

            if next_wad:
                change_level(game, next_wad, next_map_id)

    except KeyboardInterrupt:
        pass
    finally:
        if is_human and isinstance(agent, HumanAgent):
            agent.stop()
        if quit_watcher is not None:
            quit_watcher.stop()
        renderer.show_cursor()
        renderer.close()
        game.close()
        print(f"\ndoom-ascii exited.  Episodes: {episode_num}  "
              f"Total kills: {total_kills}")


if __name__ == "__main__":
    main()
