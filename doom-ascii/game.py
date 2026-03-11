"""
game.py — ViZDoom initialization and configuration for doom-ascii.
"""
import os
import sys

try:
    import vizdoom as vzd
except ImportError:
    print("ViZDoom not found. Install it with:  pip install vizdoom")
    sys.exit(1)


# Ordered list of buttons; indices match action list passed to make_action().
BUTTONS = [
    vzd.Button.MOVE_LEFT,
    vzd.Button.MOVE_RIGHT,
    vzd.Button.MOVE_FORWARD,
    vzd.Button.MOVE_BACKWARD,
    vzd.Button.TURN_LEFT,
    vzd.Button.TURN_RIGHT,
    vzd.Button.ATTACK,
    vzd.Button.USE,
]

# Human-readable names aligned with BUTTONS indices.
BUTTON_NAMES = [
    "MOVE_LEFT", "MOVE_RIGHT", "MOVE_FORWARD", "MOVE_BACKWARD",
    "TURN_LEFT", "TURN_RIGHT", "ATTACK", "USE",
]

# Index constants for building actions.
BTN_MOVE_LEFT     = 0
BTN_MOVE_RIGHT    = 1
BTN_MOVE_FORWARD  = 2
BTN_MOVE_BACKWARD = 3
BTN_TURN_LEFT     = 4
BTN_TURN_RIGHT    = 5
BTN_ATTACK        = 6
BTN_USE           = 7

N_BUTTONS = len(BUTTONS)


def _find_wad():
    """Return (wad_path, map_name, is_doom2) for the best available WAD."""
    scenarios = vzd.scenarios_path

    # Prefer freedoom2 for authentic multi-map Doom 2 experience.
    for name in ("freedoom2.wad", "Freedoom2.wad", "FREEDOOM2.WAD"):
        p = os.path.join(scenarios, name)
        if os.path.exists(p):
            return p, "MAP01", True

    # Built-in research scenarios (single room, still playable).
    for name in ("deadly_corridor.wad", "deathmatch.wad", "basic.wad"):
        p = os.path.join(scenarios, name)
        if os.path.exists(p):
            return p, None, False  # map name handled by scenario file

    raise FileNotFoundError(
        "No suitable WAD found in ViZDoom scenarios directory: " + scenarios
    )


def setup_game(scenario_path=None, color=False):
    """
    Create, configure, and initialize a DoomGame.

    Parameters
    ----------
    scenario_path : str or None
        Path to a .wad file.  None = auto-detect.
    color : bool
        True  → RGB24 screen format  (for the pygame graphics renderer)
        False → GRAY8 screen format  (for the ASCII renderer)

    Returns
    -------
    game : vzd.DoomGame  (already init'd)
    """
    game = vzd.DoomGame()

    if scenario_path:
        wad = scenario_path
        game.set_doom_scenario_path(wad)
    else:
        wad, map_name, is_doom2 = _find_wad()
        if is_doom2:
            game.set_doom_game_path(wad)
            game.set_doom_map(map_name)
        else:
            game.set_doom_scenario_path(wad)

    game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)

    if color:
        # Full RGB for pygame display.
        game.set_screen_format(vzd.ScreenFormat.RGB24)
        # Keep the in-game Doom HUD — it's part of the authentic look.
        game.set_render_hud(True)
    else:
        # Grayscale for ASCII conversion — fewer bytes, faster processing.
        game.set_screen_format(vzd.ScreenFormat.GRAY8)
        # Suppress the in-game HUD; we draw a text HUD ourselves.
        game.set_render_hud(False)

    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_render_decals(False)
    game.set_render_particles(False)

    # Headless — we handle all display.
    game.set_window_visible(False)

    game.set_mode(vzd.Mode.PLAYER)
    game.set_episode_timeout(0)   # no timeout
    game.set_episode_start_time(1)

    for btn in BUTTONS:
        game.add_available_button(btn)

    for var in (vzd.GameVariable.HEALTH,
                vzd.GameVariable.AMMO2,
                vzd.GameVariable.KILLCOUNT,
                vzd.GameVariable.POSITION_X,
                vzd.GameVariable.POSITION_Y,
                vzd.GameVariable.ANGLE):
        game.add_available_game_variable(var)

    game.set_living_reward(0)

    game.init()
    return game


def read_stats(state):
    """Return (health, ammo, kills) from a game state."""
    if state is None:
        return 0, 0, 0
    gv = state.game_variables
    health = int(gv[0]) if len(gv) > 0 else 0
    ammo   = int(gv[1]) if len(gv) > 1 else 0
    kills  = int(gv[2]) if len(gv) > 2 else 0
    return health, ammo, kills


def make_no_op():
    """Return an action with all buttons unpressed."""
    return [False] * N_BUTTONS
