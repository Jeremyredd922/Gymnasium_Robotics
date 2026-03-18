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


# Doom 2 / Freedoom 2 map progression (MAP01 → MAP32).
DOOM2_MAPS = [f"MAP{i:02d}" for i in range(1, 33)]

# Ordered built-in scenario progression used when freedoom2.wad is absent.
# Each entry: (wad_filename, display_name, kill_threshold)
#   kill_threshold > 0  →  advance after that many kills (continuous-spawn maps)
#   kill_threshold == 0 →  advance on all-enemies-dead timer OR exit trigger
_SCENARIO_LEVELS = [
    ("deadly_corridor.wad",   "Deadly Corridor",   0),   # finite enemies — use stagnation timer
    ("defend_the_line.wad",   "Defend the Line",   20),  # continuous spawns — kill 20 to advance
    ("defend_the_center.wad", "Defend the Center", 30),  # continuous spawns — kill 30 to advance
    ("my_way_home.wad",       "My Way Home",       0),   # navigation — reach the exit
]


def get_level_sequence(scenario_path=None):
    """
    Return [(wad_path, map_name_or_None, display_name, kill_threshold), ...]
    for the full level progression, or [] when a user-supplied --scenario is
    active.

    Priority:
      1. freedoom2.wad  → 32-map Doom 2 campaign (kill_threshold=0 for all)
      2. built-in scenario WADs  → _SCENARIO_LEVELS progression
    """
    if scenario_path:
        return []
    scenarios = vzd.scenarios_path
    # Prefer the full freedoom2 campaign.
    for name in ("freedoom2.wad", "Freedoom2.wad", "FREEDOOM2.WAD"):
        p = os.path.join(scenarios, name)
        if os.path.exists(p):
            return [(p, m, m, 0) for m in DOOM2_MAPS]
    # Fall back to the built-in scenario sequence.
    seq = []
    for filename, display_name, kill_threshold in _SCENARIO_LEVELS:
        p = os.path.join(scenarios, filename)
        if os.path.exists(p):
            seq.append((p, None, display_name, kill_threshold))
    return seq


def change_level(game, wad_path, map_name=None):
    """
    Switch to *wad_path* (and optionally *map_name* within it) without
    recreating the DoomGame object.  All other settings are preserved.
    """
    game.close()
    if map_name:
        game.set_doom_game_path(wad_path)
        game.set_doom_map(map_name)
    else:
        game.set_doom_scenario_path(wad_path)
    game.init()


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


def setup_game(scenario_path=None, map_name=None, color=False):
    """
    Create, configure, and initialize a DoomGame.

    Parameters
    ----------
    scenario_path : str or None
        Path to a .wad file.  None = auto-detect.
    map_name : str or None
        Specific map to load within a multi-map WAD (e.g. "MAP01").
        Ignored when scenario_path is None (auto-detect handles it).
    color : bool
        True  → RGB24 screen format  (for the pygame graphics renderer)
        False → GRAY8 screen format  (for the ASCII renderer)

    Returns
    -------
    game : vzd.DoomGame  (already init'd)
    """
    game = vzd.DoomGame()

    if scenario_path:
        if map_name:
            game.set_doom_game_path(scenario_path)
            game.set_doom_map(map_name)
        else:
            game.set_doom_scenario_path(scenario_path)
    else:
        wad, detected_map, is_doom2 = _find_wad()
        if is_doom2:
            game.set_doom_game_path(wad)
            game.set_doom_map(detected_map)
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

    # Depth buffer — used by AStarAgent for occupancy-grid building.
    game.set_depth_buffer_enabled(True)

    # Labels buffer — exposes per-pixel object tags and bounding boxes.
    # Used by agents to locate enemy hitboxes on screen for precise targeting.
    game.set_labels_buffer_enabled(True)

    # Headless — we handle all display.
    game.set_window_visible(False)

    game.set_mode(vzd.Mode.PLAYER)
    game.set_episode_timeout(0)   # no timeout
    game.set_episode_start_time(1)

    for btn in BUTTONS:
        game.add_available_button(btn)

    for var in (vzd.GameVariable.HEALTH,
                vzd.GameVariable.SELECTED_WEAPON_AMMO,  # gv[1] — ammo for active weapon
                vzd.GameVariable.KILLCOUNT,             # gv[2]
                vzd.GameVariable.POSITION_X,            # gv[3]
                vzd.GameVariable.POSITION_Y,            # gv[4]
                vzd.GameVariable.ANGLE,                 # gv[5]
                vzd.GameVariable.WEAPON1,               # gv[6]  fist
                vzd.GameVariable.WEAPON2,               # gv[7]  pistol
                vzd.GameVariable.WEAPON3,               # gv[8]  shotgun
                vzd.GameVariable.WEAPON4,               # gv[9]  chaingun
                vzd.GameVariable.WEAPON5,               # gv[10] rocket launcher
                vzd.GameVariable.WEAPON6,               # gv[11] plasma rifle
                vzd.GameVariable.WEAPON7,               # gv[12] BFG
                vzd.GameVariable.AMMO0,                 # gv[13] bullets
                vzd.GameVariable.AMMO1,                 # gv[14] shells
                vzd.GameVariable.AMMO2,                 # gv[15] rockets
                vzd.GameVariable.AMMO3):                # gv[16] cells
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


def read_position(state):
    """Return (x, y, angle_degrees) from a game state."""
    if state is None:
        return 0.0, 0.0, 0.0
    gv = state.game_variables
    x     = float(gv[3]) if len(gv) > 3 else 0.0
    y     = float(gv[4]) if len(gv) > 4 else 0.0
    angle = float(gv[5]) if len(gv) > 5 else 0.0
    return x, y, angle


def read_inventory(state):
    """
    Return (weapons, ammo) from a game state.

    weapons : set of int  — weapon slot numbers the player currently holds
                           (3=shotgun, 4=chaingun, 5=RL, 6=plasma, 7=BFG)
    ammo    : dict        — {'bullets', 'shells', 'rockets', 'cells'} counts
    """
    if state is None:
        return set(), {}
    gv = state.game_variables
    weapons = {slot for slot in range(1, 8)
               if len(gv) > 5 + slot and gv[5 + slot]}
    ammo = {
        'bullets': int(gv[13]) if len(gv) > 13 else 0,
        'shells':  int(gv[14]) if len(gv) > 14 else 0,
        'rockets': int(gv[15]) if len(gv) > 15 else 0,
        'cells':   int(gv[16]) if len(gv) > 16 else 0,
    }
    return weapons, ammo


# ZDoom console commands to give each weapon slot (slots 1/2 are always present).
_WEAPON_GIVE = {
    3: "give Shotgun",
    4: "give Chaingun",
    5: "give RocketLauncher",
    6: "give PlasmaRifle",
    7: "give BFG9000",
}


def restore_inventory(game, weapons, ammo):
    """
    Restore a saved inventory at the start of a new episode via ZDoom console
    commands.  Must be called after game.new_episode() while the game is live.
    Commands are queued and executed on the first make_action() call.
    """
    for slot, cmd in _WEAPON_GIVE.items():
        if slot in weapons:
            game.send_game_command(cmd)

    # Restore ammo in whole-pickup increments (give commands respect max caps).
    for _ in range(ammo.get('bullets', 0) // 10):
        game.send_game_command("give Clip")
    for _ in range(ammo.get('shells', 0) // 4):
        game.send_game_command("give Shell")
    for _ in range(ammo.get('rockets', 0)):
        game.send_game_command("give RocketAmmo")
    for _ in range(ammo.get('cells', 0) // 20):
        game.send_game_command("give Cell")


def make_no_op():
    """Return an action with all buttons unpressed."""
    return [False] * N_BUTTONS
